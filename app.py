import os
import ee
import json
import flask
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get environment variables
PORT = int(os.getenv('PORT', 5000))
API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')

# Handle Google credentials from environment variable
credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
if credentials_json:
    # Create a temporary credentials file
    temp_credentials_path = '/tmp/temp_credentials.json'
    with open(temp_credentials_path, 'w') as f:
        f.write(credentials_json)
    
    # Set the path to the credentials file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_path
else:
    # Fall back to the file path if JSON content isn't provided
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'rakamin--kf--analytics-9ae44db8ef2f.json')

# Initialize Earth Engine
try:
    ee.Initialize(project="rakamin--kf--analytics")
    print("Earth Engine initialized successfully")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")

# Optional: API key middleware
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if API_KEY:
            provided_key = request.headers.get('x-api-key')
            if provided_key != API_KEY:
                abort(401, description="Invalid API key")
        return f(*args, **kwargs)
    return decorated_function

# Helper Functions
def get_target_cities():
    """Return the list of target cities in Jabodetabek"""
    return [
        'Bogor', 'Kota Bogor', 'Kota Tangerang', 'Tangerang Selatan', 'Tangerang', 
        'Depok', 'Bekasi', 'Kota Bekasi', 'Jakarta Barat', 'Jakarta Utara', 
        'Jakarta Selatan', 'Jakarta Timur', 'Jakarta Pusat'
    ]

def get_filtered_regions():
    """Filter regions based on target cities"""
    table = ee.FeatureCollection("users/sucinursyifa064/batas_wilayah")
    target_cities = get_target_cities()
    
    filtered_regions = ee.FeatureCollection([])
    for city in target_cities:
        city_filter = table.filter(ee.Filter.eq('NAME_2', city))
        filtered_regions = filtered_regions.merge(city_filter)
    
    return filtered_regions

def extract_data_for_month(year, month):
    """Extract climate data for a specific year and month"""
    # Convert the input month to a number (1-12)
    month_num = int(month)
    if month_num < 1 or month_num > 12:
        raise ValueError("Month must be between 1 and 12")
    
    # Format dates (month is 0-indexed in JS Date)
    start_date = ee.Date.fromYMD(int(year), month_num, 1)
    end_date = start_date.advance(1, 'month')
    
    # Get filtered regions
    filtered_regions = get_filtered_regions()
    
    # Load datasets
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").filterDate(start_date, end_date)
    gsmap = ee.ImageCollection("JAXA/GPM_L3/GSMaP/v6/operational").filterDate(start_date, end_date).select('hourlyPrecipRate')
    modis_temp = ee.ImageCollection("MODIS/061/MOD11A1").filterDate(start_date, end_date).select('LST_Day_1km')
    era5 = ee.ImageCollection("ECMWF/ERA5/MONTHLY").filterDate(start_date, end_date).select(['mean_2m_air_temperature'])
    gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filterDate(start_date, end_date)
    srtm = ee.Image("USGS/SRTMGL1_003")
    modis_ndvi = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(start_date, end_date).select('NDVI')
    worldcover = ee.ImageCollection("ESA/WorldCover/v100").first().select('Map')
    
    # Elevation and slope
    elevation = srtm.select('elevation')
    slope = ee.Terrain.slope(elevation)
    
    # Get rainfall data with fallback mechanism
    def get_rainfall(operation):
        chirps_filtered = chirps.filterDate(start_date, end_date)
        gsmap_filtered = gsmap.filterDate(start_date, end_date)
        
        return ee.Image(ee.Algorithms.If(
            chirps_filtered.size().gt(0),
            chirps_filtered.mean().rename('rainfall') if operation == 'mean' else chirps_filtered.max().rename('rainfall'),
            
            ee.Algorithms.If(
                gsmap_filtered.size().gt(0),
                # GSMaP is mm/hr, multiply by 24 to get mm/day
                gsmap_filtered.mean().multiply(24).rename('rainfall') if operation == 'mean' else gsmap_filtered.max().multiply(24).rename('rainfall'),
                
                ee.Image.constant(0).rename('rainfall')
            )
        ))
    
    avg_rainfall = get_rainfall('mean').rename('avg_rainfall')
    max_rainfall = get_rainfall('max').rename('max_rainfall')
    
    # Temperature data with fallback
    modis_lst_filtered = modis_temp.filterDate(start_date, end_date)
    avg_temperature = ee.Image(ee.Algorithms.If(
        modis_lst_filtered.size().gt(0),
        modis_lst_filtered.mean().multiply(0.02).subtract(273.15).rename('avg_temperature'),
        # If MODIS LST is empty, try ERA5
        ee.Algorithms.If(
            era5.filterDate(start_date, end_date).size().gt(0),
            era5.filterDate(start_date, end_date).mean().select('mean_2m_air_temperature').subtract(273.15).rename('avg_temperature'),
            ee.Image.constant(27).rename('avg_temperature')  # Default value if both are empty
        )
    ))
    
    # Soil moisture - Select a single soil moisture band
    gldas_filtered = gldas.filterDate(start_date, end_date)
    soil_moisture = ee.Image(ee.Algorithms.If(
        gldas_filtered.size().gt(0),
        gldas_filtered.select('SoilMoi0_10cm_inst').mean().rename('soil_moisture'),
        ee.Image.constant(-9999).rename('soil_moisture')
    ))
    
    # NDVI
    ndvi = ee.Image(ee.Algorithms.If(
        modis_ndvi.filterDate(start_date, end_date).size().gt(0),
        modis_ndvi.filterDate(start_date, end_date).mean().select('NDVI').rename('ndvi'),
        ee.Image.constant(5000).rename('ndvi')  # Default 0.5 after conversion
    ))
    
    # Apply scaling factors
    ndvi = ndvi.multiply(0.0001)  # MODIS NDVI scaling factor
    
    # Combine into a single numeric image
    numeric_image = ee.Image.cat([
        avg_rainfall.rename('avg_rainfall'),
        max_rainfall.rename('max_rainfall'),
        avg_temperature.rename('avg_temperature'),
        soil_moisture.rename('soil_moisture'),
        elevation.rename('elevation'),
        slope.rename('slope'),
        ndvi.rename('ndvi')
    ])
    
    # Get statistics per region
    numeric_stats = numeric_image.reduceRegions(
        collection= filtered_regions,
        reducer= ee.Reducer.mean(),
        scale= 1000
    )
    
    # Land cover mode per region
    lc_stats = worldcover.reduceRegions(
        collection= filtered_regions,
        reducer= ee.Reducer.mode(),
        scale= 100
    )
    
    # Define a function to apply to each feature
    def add_landcover_class(feat):
        # Use filter to find matching feature by NAME_3
        lc = lc_stats.filter(ee.Filter.eq('NAME_3', feat.get('NAME_3'))).first()
        
        # Get the mode value and round it to the nearest integer
        lc_val = ee.Number(ee.Algorithms.If(lc, lc.get('mode'), None))
        lc_val_rounded = ee.Number(ee.Algorithms.If(lc_val, lc_val.round(), None))
        
        # Using EE's conditional operations for server-side execution
        lc_class = ee.Algorithms.If(
            ee.Number(lc_val_rounded).eq(10), 'Tree cover',
            ee.Algorithms.If(
                ee.Number(lc_val_rounded).eq(20), 'Shrubland',
                ee.Algorithms.If(
                    ee.Number(lc_val_rounded).eq(30), 'Grassland',
                    ee.Algorithms.If(
                        ee.Number(lc_val_rounded).eq(40), 'Cropland',
                        ee.Algorithms.If(
                            ee.Number(lc_val_rounded).eq(50), 'Built-up',
                            ee.Algorithms.If(
                                ee.Number(lc_val_rounded).eq(60), 'Bare / sparse vegetation',
                                ee.Algorithms.If(
                                    ee.Number(lc_val_rounded).eq(70), 'Snow and ice',
                                    ee.Algorithms.If(
                                        ee.Number(lc_val_rounded).eq(80), 'Permanent water bodies',
                                        ee.Algorithms.If(
                                            ee.Number(lc_val_rounded).eq(90), 'Herbaceous wetland',
                                            ee.Algorithms.If(
                                                ee.Number(lc_val_rounded).eq(95), 'Mangroves',
                                                ee.Algorithms.If(
                                                    ee.Number(lc_val_rounded).eq(100), 'Moss and lichen',
                                                    'Unknown'
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        
        return feat.set({
            'landcover_class': lc_class,
            'year': int(year),
            'month': month_num
        })
    
    # Apply the function to each feature in the collection
    joined = numeric_stats.map(add_landcover_class)
    
    # Get the data as list of dictionaries
    joined_list = joined.getInfo()['features']
    
    # Format the output
    result = []
    for item in joined_list:
        props = item['properties']
        data_point = {
            'NAME_2': props.get('NAME_2', ''),
            'NAME_3': props.get('NAME_3', ''),
            'avg_rainfall': props.get('avg_rainfall', 0),
            'max_rainfall': props.get('max_rainfall', 0),
            'avg_temperature': props.get('avg_temperature', 0),
            'elevation': props.get('elevation', 0),
            'landcover_class': props.get('landcover_class', 'Unknown'),
            'ndvi': props.get('ndvi', 0),
            'slope': props.get('slope', 0),
            'soil_moisture': props.get('soil_moisture', 0),
            'year': int(year),
            'month': month_num
        }
        result.append(data_point)
    
    return result

# API Routes
@app.route('/')
def home():
    return jsonify({
        'name': 'Jabodetabek Climate Data API',
        'description': 'API for climate and flood prediction data for the Greater Jakarta area',
        'endpoints': [
            f'{BASE_URL}/api/data/<year>/<month>',
            f'{BASE_URL}/api/data/<year>'
        ],
        'status': 'online'
    })

@app.route('/api/data/<year>/<month>')
@require_api_key
def get_data_by_month(year, month):
    """Get data for a specific year and month"""
    try:
        data = extract_data_for_month(year, month)
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data),
            'year': int(year),
            'month': int(month)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/<year>')
@require_api_key
def get_data_by_year(year):
    """Get data for all months in a specified year"""
    try:
        year_data = []
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Determine how many months to process based on the year
        months_to_process = 12
        
        # If it's the current year, only process up to the current month
        if int(year) == current_year:
            months_to_process = current_month
        
        for month in range(1, months_to_process + 1):
            try:
                month_data = extract_data_for_month(year, month)
                year_data.extend(month_data)
            except Exception as e:
                print(f"Error processing {year}-{month}: {e}")
                # Continue with the next month even if one fails
                continue
        
        return jsonify({
            'success': True,
            'data': year_data,
            'count': len(year_data),
            'year': int(year)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({
        'success': False,
        'error': 'Unauthorized. Valid API key required.'
    }), 401

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Main entry point
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=(os.getenv('FLASK_DEBUG', 'False').lower() == 'true'))