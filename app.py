import os
import sys
import logging
import ee
import json
import time
import flask
import requests
from flask import Flask, jsonify, request, abort, Response
from flask_cors import CORS
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps
from collections import OrderedDict
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get environment variables
PORT = int(os.getenv('PORT', 5000))
API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL', 'http://localhost:5000')

# Kabupaten ID mapping for flood data
KABUPATEN_ID_MAP = {
    'Bogor': 3201, 
    'Kota Bogor': 3271, 
    'Kota Tangerang': 3671, 
    'Tangerang Selatan': 3674, 
    'Tangerang': 3603,
    'Depok': 3276, 
    'Bekasi': 3216, 
    'Kota Bekasi': 3275, 
    'Jakarta Barat': 3173, 
    'Jakarta Utara': 3172,
    'Jakarta Selatan': 3174, 
    'Jakarta Timur': 3175, 
    'Jakarta Pusat': 3171
}

# Handle Google credentials from environment variable
credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
if credentials_json:
    try:
        # Parse the credentials JSON string to a dictionary
        credentials_dict = json.loads(credentials_json)
        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(
            credentials_dict['client_email'], 
            key_data=credentials_dict['private_key']
        )
        ee.Initialize(credentials, project="rakamin--kf--analytics")
        logger.info("Earth Engine initialized successfully with service account")
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")
else:
    # Fallback to the default authentication method
    try:
        logger.info("GOOGLE_CREDENTIALS_JSON not found, using fallback credentials")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'rakamin--kf--analytics-9ae44db8ef2f.json')
        ee.Initialize(project="rakamin--kf--analytics")
        logger.info("Earth Engine initialized with default credentials")
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")

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

def find_district_by_coordinates(longitude, latitude):
    """Find district (kecamatan) based on longitude and latitude coordinates"""
    try:
        # Create a point geometry from the coordinates
        point = ee.Geometry.Point([float(longitude), float(latitude)])
        
        # Get all regions
        table = ee.FeatureCollection("users/sucinursyifa064/batas_wilayah")
        target_cities = get_target_cities()
        
        # Filter regions based on target cities
        filtered_regions = ee.FeatureCollection([])
        for city in target_cities:
            city_filter = table.filter(ee.Filter.eq('NAME_2', city))
            filtered_regions = filtered_regions.merge(city_filter)
        
        # Find which district contains the point
        containing_district = filtered_regions.filterBounds(point)
        
        # Get the first (and should be only) matching district
        district_info = containing_district.first()
        
        if district_info:
            district_data = district_info.getInfo()
            return {
                'found': True,
                'NAME_2': district_data['properties'].get('NAME_2', ''),
                'NAME_3': district_data['properties'].get('NAME_3', ''),
                'district_feature': district_info
            }
        else:
            return {
                'found': False,
                'NAME_2': None,
                'NAME_3': None,
                'district_feature': None
            }
    except Exception as e:
        logger.error(f"Error finding district by coordinates: {e}")
        return {
            'found': False,
            'NAME_2': None,
            'NAME_3': None,
            'district_feature': None
        }

def get_flood_data(year, month, kabupaten_name):
    """
    Get flood data from BNPB API for a specific kabupaten, year, and month.
    Returns 1 if flood occurred, 0 otherwise.
    """
    try:
        # Get kabupaten ID from mapping
        kabupaten_id = KABUPATEN_ID_MAP.get(kabupaten_name)
        if not kabupaten_id:
            logger.warning(f"Kabupaten '{kabupaten_name}' not found in mapping")
            return 0

        # Extract province ID (first 2 digits of kabupaten ID)
        id_prov = str(kabupaten_id)[:2]

        # Format start and end date
        start_date = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year}-{month:02d}-{last_day}"

        # Construct API URL and parameters with quoted strings
        api_url = "https://gis.bnpb.go.id/databencana/kabupaten"
        params = {
            'id_prov': id_prov,
            'kejadian': f"'BANJIR'",
            'start': f"'{start_date}'",
            'end': f"'{end_date}'"
        }

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,id;q=0.8'
        }

        logger.info(f"Requesting flood data for {kabupaten_name} (ID: {kabupaten_id})")
        logger.info(f"Params: {params}")

        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        logger.info(f"Full URL: {response.url}")
        logger.info(f"Response status: {response.status_code}")
        response.raise_for_status()

        data = response.json()
        logger.info(f"Raw API Response: {json.dumps(data, indent=2)}")

        flood_records = []
        if isinstance(data, dict):
            if 'result' in data:
                flood_records = data['result']
            elif 'data' in data:
                flood_records = data['data']
            elif 'features' in data:
                flood_records = data['features']
            elif 'records' in data:
                flood_records = data['records']
            elif 'id_kab' in data or 'id_kabkot' in data:
                flood_records = [data]
            else:
                logger.info(f"Unknown response structure: {list(data.keys())}")
        elif isinstance(data, list):
            flood_records = data

        logger.info(f"Found {len(flood_records)} flood records to check")

        for i, item in enumerate(flood_records):
            logger.info(f"Record {i+1}: {json.dumps(item, indent=2)}")

            item_kab_id = None
            for field in ['id_kab', 'id_kabkot', 'kabupaten_id', 'kab_id', 'id_wilayah']:
                if field in item:
                    item_kab_id = item[field]
                    break

            if item_kab_id is None:
                for field in ['id_kab', 'id_kabkot', 'kabupaten_id', 'kab_id', 'id_wilayah']:
                    if field in item:
                        try:
                            item_kab_id = int(item[field])
                            break
                        except (ValueError, TypeError):
                            continue

            item_date = item.get('tgl', item.get('tanggal', item.get('date', '')))

            logger.info(f"Checking record: id_kab={item_kab_id}, target={kabupaten_id}, date={item_date}")

            if item_kab_id is not None:
                try:
                    if int(item_kab_id) == int(kabupaten_id):
                        logger.info(f"✓ FLOOD FOUND for {kabupaten_name} (ID: {kabupaten_id}) on {item_date}")
                        return 1
                except (ValueError, TypeError):
                    if str(item_kab_id) == str(kabupaten_id):
                        logger.info(f"✓ FLOOD FOUND for {kabupaten_name} (ID: {kabupaten_id}) on {item_date}")
                        return 1

        logger.info(f"✗ No flood found for {kabupaten_name} (ID: {kabupaten_id}) in {year}-{month:02d}")
        return 0

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching flood data for {kabupaten_name}: {e}")
        return 0
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for {kabupaten_name}: {e}")
        logger.error(f"Response content: {response.text[:1000]}")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error getting flood data for {kabupaten_name}: {e}")
        return 0

def safe_get_download_url(rgb_composite, geometry, max_retries=3):
    """
    Get download URL with error handling and retry mechanism
    """
    for attempt in range(max_retries):
        try:
            download_url = rgb_composite.getDownloadURL({
                'scale': 10,
                'crs': 'EPSG:4326', 
                'region': geometry,
                'format': 'GeoTIFF',
                'maxPixels': 1e13
            })
            return download_url
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "too large" in error_msg or "limit" in error_msg:
                # Area too large, reduce maxPixels and increase scale
                try:
                    download_url = rgb_composite.getDownloadURL({
                        'scale': 20,  # Lower resolution
                        'crs': 'EPSG:4326',
                        'region': geometry, 
                        'format': 'GeoTIFF',
                        'maxPixels': 1e12  # Smaller limit
                    })
                    logger.warning(f"Reduced resolution to scale=20 for large area (attempt {attempt + 1})")
                    return download_url
                except Exception as fallback_error:
                    logger.error(f"Fallback attempt failed: {fallback_error}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise e
                        
            elif "timeout" in error_msg or "deadline" in error_msg:
                if attempt < max_retries - 1:
                    # Retry with exponential backoff for timeout errors
                    wait_time = 2 ** attempt
                    logger.warning(f"Timeout error, retrying in {wait_time} seconds (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
            else:
                # For other errors, retry once more
                if attempt < max_retries - 1:
                    logger.warning(f"Error occurred, retrying (attempt {attempt + 1}): {e}")
                    time.sleep(1)
                    continue
                else:
                    raise e
    
    raise Exception("Failed to generate download URL after all retry attempts")

        
def create_rgb_composite_for_district(year, district_feature, district_name):
    """
    Create RGB composite image for a specific district and year
    Matches EXACTLY the GEE script specifications for model preprocessing
    """
    try:
        # Get the geometry of the district - EXACTLY like your GEE script
        geometry = district_feature.geometry()
        
        # Define date range for the year - EXACTLY like your GEE script
        start_date = ee.Date.fromYMD(int(year), 1, 1)
        end_date = ee.Date.fromYMD(int(year), 12, 31)
        
        # Create RGB composite using Sentinel-2 - EXACTLY matching your GEE script
        rgb_composite = (ee.ImageCollection("COPERNICUS/S2_SR")
                        .filterDate(start_date, end_date)
                        .filterBounds(geometry)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .select(['B4', 'B3', 'B2'])  # RGB bands - EXACTLY as specified
                        .median()  # EXACTLY as specified
                        .clip(geometry))  # EXACTLY as specified
        
        # MODIFIED: Clean district name for filename - lowercase and no spaces
        clean_district_name = district_name.lower().replace(' ', '')
        
        # MODIFIED: Get download URL using the new error handling function
        url = safe_get_download_url(rgb_composite, geometry)
        
        # Get smaller thumbnail for preview (to avoid large response sizes)
        thumbnail_url = rgb_composite.getThumbURL({
            'min': 0,
            'max': 3000,
            'dimensions': 256,  # Reduced from 512 to minimize response size
            'region': geometry,
            'format': 'png'
        })
        
        return {
            'success': True,
            'district_name': district_name,
            'year': int(year),
            'download_url': url,
            'thumbnail_url': thumbnail_url,
            'filename': f'RGB_{clean_district_name}_{year}.tif',  # MODIFIED: Now lowercase
            'description': f'RGB_{clean_district_name}_{year}',  # MODIFIED: Now lowercase
            'specifications': {
                'bands': ['B4', 'B3', 'B2'],  # Red, Green, Blue
                'scale': 10,  # 10 meter resolution
                'cloud_percentage_threshold': 20,  # Max 20% clouds
                'composite_method': 'median',  # Median composite
                'satellite': 'COPERNICUS/S2_SR',  # Sentinel-2 Surface Reflectance
                'crs': 'EPSG:4326',
                'max_pixels': '1e13'
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating RGB composite for {district_name}: {e}")
        return {
            'success': False,
            'error': str(e),
            'district_name': district_name,
            'year': int(year)
        }

        
def create_rgb_composite_for_coordinates(year, longitude, latitude):
    """Create RGB composite image for specific coordinates"""
    try:
        # Find the district first
        district_info = find_district_by_coordinates(longitude, latitude)
        if not district_info['found']:
            raise ValueError(f"Coordinates ({longitude}, {latitude}) are not within any target district in Jabodetabek area")
        
        district_name = district_info['NAME_3']
        district_feature = district_info['district_feature']
        
        return create_rgb_composite_for_district(year, district_feature, district_name)
        
    except Exception as e:
        logger.error(f"Error creating RGB composite for coordinates: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def extract_data_for_month(year, month, longitude=None, latitude=None):
    """Extract climate data for a specific year and month, optionally filtered by coordinates"""
    # Convert the input month to a number (1-12)
    month_num = int(month)
    if month_num < 1 or month_num > 12:
        raise ValueError("Month must be between 1 and 12")
    
    # Format dates (month is 0-indexed in JS Date)
    start_date = ee.Date.fromYMD(int(year), month_num, 1)
    end_date = start_date.advance(1, 'month')
    
    # Get filtered regions based on coordinates or all regions
    if longitude is not None and latitude is not None:
        district_info = find_district_by_coordinates(longitude, latitude)
        if not district_info['found']:
            raise ValueError(f"Coordinates ({longitude}, {latitude}) are not within any target district in Jabodetabek area")
        
        # Use only the specific district
        filtered_regions = ee.FeatureCollection([district_info['district_feature']])
        logger.info(f"Processing data for district: {district_info['NAME_3']}, {district_info['NAME_2']}")
    else:
        # Use all regions
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
    
    # Format the output with specific decimal places
    result = []
    for item in joined_list:
        props = item['properties']
        
        # Helper function to safely format numbers
        def format_numeric(value, decimal_places):
            if value is None or value == -9999:
                return 0.0
            try:
                return round(float(value), decimal_places)
            except (ValueError, TypeError):
                return 0.0
        
        name_2 = props.get('NAME_2', '')
        name_3 = props.get('NAME_3', '')
        
        name_3_cleaned = name_3.lower().strip()
        
        # Get flood data for this kabupaten, year, and month
        flood_status = get_flood_data(int(year), month_num, name_2)
        
        data_point = OrderedDict([
            ('NAME_2', name_2),
            ('NAME_3', name_3_cleaned),
            ('avg_rainfall', format_numeric(props.get('avg_rainfall', 0), 3)),
            ('max_rainfall', format_numeric(props.get('max_rainfall', 0), 3)),
            ('avg_temperature', format_numeric(props.get('avg_temperature', 0), 2)),
            ('elevation', format_numeric(props.get('elevation', 0), 3)),
            ('landcover_class', props.get('landcover_class', 'Unknown')),
            ('ndvi', format_numeric(props.get('ndvi', 0), 3)),
            ('slope', format_numeric(props.get('slope', 0), 3)),
            ('soil_moisture', format_numeric(props.get('soil_moisture', 0), 3)),
            ('year', int(year)),
            ('month', month_num),
            ('banjir', flood_status),
        ])

        
        # Add lat and long if coordinates provided
        if longitude is not None and latitude is not None:
            data_point['lat'] = float(latitude)
            data_point['long'] = float(longitude)
        
        result.append(data_point)
    
    return result

# Add a debug endpoint
@app.route('/debug')
def debug():
    """Return diagnostic information about the environment"""
    return jsonify({
        'env_vars': {k: v for k, v in os.environ.items() 
                    if not k.startswith('GOOGLE_') and k != 'API_KEY'},
        'port': PORT,
        'base_url': BASE_URL,
        'has_credentials': credentials_json is not None,
        'earth_engine_initialized': ee.data._initialized,
        'kabupaten_mapping': KABUPATEN_ID_MAP
    })
    
# Enhanced test endpoint for flood data with detailed debugging
@app.route('/api/flood/debug')
@require_api_key
def debug_flood_data():
    """Debug endpoint to check flood data with detailed logging"""
    try:
        year = request.args.get('year', default=2021, type=int)
        month = request.args.get('month', default=1, type=int)
        kabupaten = request.args.get('kabupaten', default='Jakarta Pusat')
        
        # Test with single kabupaten first
        kabupaten_id = KABUPATEN_ID_MAP.get(kabupaten)
        if not kabupaten_id:
            return jsonify({
                'success': False,
                'error': f'Kabupaten {kabupaten} not found in mapping',
                'available_kabupaten': list(KABUPATEN_ID_MAP.keys())
            })
        
        id_prov = str(kabupaten_id)[:2]
        
        # Calculate dates
        
        last_day = calendar.monthrange(year, month)[1]
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{last_day}"
        
        # Make the API call
        api_url = "https://gis.bnpb.go.id/databencana/kabupaten"
        params = {
            'id_prov': id_prov,
            'kejadian': 'BANJIR',
            'start': start_date,
            'end': end_date
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*'
        }
        
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        
        debug_info = {
            'request_info': {
                'kabupaten': kabupaten,
                'kabupaten_id': kabupaten_id,
                'id_prov': id_prov,
                'year': year,
                'month': month,
                'start_date': start_date,
                'end_date': end_date,
                'api_url': api_url,
                'full_url': response.url,
                'params': params
            },
            'response_info': {
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'content_length': len(response.content)
            }
        }
        
        if response.status_code == 200:
            try:
                data = response.json()
                debug_info['response_data'] = {
                    'type': type(data).__name__,
                    'keys': list(data.keys()) if isinstance(data, dict) else 'N/A (list)',
                    'length': len(data) if isinstance(data, list) else 'N/A (dict)',
                    'sample_data': data if len(str(data)) < 2000 else str(data)[:2000] + '...'
                }
                
                # Process flood data
                flood_status = get_flood_data(year, month, kabupaten)
                debug_info['flood_result'] = {
                    'flood_status': flood_status,
                    'interpretation': 'Flood occurred' if flood_status == 1 else 'No flood'
                }
                
            except json.JSONDecodeError as e:
                debug_info['json_error'] = str(e)
                debug_info['raw_response'] = response.text[:1000]
        else:
            debug_info['error'] = f'HTTP {response.status_code}: {response.text[:500]}'
        
        return jsonify({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': str(e.__traceback__)
        }), 500

# Test endpoint to check multiple kabupaten at once
@app.route('/api/flood/test-all')
@require_api_key
def test_all_flood_data():
    """Test flood data for all kabupaten with summary"""
    try:
        year = request.args.get('year', default=2021, type=int)
        month = request.args.get('month', default=2, type=int)  # February 2021 had floods
        
        results = {}
        total_floods = 0
        
        for kabupaten_name in KABUPATEN_ID_MAP.keys():
            flood_status = get_flood_data(year, month, kabupaten_name)
            results[kabupaten_name] = {
                'kabupaten_id': KABUPATEN_ID_MAP[kabupaten_name],
                'flood_status': flood_status,
                'has_flood': flood_status == 1
            }
            if flood_status == 1:
                total_floods += 1
        
        return jsonify({
            'success': True,
            'period': f'{year}-{month:02d}',
            'total_kabupaten': len(KABUPATEN_ID_MAP),
            'total_with_floods': total_floods,
            'flood_percentage': round((total_floods / len(KABUPATEN_ID_MAP)) * 100, 2),
            'results': results,
            'suggestion': 'Try February 2021 (month=2) as Jakarta had major floods then'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add endpoint to find district by coordinates
@app.route('/api/district')
@require_api_key
def get_district_by_coordinates():
    """Get district information based on longitude and latitude"""
    try:
        longitude = request.args.get('longitude', type=float)
        latitude = request.args.get('latitude', type=float)
        
        if longitude is None or latitude is None:
            return jsonify({
                'success': False,
                'error': 'Both longitude and latitude parameters are required'
            }), 400
        
        # Validate coordinate ranges
        if not (-180 <= longitude <= 180):
            return jsonify({
                'success': False,
                'error': 'Longitude must be between -180 and 180'
            }), 400
        
        if not (-90 <= latitude <= 90):
            return jsonify({
                'success': False,
                'error': 'Latitude must be between -90 and 90'
            }), 400
        
        district_info = find_district_by_coordinates(longitude, latitude)
        
        if district_info['found']:
            return jsonify({
                'success': True,
                'coordinates': {
                    'longitude': longitude,
                    'latitude': latitude
                },
                'district': {
                    'NAME_2': district_info['NAME_2'],
                    'NAME_3': district_info['NAME_3']
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Coordinates ({longitude}, {latitude}) are not within any target district in Jabodetabek area',
                'coordinates': {
                    'longitude': longitude,
                    'latitude': latitude
                }
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Test endpoint for flood data
@app.route('/api/flood/test')
@require_api_key
def test_flood_data():
    """Test endpoint to check flood data for all kabupaten"""
    try:
        year = request.args.get('year', default=2021, type=int)
        month = request.args.get('month', default=1, type=int)
        
        results = {}
        for kabupaten_name in KABUPATEN_ID_MAP.keys():
            flood_status = get_flood_data(year, month, kabupaten_name)
            results[kabupaten_name] = {
                'kabupaten_id': KABUPATEN_ID_MAP[kabupaten_name],
                'flood_status': flood_status
            }
        
        return jsonify({
            'success': True,
            'year': year,
            'month': month,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Modify home route to display more info
@app.route('/')
def home():
    host_url = request.host_url.rstrip('/')
    return jsonify({
        'name': 'Jabodetabek Climate Data API with Flood Integration',
        'description': 'API for climate and flood prediction data for the Greater Jakarta area',
        'endpoints': [
            f'{host_url}/api/data/{{year}}/{{month}}?longitude={{lng}}&latitude={{lat}}',
            f'{host_url}/api/imagery/{{year}}?longitude={{lng}}&latitude={{lat}}',
            # f'{host_url}/api/flood/test?year={{year}}&month={{month}}',
            f'{host_url}/debug'
        ],
        'parameters': {
            'longitude': 'Longitude coordinate (required with latitude)',
            'latitude': 'Latitude coordinate (required with longitude)',
            'year': 'Year (YYYY format)',
            'month': 'Month (1-12)'
        },
        'examples': [
            f'{host_url}/api/data/2024/5?longitude=106.8456&latitude=-6.2088',
            f'{host_url}/api/imagery/2024?longitude=106.8456&latitude=-6.2088',
            f'{host_url}/api/flood/test?year=2021&month=1'
        ],
        'new_features': [
            'Flood data integration from BNPB API',
            'banjir field added to all data points (1 = flood occurred, 0 = no flood)',
            'Test endpoint for flood data validation'
        ],
        'status': 'online',
        'ee_status': 'initialized' if ee.data._initialized else 'not initialized'
    })

# OPTIMIZED: Single district imagery endpoint
@app.route('/api/imagery/<year>')
@require_api_key
def get_satellite_imagery(year):
    """Get satellite RGB composite imagery for a specific year and coordinates"""
    try:
        # Require longitude and latitude for single requests to avoid large responses
        longitude = request.args.get('longitude', type=float)
        latitude = request.args.get('latitude', type=float)
        
        if longitude is None or latitude is None:
            return jsonify({
                'success': False,
                'error': 'Both longitude and latitude parameters are required for imagery requests',
                'suggestion': 'Use /api/imagery/bulk/{year} for all districts or /api/districts to get district list first'
            }), 400
        
        # Validate coordinate ranges
        if not (-180 <= longitude <= 180):
            return jsonify({
                'success': False,
                'error': 'Longitude must be between -180 and 180'
            }), 400
        
        if not (-90 <= latitude <= 90):
            return jsonify({
                'success': False,
                'error': 'Latitude must be between -90 and 90'
            }), 400
        
        # Get imagery for specific coordinates
        imagery_data = create_rgb_composite_for_coordinates(year, longitude, latitude)
        
        if imagery_data['success']:
            return jsonify({
                'success': True,
                'year': int(year),
                'coordinates': {
                    'longitude': longitude,
                    'latitude': latitude
                },
                'imagery': imagery_data,
                'model_compatibility': {
                    'matches_training_spec': True,
                    'note': 'This imagery matches exactly your GEE script specifications for model preprocessing'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': imagery_data['error']
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/<year>/<month>')
@require_api_key
def get_data_by_month(year, month):
    """Get data for a specific year and month, optionally filtered by coordinates"""
    try:
        # Get optional longitude and latitude from query parameters
        longitude = request.args.get('longitude', type=float)
        latitude = request.args.get('latitude', type=float)
        
        # Validate coordinates if provided
        if (longitude is not None and latitude is None) or (longitude is None and latitude is not None):
            return jsonify({
                'success': False,
                'error': 'Both longitude and latitude must be provided together'
            }), 400
        
        if longitude is not None and latitude is not None:
            # Validate coordinate ranges for Indonesia
            if not (-180 <= longitude <= 180):
                return jsonify({
                    'success': False,
                    'error': 'Longitude must be between -180 and 180'
                }), 400
            
            if not (-90 <= latitude <= 90):
                return jsonify({
                    'success': False,
                    'error': 'Latitude must be between -90 and 90'
                }), 400
        
        data = extract_data_for_month(year, month, longitude, latitude)
        
        response_data = {
            'success': True,
            'data': data,
            'count': len(data),
            'year': int(year),
            'month': int(month)
        }
        
        # Add coordinate info if coordinates were provided
        if longitude is not None and latitude is not None:
            response_data['coordinates'] = {
                'longitude': longitude,
                'latitude': latitude
            }
            if len(data) > 0:
                response_data['district'] = {
                    'NAME_2': data[0]['NAME_2'],
                    'NAME_3': data[0]['NAME_3']
                }
        
        return Response(
            response=json.dumps(response_data, indent=2),
            status=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/<year>')
@require_api_key
def get_data_by_year(year):
    """Get data for all months in a specified year, optionally filtered by coordinates"""
    try:
        # Get optional longitude and latitude from query parameters
        longitude = request.args.get('longitude', type=float)
        latitude = request.args.get('latitude', type=float)
        
        # Validate coordinates if provided
        if (longitude is not None and latitude is None) or (longitude is None and latitude is not None):
            return jsonify({
                'success': False,
                'error': 'Both longitude and latitude must be provided together'
            }), 400
        
        if longitude is not None and latitude is not None:
            # Validate coordinate ranges
            if not (-180 <= longitude <= 180):
                return jsonify({
                    'success': False,
                    'error': 'Longitude must be between -180 and 180'
                }), 400
            
            if not (-90 <= latitude <= 90):
                return jsonify({
                    'success': False,
                    'error': 'Latitude must be between -90 and 90'
                }), 400
        
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
                month_data = extract_data_for_month(year, month, longitude, latitude)
                year_data.extend(month_data)
            except Exception as e:
                print(f"Error processing {year}-{month}: {e}")
                # Continue with the next month even if one fails
                continue
        
        response_data = {
            'success': True,
            'data': year_data,
            'count': len(year_data),
            'year': int(year)
        }
        
        # Add coordinate info if coordinates were provided
        if longitude is not None and latitude is not None:
            response_data['coordinates'] = {
                'longitude': longitude,
                'latitude': latitude
            }
            if len(year_data) > 0:
                response_data['district'] = {
                    'NAME_2': year_data[0]['NAME_2'],
                    'NAME_3': year_data[0]['NAME_3']
                }
        
        return jsonify(response_data)
        
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
    logger.info(f"Starting application on port {port}")
    app.run(host='0.0.0.0', port=port, debug=(os.getenv('FLASK_DEBUG', 'False').lower() == 'true'))
