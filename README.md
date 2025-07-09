# 🌊 GEE Data Extraction for Flood Prediction

This repository contains Python scripts for accessing and processing geospatial datasets from: 
- **Google Earth Engine (GEE)** — for satellite-based environmental data  
- **BNPB Open API** — for historical disaster records in Indonesia.  
The code was developed as part of an exploratory study to support **flood prediction modeling**.

---

## 🌍 Description

These scripts allow extraction of Earth observation datasets such as:

- Precipitation, Land Surface Elevation, Vegetation, and Land Cover via GEE public dataset
- Historical disaster event data from BNPB (e.g., flood occurrences by region & date)

The output is used in downstream tasks like building flood prediction models or identifying vulnerable regions.

> **Note:** This repository contains **data access utilities only** — not model training or final analysis code.

---

## 🛠️ Requirements

Make sure to install required packages:

```bash
pip install -r requirements.txt
```

---

## 🌐 Deployed API

This project includes a live deployment of the data access API.

> 🔗 [http://suciihtisabi-datafloodsight.hf.space/](http://suciihtisabi-datafloodsight.hf.space/)

Use it to fetch flood event data by date and region.  
(*Note: Free-tier deployment — may sleep or be rate-limited.*)

### 📌 Available Endpoints
- Retrieve tabular geospatial data (climate, vegetation, elevation, etc.) for a specific location and month.
```bash
http://gee.up.railway.app/api/data/{year}/{month}?longitude={lng}&latitude={lat}
```
- Generate satellite-based image for a location and year.
```bash
http://gee.up.railway.app/api/imagery/{year}?longitude={lng}&latitude={lat}
```

Example request:
```bash
curl http://gee.up.railway.app/api/data/2024/5?longitude=106.8456&latitude=-6.2088
curl http://gee.up.railway.app/api/imagery/2024?longitude=106.8456&latitude=-6.2088
```

---

## 🧩 Project Affiliation

This repository is part of **FloodSight** — a geospatial flood prediction system developed by  
**Tim CC25-CF096** at **Coding Camp 2025**.

🔗 GitHub Org: [github.com/FloodSight](https://github.com/FloodSight)  
🌐 Frontend App: [fe-flood-sight.vercel.app](https://fe-flood-sight.vercel.app/)

This service provides **environmental feature extraction and flood history data**, which are used as **input for machine learning models** in the FloodSight platform.
