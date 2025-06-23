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
