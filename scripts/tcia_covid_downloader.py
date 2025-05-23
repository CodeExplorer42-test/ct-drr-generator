#!/usr/bin/env python3
"""
TCIA COVID-19 CT Dataset Downloader
Downloads chest CT data from COVID19-CT-dataset collection for stereo DRR generation
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from tcia_utils import nbia

# Collection details
COLLECTION_NAME = "COVID-19-NY-SBU"  # Has 458 CT series
OUTPUT_DIR = "data/tciaDownload"
METADATA_DIR = "data"
LOG_FILE = "logs/covid_ct_download_log.txt"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def get_collection_info():
    """Get information about the COVID-19 CT collection"""
    log_message("Fetching COVID-19 CT collection information...")
    
    try:
        # Get collection details
        collections = nbia.getCollections()
        covid_collections = [c for c in collections if "COVID" in c.get("Collection", "").upper()]
        
        log_message(f"Found {len(covid_collections)} COVID-related collections")
        for col in covid_collections:
            log_message(f"  - {col.get('Collection', 'Unknown')}")
        
        return covid_collections
    except Exception as e:
        log_message(f"Error fetching collection info: {e}")
        return []

def get_patient_studies():
    """Get patient and study information for the collection"""
    log_message(f"Fetching patient studies for {COLLECTION_NAME}...")
    
    try:
        # Get series information directly (without going through patients first)
        series_df = nbia.getSeries(collection=COLLECTION_NAME, modality="CT", format="df")
        
        if series_df.empty:
            log_message("No CT series found in collection")
            return []
        
        log_message(f"Found {len(series_df)} CT series in collection")
        
        # Filter for chest CT series
        chest_indicators = ['CHEST', 'THORAX', 'LUNG', 'COVID']
        chest_series_df = series_df
        
        # Try to filter by body part or description
        for col in ['BodyPartExamined', 'SeriesDescription']:
            if col in series_df.columns:
                mask = series_df[col].str.contains('|'.join(chest_indicators), case=False, na=False)
                if mask.any():
                    chest_series_df = series_df[mask]
                    log_message(f"Filtered to {len(chest_series_df)} chest-specific series")
                    break
        
        # Convert DataFrame to list of dicts for compatibility
        all_series = chest_series_df.to_dict('records')
        
        # Log some sample series
        for i, series in enumerate(all_series[:5]):
            log_message(f"  Series {i+1}: {series.get('SeriesInstanceUID', 'Unknown')[:30]}... "
                       f"({series.get('ImageCount', 'Unknown')} images)")
        
        return all_series
    except Exception as e:
        log_message(f"Error fetching patient studies: {e}")
        return []

def analyze_series_metadata(series_list):
    """Analyze series to find suitable candidates for DRR"""
    log_message("Analyzing series metadata for DRR suitability...")
    
    suitable_series = []
    
    for series in series_list:
        series_uid = series.get("SeriesInstanceUID", "")
        slice_count = int(series.get("ImageCount", 0))
        manufacturer = series.get("Manufacturer", "Unknown")
        slice_thickness = series.get("SliceThickness", "Unknown")
        
        # Look for series with enough slices for good volume reconstruction
        if slice_count >= 100:
            suitable_series.append({
                "SeriesInstanceUID": series_uid,
                "PatientID": series.get("PatientID", ""),
                "StudyDate": series.get("StudyDate", ""),
                "SliceCount": slice_count,
                "SliceThickness": slice_thickness,
                "Manufacturer": manufacturer,
                "BodyPart": series.get("BodyPartExamined", ""),
                "SeriesDescription": series.get("SeriesDescription", "")
            })
            
            log_message(f"  Suitable: {series_uid[:30]}... ({slice_count} slices, {slice_thickness}mm)")
    
    log_message(f"Found {len(suitable_series)} suitable series")
    return suitable_series

def save_metadata(series_list):
    """Save metadata to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{METADATA_DIR}/covid_ct_metadata_{timestamp}.csv"
    
    os.makedirs(METADATA_DIR, exist_ok=True)
    
    df = pd.DataFrame(series_list)
    df.to_csv(filename, index=False)
    log_message(f"Metadata saved to {filename}")
    
    return filename

def download_series(series_data, patient_id):
    """Download a specific series"""
    series_uid = series_data['SeriesInstanceUID']
    log_message(f"Downloading series {series_uid} for patient {patient_id}")
    
    try:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create dataframe with the series to download
        series_df = pd.DataFrame([series_data])
        
        # Download the series
        df_result = nbia.downloadSeries(
            series_data=series_df, 
            input_type="df", 
            number=1, 
            format="csv"  # Save CSV with metadata
        )
        
        log_message(f"  Download complete!")
        log_message(f"  Files saved to: {OUTPUT_DIR}")
        return True
        
    except Exception as e:
        log_message(f"  Error downloading series: {e}")
        return False

def main():
    """Main execution function"""
    log_message("=== TCIA COVID-19 CT Dataset Downloader Started ===")
    
    # Step 1: Get collection info
    collections = get_collection_info()
    
    # Step 2: Get patient studies
    series_list = get_patient_studies()
    
    if not series_list:
        log_message("No series found. Exiting.")
        return
    
    # Step 3: Analyze metadata
    suitable_series = analyze_series_metadata(series_list)
    
    if not suitable_series:
        log_message("No suitable series found. Exiting.")
        return
    
    # Step 4: Save metadata
    metadata_file = save_metadata(suitable_series)
    
    # Step 5: Ask user to select a series
    log_message("\nSuitable series for stereo DRR:")
    for i, series in enumerate(suitable_series[:5]):  # Show top 5
        log_message(f"{i+1}. Patient: {series['PatientID']}, "
                   f"Slices: {series['SliceCount']}, "
                   f"Thickness: {series['SliceThickness']}mm")
    
    # For automation, select the first suitable series
    if suitable_series:
        selected = suitable_series[0]
        log_message(f"\nAuto-selecting first series: {selected['SeriesInstanceUID']}")
        
        # Download the selected series
        success = download_series(selected, selected['PatientID'])
        
        if success:
            log_message(f"\nDownload complete! Series saved to: {OUTPUT_DIR}/{selected['SeriesInstanceUID']}")
            log_message(f"Next step: Run extract_drr_metadata.py to verify geometric completeness")
        else:
            log_message("Download failed.")
    
    log_message("=== TCIA COVID-19 CT Dataset Downloader Completed ===")

if __name__ == "__main__":
    main()