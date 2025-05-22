#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRR-Ready Chest CT Downloader from TCIA

This script finds and downloads chest CT data suitable for DRR generation,
focusing on collections with complete DICOM geometric metadata.
"""

import sys
import requests
import pandas as pd
import subprocess

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-q", "tcia_utils"])
        print("✓ tcia_utils installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing tcia_utils: {e}")

# Install and import
install_requirements()

try:
    from tcia_utils import nbia
    print("✓ tcia_utils imported successfully")
except ImportError as e:
    print(f"❌ Error importing tcia_utils: {e}")
    print("Please run: pip install tcia_utils")
    sys.exit(1)

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

def find_drr_suitable_collections():
    """Find collections suitable for DRR generation"""
    print("🔍 Finding collections suitable for DRR generation...")
    
    # Collections known to have good geometric metadata for DRR
    drr_priority_collections = [
        'NSCLC-Radiomics',           # Radiomics data typically has complete metadata
        'RIDER Lung CT',             # RIDER datasets are well-curated
        'LungCT-Diagnosis',          # Diagnostic CTs have proper geometry
        'NSCLC-Radiomics-Genomics',  # Another radiomics dataset
        'QIN LUNG CT',               # QIN datasets are well-standardized
        'LIDC-IDRI'                  # Well-known lung screening dataset
    ]
    
    # Get all available collections
    all_collections = nbia.getCollections(format="df")
    available_collections = all_collections['Collection'].tolist()
    
    # Find which priority collections are available
    suitable_collections = []
    for collection in drr_priority_collections:
        if collection in available_collections:
            suitable_collections.append(collection)
            print(f"✓ Found: {collection}")
    
    return suitable_collections

def analyze_collection_for_drr(collection_name):
    """Analyze if a collection has the metadata needed for DRR"""
    print(f"\n📊 Analyzing {collection_name} for DRR suitability...")
    
    # Check modalities
    modalities = nbia.getModalityCounts(collection=collection_name, format="df")
    print("Available modalities:")
    print(modalities)
    
    if 'CT' not in modalities['criteria'].values:
        print("❌ No CT data found")
        return False, None
    
    # Get sample series to check metadata completeness
    print(f"\n🔍 Getting sample CT series from {collection_name}...")
    series_data = nbia.getSeries(collection=collection_name, modality="CT", format="df")
    
    if series_data.empty:
        print("❌ No CT series found")
        return False, None
    
    print(f"✓ Found {len(series_data)} CT series")
    
    # Check for chest/lung specific data
    chest_indicators = ['CHEST', 'THORAX', 'LUNG']
    chest_series = series_data
    
    for col in ['BodyPartExamined', 'SeriesDescription']:
        if col in series_data.columns:
            mask = series_data[col].str.contains('|'.join(chest_indicators), case=False, na=False)
            if mask.any():
                chest_series = series_data[mask]
                print(f"✓ Found {len(chest_series)} chest-specific series")
                break
    
    # Display sample metadata
    print(f"\n📋 Sample series metadata:")
    display_cols = ['PatientID', 'SeriesDescription', 'BodyPartExamined', 'SliceThickness', 'PixelSpacing']
    available_cols = [col for col in display_cols if col in chest_series.columns]
    print(chest_series[available_cols].head(3))
    
    return True, chest_series

def download_drr_sample(series_data, collection_name, num_series=1):
    """Download a sample suitable for DRR generation"""
    print(f"\n⬇️ Downloading {num_series} sample from {collection_name} for DRR analysis...")
    
    try:
        # Download the series
        df_result = nbia.downloadSeries(
            series_data, 
            input_type="df", 
            number=num_series, 
            format="csv"  # Save CSV with metadata
        )
        
        print("✅ Download completed successfully!")
        print("\n📁 Downloaded files structure:")
        print("  └── tciaDownload/")
        print("      ├── [SeriesInstanceUID]/  # DICOM files")
        print("      └── series_metadata.csv   # Geometric metadata")
        
        print("\n📋 Downloaded series details:")
        display_cols = ['PatientID', 'SeriesInstanceUID', 'SeriesDescription', 'Modality', 'SliceThickness']
        available_cols = [col for col in display_cols if col in df_result.columns]
        print(df_result[available_cols])
        
        # Provide DRR-specific guidance
        print("\n🎯 DRR Generation Notes:")
        print("  • DICOM files contain geometric metadata in headers")
        print("  • Key tags for DRR: (0018,1110) Distance Source to Detector")
        print("  • (0018,1111) Distance Source to Patient")
        print("  • (0028,0030) Pixel Spacing")
        print("  • (0018,0050) Slice Thickness")
        print("  • (0020,0032) Image Position Patient")
        print("  • (0020,0037) Image Orientation Patient")
        
        return df_result
        
    except Exception as e:
        print(f"❌ Error downloading series: {e}")
        return None

def main():
    """Main function to find and download DRR-ready chest CT"""
    print("🏥 === DRR-Ready Chest CT Downloader ===")
    print("Finding chest CT data with complete geometric metadata for DRR generation...\n")
    
    # Find suitable collections
    suitable_collections = find_drr_suitable_collections()
    
    if not suitable_collections:
        print("❌ No suitable collections found for DRR")
        return
    
    print(f"\n🎯 Found {len(suitable_collections)} collections suitable for DRR:")
    for i, collection in enumerate(suitable_collections, 1):
        print(f"  {i}. {collection}")
    
    # Analyze the first suitable collection
    target_collection = suitable_collections[0]
    print(f"\n🔍 Analyzing {target_collection} (most suitable for DRR)...")
    
    is_suitable, series_data = analyze_collection_for_drr(target_collection)
    
    if is_suitable and series_data is not None:
        print(f"\n✅ {target_collection} appears suitable for DRR generation!")
        
        # Download sample
        result = download_drr_sample(series_data, target_collection, num_series=1)
        
        if result is not None:
            print(f"\n🎉 Success! Downloaded DRR-ready chest CT from {target_collection}")
            print("\nNext steps for DRR generation:")
            print("1. Load DICOM files using pydicom or SimpleITK")
            print("2. Extract geometric parameters from DICOM headers")
            print("3. Set up projection geometry (source-detector-patient positions)")
            print("4. Use DRR libraries like TIGRE, RTK, or ASTRA Toolbox")
            
    else:
        print(f"❌ {target_collection} not suitable, trying next collection...")
        # Could implement logic to try next collection

if __name__ == "__main__":
    main() 