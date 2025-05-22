#!/usr/bin/env python3
"""
Extract DRR-essential geometric metadata from downloaded DICOM CT data
"""

import os
import glob
import numpy as np

def install_pydicom():
    """Install pydicom if not available"""
    try:
        import pydicom
        return pydicom
    except ImportError:
        import subprocess
        import sys
        print("Installing pydicom...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydicom"])
        import pydicom
        return pydicom

def extract_drr_metadata(dicom_dir):
    """Extract essential geometric metadata for DRR generation"""
    pydicom = install_pydicom()
    
    print(f"🔍 Extracting DRR metadata from: {dicom_dir}")
    
    # Find all DICOM files
    dicom_files = glob.glob(os.path.join(dicom_dir, "*"))
    dicom_files = [f for f in dicom_files if os.path.isfile(f)]
    
    if not dicom_files:
        print("❌ No DICOM files found")
        return None
    
    print(f"📁 Found {len(dicom_files)} DICOM files")
    
    # Read first DICOM to get geometric parameters
    try:
        ds = pydicom.dcmread(dicom_files[0], force=True)
        print("✅ Successfully read DICOM file")
    except Exception as e:
        print(f"❌ Error reading DICOM: {e}")
        return None
    
    # Extract key geometric parameters for DRR
    metadata = {}
    
    print("\n📐 === DRR-Essential Geometric Parameters ===")
    
    # Patient positioning and orientation
    try:
        metadata['PatientID'] = getattr(ds, 'PatientID', 'Unknown')
        metadata['PatientPosition'] = getattr(ds, 'PatientPosition', 'Unknown')
        print(f"👤 Patient ID: {metadata['PatientID']}")
        print(f"📍 Patient Position: {metadata['PatientPosition']}")
    except:
        print("⚠️  Patient positioning info not available")
    
    # Image dimensions and spacing
    try:
        metadata['Rows'] = ds.Rows
        metadata['Columns'] = ds.Columns
        metadata['PixelSpacing'] = ds.PixelSpacing
        metadata['SliceThickness'] = getattr(ds, 'SliceThickness', 'Not specified')
        
        print(f"🖼️  Image dimensions: {metadata['Rows']} x {metadata['Columns']}")
        print(f"📏 Pixel spacing: {metadata['PixelSpacing']} mm")
        print(f"📐 Slice thickness: {metadata['SliceThickness']} mm")
    except Exception as e:
        print(f"⚠️  Image spacing info incomplete: {e}")
    
    # CT-specific geometric parameters for DRR
    try:
        # Source-to-detector distance (crucial for DRR)
        if hasattr(ds, 'DistanceSourceToDetector'):
            metadata['SourceToDetector'] = ds.DistanceSourceToDetector
            print(f"🎯 Source-to-Detector Distance: {metadata['SourceToDetector']} mm")
        else:
            print("⚠️  Source-to-Detector distance not found")
        
        # Source-to-patient distance
        if hasattr(ds, 'DistanceSourceToPatient'):
            metadata['SourceToPatient'] = ds.DistanceSourceToPatient
            print(f"👤 Source-to-Patient Distance: {metadata['SourceToPatient']} mm")
        else:
            print("⚠️  Source-to-Patient distance not found")
            
    except Exception as e:
        print(f"⚠️  CT geometric parameters incomplete: {e}")
    
    # Image position and orientation (essential for 3D reconstruction)
    try:
        if hasattr(ds, 'ImagePositionPatient'):
            metadata['ImagePositionPatient'] = ds.ImagePositionPatient
            print(f"📍 Image Position: [{', '.join([f'{x:.2f}' for x in metadata['ImagePositionPatient']])}] mm")
        
        if hasattr(ds, 'ImageOrientationPatient'):
            metadata['ImageOrientationPatient'] = ds.ImageOrientationPatient
            print(f"🧭 Image Orientation: [{', '.join([f'{x:.3f}' for x in metadata['ImageOrientationPatient'][:3]])}] (row)")
            print(f"   {''.ljust(18)} [{', '.join([f'{x:.3f}' for x in metadata['ImageOrientationPatient'][3:]])}] (col)")
    except Exception as e:
        print(f"⚠️  Position/orientation info incomplete: {e}")
    
    # Scanner and acquisition info
    try:
        metadata['Manufacturer'] = getattr(ds, 'Manufacturer', 'Unknown')
        metadata['ManufacturerModelName'] = getattr(ds, 'ManufacturerModelName', 'Unknown')
        metadata['KVP'] = getattr(ds, 'KVP', 'Unknown')
        
        print(f"\n🏥 Scanner: {metadata['Manufacturer']} {metadata['ManufacturerModelName']}")
        print(f"⚡ kVp: {metadata['KVP']}")
    except Exception as e:
        print(f"⚠️  Scanner info incomplete: {e}")
    
    # Analyze all slices for 3D volume info
    print(f"\n📊 === Volume Analysis (analyzing {min(len(dicom_files), 10)} slices) ===")
    
    positions = []
    for i, dcm_file in enumerate(dicom_files[:10]):  # Sample first 10 slices
        try:
            ds_slice = pydicom.dcmread(dcm_file, force=True)
            if hasattr(ds_slice, 'ImagePositionPatient'):
                positions.append(ds_slice.ImagePositionPatient[2])  # Z-coordinate
        except:
            continue
    
    if len(positions) > 1:
        positions.sort()
        slice_spacing = abs(positions[1] - positions[0]) if len(positions) > 1 else 0
        volume_extent = abs(positions[-1] - positions[0]) if len(positions) > 1 else 0
        print(f"📐 Calculated slice spacing: {slice_spacing:.2f} mm")
        print(f"📏 Volume extent (Z): {volume_extent:.2f} mm")
        metadata['CalculatedSliceSpacing'] = slice_spacing
        metadata['VolumeExtentZ'] = volume_extent
    
    # DRR generation recommendations
    print(f"\n🎯 === DRR Generation Recommendations ===")
    
    # Check if we have essential parameters
    has_geometry = any([
        hasattr(ds, 'DistanceSourceToDetector'),
        hasattr(ds, 'DistanceSourceToPatient'),
        hasattr(ds, 'ImagePositionPatient'),
        hasattr(ds, 'ImageOrientationPatient')
    ])
    
    if has_geometry:
        print("✅ This CT has sufficient geometric metadata for DRR generation!")
        print("📚 Recommended DRR libraries:")
        print("   • TIGRE (Python/MATLAB) - Good for cone-beam geometry")
        print("   • RTK (ITK-based) - Robust reconstruction toolkit")
        print("   • ASTRA Toolbox - High-performance projections")
        print("   • SimpleITK - Easy integration with Python")
        
        print("\n💡 DRR Generation Steps:")
        print("1. Load volume with SimpleITK or ITK")
        print("2. Set up projection geometry using extracted parameters")
        print("3. Define source and detector positions")
        print("4. Generate DRR using ray-casting or forward projection")
        
    else:
        print("⚠️  Limited geometric metadata - DRR generation may require assumptions")
        print("💡 You can still generate DRRs with typical CT scanner parameters")
        
    return metadata

def main():
    """Main function to extract DRR metadata"""
    print("🏥 === DRR Metadata Extractor ===\n")
    
    # Find the downloaded DICOM directory
    download_dirs = glob.glob("data/tciaDownload/*")
    dicom_dirs = [d for d in download_dirs if os.path.isdir(d)]
    
    if not dicom_dirs:
        print("❌ No DICOM directories found in data/tciaDownload/")
        return
    
    for dicom_dir in dicom_dirs:
        metadata = extract_drr_metadata(dicom_dir)
        
        if metadata:
            print(f"\n💾 Metadata extraction complete for {os.path.basename(dicom_dir)}")
            print("🚀 Ready for DRR generation!")
        else:
            print(f"❌ Failed to extract metadata from {dicom_dir}")

if __name__ == "__main__":
    main() 