#!/usr/bin/env python3
"""
Stereo DRR Generator - Creates stereo pairs with 3-degree separation
Based on drr_refined.py (V8) which produces clinical-quality DRRs

This script generates stereo DRR pairs for 3D reconstruction by creating
two projections with a small angular separation (±3 degrees).
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime

# Stereo parameters
STEREO_ANGLE = 3.0  # degrees
OUTPUT_DIR = "outputs/stereo"
LOG_FILE = "logs/stereo_drr_generation.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def load_ct_volume(dicom_directory):
    """Load CT volume from DICOM series"""
    log_message(f"Loading CT volume from: {dicom_directory}")
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
    reader.SetFileNames(dicom_names)
    
    volume = reader.Execute()
    
    # Get volume properties
    size = volume.GetSize()
    spacing = volume.GetSpacing()
    origin = volume.GetOrigin()
    direction = volume.GetDirection()
    
    log_message(f"Volume loaded: {size[0]}×{size[1]}×{size[2]} voxels")
    log_message(f"Voxel spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")
    log_message(f"Physical size: ({size[0]*spacing[0]:.1f}, {size[1]*spacing[1]:.1f}, {size[2]*spacing[2]:.1f}) mm")
    
    return volume

def rotate_volume(volume, angle_degrees, axis='z'):
    """Rotate volume around specified axis"""
    angle_radians = np.radians(angle_degrees)
    
    # Get volume center
    size = volume.GetSize()
    spacing = volume.GetSpacing()
    center = [size[i] * spacing[i] / 2.0 for i in range(3)]
    
    # Create rotation transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    
    if axis == 'z':
        transform.SetRotation(0, 0, angle_radians)
    elif axis == 'y':
        transform.SetRotation(0, angle_radians, 0)
    elif axis == 'x':
        transform.SetRotation(angle_radians, 0, 0)
    
    # Resample volume with rotation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000)  # Air value
    resampler.SetTransform(transform)
    
    rotated_volume = resampler.Execute(volume)
    
    return rotated_volume

def generate_drr_projection(volume_array, projection_axis='AP', angle=0.0):
    """
    Generate a single DRR projection with optional rotation
    Based on drr_refined.py implementation
    """
    log_message(f"Generating {projection_axis} projection at angle {angle:.1f}°")
    
    # Ensure proper orientation (z, y, x) for SimpleITK volumes
    if projection_axis == 'AP':
        # AP view: integrate along Y axis (anterior to posterior)
        projection = np.sum(volume_array, axis=1)
    elif projection_axis == 'Lateral':
        # Lateral view: integrate along X axis (right to left)
        projection = np.sum(volume_array, axis=2)
    else:
        raise ValueError(f"Unknown projection axis: {projection_axis}")
    
    # Invert z-axis for radiographic convention
    projection = projection[::-1, :]
    
    # Apply clinical transformations (from drr_refined.py)
    # 1. Scale by voxel dimensions
    projection_scaled = projection * 3.0  # Assuming 3mm slice thickness
    
    # 2. Convert HU to linear attenuation coefficients
    # Using tissue-specific values
    projection_atten = np.zeros_like(projection_scaled)
    
    # Air/lung
    mask_air = projection_scaled < -500
    projection_atten[mask_air] = 0.0001 * projection_scaled[mask_air]
    
    # Soft tissue
    mask_soft = (projection_scaled >= -500) & (projection_scaled < 200)
    mu_water = 0.019
    projection_atten[mask_soft] = mu_water * (1.0 + projection_scaled[mask_soft] / 1000.0)
    
    # Bone
    mask_bone = projection_scaled >= 200
    projection_atten[mask_bone] = mu_water * (1.5 + projection_scaled[mask_bone] / 1000.0)
    
    # 3. Apply Beer-Lambert law
    epsilon = 1e-10
    transmission = np.exp(-projection_atten)
    
    # 4. Logarithmic film response
    drr = -np.log10(transmission + epsilon) / 3.0
    drr = np.clip(drr, 0, 1)
    
    # 5. Apply clinical window/level and gamma correction
    drr = np.power(drr, 0.5)  # Gamma correction
    
    # 6. Invert for radiographic appearance
    drr = 1.0 - drr
    
    return drr

def generate_stereo_pair(volume, projection_axis='AP'):
    """Generate stereo DRR pair with ±3 degree separation"""
    log_message(f"Generating stereo pair for {projection_axis} view")
    
    # Convert to numpy array
    volume_array = sitk.GetArrayFromImage(volume)
    
    # Generate left eye view (-3 degrees)
    log_message("Generating left eye view (-3°)")
    volume_left = rotate_volume(volume, -STEREO_ANGLE, axis='y' if projection_axis == 'AP' else 'z')
    array_left = sitk.GetArrayFromImage(volume_left)
    drr_left = generate_drr_projection(array_left, projection_axis, -STEREO_ANGLE)
    
    # Generate right eye view (+3 degrees)
    log_message("Generating right eye view (+3°)")
    volume_right = rotate_volume(volume, STEREO_ANGLE, axis='y' if projection_axis == 'AP' else 'z')
    array_right = sitk.GetArrayFromImage(volume_right)
    drr_right = generate_drr_projection(array_right, projection_axis, STEREO_ANGLE)
    
    # Generate center view (0 degrees) for reference
    log_message("Generating center view (0°)")
    drr_center = generate_drr_projection(volume_array, projection_axis, 0.0)
    
    return drr_left, drr_center, drr_right

def save_stereo_images(drr_left, drr_center, drr_right, view_name, patient_id):
    """Save stereo DRR images and create comparison visualization"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save individual images
    plt.figure(figsize=(10, 10))
    
    # Left eye
    plt.imshow(drr_left, cmap='gray')
    plt.axis('off')
    filename_left = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_left.png"
    plt.savefig(filename_left, dpi=300, bbox_inches='tight', pad_inches=0)
    log_message(f"Saved left eye: {filename_left}")
    
    # Right eye
    plt.imshow(drr_right, cmap='gray')
    plt.axis('off')
    filename_right = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_right.png"
    plt.savefig(filename_right, dpi=300, bbox_inches='tight', pad_inches=0)
    log_message(f"Saved right eye: {filename_right}")
    
    # Center (reference)
    plt.imshow(drr_center, cmap='gray')
    plt.axis('off')
    filename_center = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_center.png"
    plt.savefig(filename_center, dpi=300, bbox_inches='tight', pad_inches=0)
    log_message(f"Saved center: {filename_center}")
    
    plt.close()
    
    # Create stereo comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f'Stereo DRR Pair - {view_name} View (Patient: {patient_id})', fontsize=16, y=0.98)
    
    axes[0].imshow(drr_left, cmap='gray')
    axes[0].set_title('Left Eye (-3°)')
    axes[0].axis('off')
    
    axes[1].imshow(drr_center, cmap='gray')
    axes[1].set_title('Center (0°)')
    axes[1].axis('off')
    
    axes[2].imshow(drr_right, cmap='gray')
    axes[2].set_title('Right Eye (+3°)')
    axes[2].axis('off')
    
    plt.tight_layout()
    filename_comparison = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_stereo_comparison.png"
    plt.savefig(filename_comparison, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved comparison: {filename_comparison}")
    
    # Create anaglyph (red-cyan) 3D image
    create_anaglyph(drr_left, drr_right, view_name, patient_id)

def create_anaglyph(drr_left, drr_right, view_name, patient_id):
    """Create red-cyan anaglyph 3D image"""
    log_message("Creating anaglyph 3D image")
    
    # Create RGB image
    height, width = drr_left.shape
    anaglyph = np.zeros((height, width, 3))
    
    # Red channel from left eye
    anaglyph[:, :, 0] = drr_left
    
    # Green and blue channels from right eye
    anaglyph[:, :, 1] = drr_right
    anaglyph[:, :, 2] = drr_right
    
    plt.figure(figsize=(10, 10))
    plt.imshow(anaglyph)
    plt.axis('off')
    plt.title(f'Anaglyph 3D - {view_name} View (use red-cyan glasses)')
    
    filename_anaglyph = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_anaglyph.png"
    plt.savefig(filename_anaglyph, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved anaglyph: {filename_anaglyph}")

def main():
    """Main execution function"""
    log_message("=== Stereo DRR Generator Started ===")
    log_message(f"Stereo angle separation: ±{STEREO_ANGLE}°")
    
    # Define CT datasets
    datasets = [
        {
            'name': 'NSCLC-Radiomics',
            'path': 'data/tciaDownload/1.3.6.1.4.1.32722.99.99.298991776521342375010861296712563382046',
            'patient_id': 'LUNG1-001'
        },
        {
            'name': 'COVID-19-NY-SBU',
            'path': 'data/tciaDownload/1.3.6.1.4.1.14519.5.2.1.99.1071.29029751181371965166204843962164',
            'patient_id': 'A670621'
        }
    ]
    
    for dataset in datasets:
        log_message(f"\nProcessing dataset: {dataset['name']}")
        
        try:
            # Load CT volume
            volume = load_ct_volume(dataset['path'])
            
            # Generate AP stereo pair
            log_message("\n--- Generating AP stereo pair ---")
            ap_left, ap_center, ap_right = generate_stereo_pair(volume, 'AP')
            save_stereo_images(ap_left, ap_center, ap_right, 'AP', dataset['patient_id'])
            
            # Generate Lateral stereo pair
            log_message("\n--- Generating Lateral stereo pair ---")
            lat_left, lat_center, lat_right = generate_stereo_pair(volume, 'Lateral')
            save_stereo_images(lat_left, lat_center, lat_right, 'Lateral', dataset['patient_id'])
            
            log_message(f"✅ Successfully generated stereo pairs for {dataset['name']}")
            
        except Exception as e:
            log_message(f"❌ Error processing {dataset['name']}: {e}")
            import traceback
            log_message(traceback.format_exc())
    
    log_message("\n=== Stereo DRR Generation Complete ===")
    log_message(f"Output directory: {OUTPUT_DIR}")
    log_message("\nStereo pairs can be used for:")
    log_message("1. 3D reconstruction algorithms")
    log_message("2. Depth perception in medical visualization")
    log_message("3. Stereoscopic display systems")
    log_message("4. Training AI models for 3D understanding")

if __name__ == "__main__":
    main()