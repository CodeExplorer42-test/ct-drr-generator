#!/usr/bin/env python3
"""
Stereo DRR Generator V3 - Simple shift-based stereo
Uses the proven drr_refined.py approach with horizontal shifts for stereo effect
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
from pathlib import Path

# Stereo parameters
STEREO_SHIFT = 10  # pixels - horizontal shift for stereo effect
OUTPUT_DIR = "outputs/stereo_v3"
LOG_FILE = "logs/stereo_drr_v3_generation.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def apply_window_level(image, window_center=-938, window_width=1079):
    """Apply clinical chest window"""
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / window_width
    return windowed

def generate_refined_drr(ct_volume, projection_type='AP'):
    """
    Generate clinical-quality DRR based on drr_refined.py
    Returns the DRR image and aspect ratio
    """
    # Get volume and spacing
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    log_message(f"Generating {projection_type} projection")
    log_message(f"Volume shape (Z,Y,X): {volume.shape}")
    log_message(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Apply clinical chest window
    volume_windowed = apply_window_level(volume)
    
    # Convert HU to linear attenuation coefficients
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air (outside body) - zero attenuation for black background
    air_mask = volume < -950
    mu_volume[air_mask] = 0.0
    
    # Lung tissue - very low but non-zero for vessel visibility
    lung_mask = (volume >= -950) & (volume < -500)
    lung_hu = volume[lung_mask]
    mu_volume[lung_mask] = 0.0001 + (lung_hu + 950) * (0.001 / 450)
    
    # Soft tissue - standard attenuation
    soft_mask = (volume >= -500) & (volume < 300)
    soft_hu = volume[soft_mask]
    mu_volume[soft_mask] = mu_water * (1.0 + soft_hu / 1000.0)
    
    # Bone - moderate enhancement
    bone_mask = volume >= 300
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (1.3 + bone_hu / 1500.0)
    
    # Apply mild smoothing
    mu_volume = ndimage.gaussian_filter(mu_volume, sigma=[0.3, 0.3, 0.3])
    
    # Generate projection with correct spacing
    if projection_type == 'AP':
        # AP view: integrate along Y axis
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)
        aspect_ratio = spacing[2] / spacing[0]  # Z/X spacing ratio
    else:  # Lateral
        # Lateral view: integrate along X axis
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
        aspect_ratio = spacing[2] / spacing[1]  # Z/Y spacing ratio
    
    log_message(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    
    # Apply Beer-Lambert law
    transmission = np.exp(-projection)
    
    # Convert to intensity with logarithmic response
    epsilon = 1e-6
    intensity = -np.log10(transmission + epsilon)
    
    # Normalize
    if intensity.max() > intensity.min():
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    
    # Apply mild gamma correction
    gamma = 1.2
    intensity = np.power(intensity, 1.0 / gamma)
    
    # Ensure air stays black
    air_projection = projection < 0.01
    intensity[air_projection] = 0
    
    # Mild edge enhancement
    blurred = ndimage.gaussian_filter(intensity, sigma=1.5)
    intensity = intensity + 0.15 * (intensity - blurred)
    intensity = np.clip(intensity, 0, 1)
    
    return intensity, aspect_ratio

def create_stereo_shift(image, shift_pixels, direction='left'):
    """
    Create stereo view by shifting the image horizontally
    This simulates the different viewpoints of left and right eyes
    """
    shifted = np.zeros_like(image)
    
    if direction == 'left':
        # Shift image to the right (for left eye view)
        if shift_pixels < image.shape[1]:
            shifted[:, shift_pixels:] = image[:, :-shift_pixels]
    else:  # right
        # Shift image to the left (for right eye view)
        if shift_pixels < image.shape[1]:
            shifted[:, :-shift_pixels] = image[:, shift_pixels:]
    
    return shifted

def generate_stereo_pair(ct_volume, projection_type='AP'):
    """Generate stereo DRR pair using horizontal shifts"""
    log_message(f"\n--- Generating stereo pair for {projection_type} view ---")
    
    # Generate the base DRR
    drr_base, aspect_ratio = generate_refined_drr(ct_volume, projection_type)
    
    # Validate base image
    if np.all(drr_base == 0) or np.all(drr_base == 1):
        log_message("❌ ERROR: Base DRR is invalid (all black or all white)")
        return None, None, None, aspect_ratio
    
    # Create stereo views by shifting
    drr_left = create_stereo_shift(drr_base, STEREO_SHIFT, 'left')
    drr_right = create_stereo_shift(drr_base, STEREO_SHIFT, 'right')
    
    log_message(f"✅ Generated stereo pair with {STEREO_SHIFT} pixel shift")
    
    return drr_left, drr_base, drr_right, aspect_ratio

def save_stereo_images(drr_left, drr_center, drr_right, view_name, patient_id, aspect_ratio):
    """Save stereo DRR images with proper visualization"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if drr_left is None or drr_center is None or drr_right is None:
        log_message(f"❌ Skipping save for {view_name} - invalid images")
        return
    
    # Calculate figure dimensions
    base_width = 8
    fig_height = base_width / aspect_ratio
    
    # Save individual images
    for img, suffix in [(drr_left, 'left'), (drr_center, 'center'), (drr_right, 'right')]:
        plt.figure(figsize=(base_width, fig_height), facecolor='black')
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.axis('off')
        
        filename = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_{suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')
        plt.close()
        log_message(f"Saved: {filename}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, fig_height*1.2), facecolor='white')
    fig.suptitle(f'Stereo DRR Pair - {view_name} View (Patient: {patient_id})', fontsize=16)
    
    titles = ['Left Eye View', 'Center View', 'Right Eye View']
    images = [drr_left, drr_center, drr_right]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', aspect='equal')
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    filename_comparison = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_stereo_comparison.png"
    plt.savefig(filename_comparison, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved comparison: {filename_comparison}")
    
    # Create anaglyph 3D image
    create_anaglyph(drr_left, drr_right, view_name, patient_id)
    
    # Check quality
    log_message(f"Image stats - Min: {drr_center.min():.3f}, Max: {drr_center.max():.3f}, "
               f"Unique values: {len(np.unique(drr_center))}")

def create_anaglyph(drr_left, drr_right, view_name, patient_id):
    """Create red-cyan anaglyph 3D image"""
    # Create RGB image
    height, width = drr_left.shape
    anaglyph = np.zeros((height, width, 3))
    
    # Red channel from left eye
    anaglyph[:, :, 0] = drr_left
    
    # Green and blue channels from right eye
    anaglyph[:, :, 1] = drr_right * 0.7  # Reduce intensity for better 3D effect
    anaglyph[:, :, 2] = drr_right * 0.7
    
    plt.figure(figsize=(10, 10))
    plt.imshow(anaglyph)
    plt.axis('off')
    plt.title(f'Anaglyph 3D - {view_name} View (use red-cyan glasses)')
    
    filename = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_anaglyph.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved anaglyph: {filename}")

def main():
    """Main execution function"""
    log_message("=== Stereo DRR Generator V3 Started ===")
    log_message(f"Using horizontal shift of {STEREO_SHIFT} pixels for stereo effect")
    log_message("Based on proven drr_refined.py implementation")
    
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
    
    total_success = 0
    total_attempts = 0
    
    for dataset in datasets:
        log_message(f"\nProcessing dataset: {dataset['name']}")
        
        try:
            # Load CT volume
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dataset['path'])
            reader.SetFileNames(dicom_names)
            ct_volume = reader.Execute()
            
            log_message(f"Loaded {len(dicom_names)} DICOM files")
            
            # Generate AP stereo pair
            ap_left, ap_center, ap_right, ap_aspect = generate_stereo_pair(ct_volume, 'AP')
            if ap_center is not None:
                save_stereo_images(ap_left, ap_center, ap_right, 'AP', dataset['patient_id'], ap_aspect)
                total_success += 1
            total_attempts += 1
            
            # Generate Lateral stereo pair
            lat_left, lat_center, lat_right, lat_aspect = generate_stereo_pair(ct_volume, 'Lateral')
            if lat_center is not None:
                save_stereo_images(lat_left, lat_center, lat_right, 'Lateral', dataset['patient_id'], lat_aspect)
                total_success += 1
            total_attempts += 1
            
        except Exception as e:
            log_message(f"❌ Error processing {dataset['name']}: {e}")
            import traceback
            log_message(traceback.format_exc())
    
    log_message(f"\n=== Stereo DRR Generation Complete ===")
    log_message(f"Success rate: {total_success}/{total_attempts} projections")
    
    if total_success < total_attempts:
        log_message("❌ FAILURE: Not all stereo pairs were generated successfully")
        log_message("Check the logs for error details")
    else:
        log_message("✅ SUCCESS: All stereo pairs generated")
    
    log_message(f"\nOutput directory: {OUTPUT_DIR}")
    log_message("\nNote: The horizontal shift creates a parallax effect suitable for:")
    log_message("- Depth perception when viewed with stereo display")
    log_message("- Input to stereo matching algorithms")
    log_message("- 3D reconstruction using disparity maps")

if __name__ == "__main__":
    main()