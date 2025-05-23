#!/usr/bin/env python3
"""
Stereo DRR Generator V2 - Corrected Implementation
Uses sheared projections instead of volume rotation for proper stereo effect
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage

# Stereo parameters
STEREO_ANGLE = 3.0  # degrees
OUTPUT_DIR = "outputs/stereo_v2"
LOG_FILE = "logs/stereo_drr_v2_generation.log"

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
    
    log_message(f"Volume loaded: {size[0]}×{size[1]}×{size[2]} voxels")
    log_message(f"Voxel spacing: ({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm")
    
    return volume

def apply_window_level(image, window_center=-938, window_width=1079):
    """Apply clinical chest window"""
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(image, lower, upper)
    return windowed

def generate_sheared_projection(volume_array, spacing, projection_type='AP', shear_angle=0.0):
    """
    Generate DRR with sheared projection to simulate stereo
    Shear angle in degrees - positive for right eye, negative for left eye
    """
    log_message(f"Generating {projection_type} projection with shear angle {shear_angle:.1f}°")
    
    # Apply clinical window
    volume_windowed = apply_window_level(volume_array)
    
    # Convert HU to attenuation coefficients (from drr_refined.py)
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume_windowed, dtype=np.float32)
    
    # Air - black background
    air_mask = volume_array < -950
    mu_volume[air_mask] = 0.0
    
    # Lung tissue
    lung_mask = (volume_array >= -950) & (volume_array < -500)
    lung_hu = volume_array[lung_mask]
    mu_volume[lung_mask] = 0.0001 + (lung_hu + 950) * (0.001 / 450)
    
    # Soft tissue
    soft_mask = (volume_array >= -500) & (volume_array < 300)
    soft_hu = volume_array[soft_mask]
    mu_volume[soft_mask] = mu_water * (1.0 + soft_hu / 1000.0)
    
    # Bone
    bone_mask = volume_array >= 300
    bone_hu = volume_array[bone_mask]
    mu_volume[bone_mask] = mu_water * (1.3 + bone_hu / 1500.0)
    
    # Apply smoothing
    mu_volume = ndimage.gaussian_filter(mu_volume, sigma=[0.3, 0.3, 0.3])
    
    # Convert shear angle to radians
    shear_rad = np.radians(shear_angle)
    
    # Generate projection with shear
    if projection_type == 'AP':
        # For AP view, shear in X direction as we integrate along Y
        # This simulates horizontal eye separation
        projection = np.zeros((mu_volume.shape[0], mu_volume.shape[2]))
        
        for z in range(mu_volume.shape[0]):
            # Calculate shear offset for this slice
            # Shear increases with depth (Y position)
            for x in range(mu_volume.shape[2]):
                ray_sum = 0
                for y in range(mu_volume.shape[1]):
                    # Calculate sheared X position
                    x_sheared = x + int(y * np.tan(shear_rad))
                    if 0 <= x_sheared < mu_volume.shape[2]:
                        ray_sum += mu_volume[z, y, x_sheared]
                projection[z, x] = ray_sum * spacing[1]
        
        projection = np.flipud(projection)
        aspect_ratio = spacing[2] / spacing[0]
        
    else:  # Lateral
        # For Lateral view, shear in Y direction as we integrate along X
        projection = np.zeros((mu_volume.shape[0], mu_volume.shape[1]))
        
        for z in range(mu_volume.shape[0]):
            for y in range(mu_volume.shape[1]):
                ray_sum = 0
                for x in range(mu_volume.shape[2]):
                    # Calculate sheared Y position
                    y_sheared = y + int(x * np.tan(shear_rad))
                    if 0 <= y_sheared < mu_volume.shape[1]:
                        ray_sum += mu_volume[z, y_sheared, x]
                projection[z, y] = ray_sum * spacing[0]
        
        projection = np.flipud(projection)
        aspect_ratio = spacing[2] / spacing[1]
    
    log_message(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    
    # Apply Beer-Lambert law
    transmission = np.exp(-projection)
    
    # Convert to intensity
    epsilon = 1e-6
    intensity = -np.log10(transmission + epsilon)
    
    # Normalize
    if intensity.max() > intensity.min():
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    
    # Apply gamma correction
    gamma = 1.2
    intensity = np.power(intensity, 1.0 / gamma)
    
    # Ensure air stays black
    air_projection = projection < 0.01
    intensity[air_projection] = 0
    
    # Edge enhancement
    blurred = ndimage.gaussian_filter(intensity, sigma=1.5)
    intensity = intensity + 0.15 * (intensity - blurred)
    intensity = np.clip(intensity, 0, 1)
    
    return intensity, aspect_ratio

def generate_stereo_pair(ct_volume, projection_type='AP'):
    """Generate stereo DRR pair using sheared projections"""
    log_message(f"Generating stereo pair for {projection_type} view")
    
    # Get volume array and spacing
    volume_array = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    
    log_message(f"Volume shape (Z,Y,X): {volume_array.shape}")
    log_message(f"HU range: [{volume_array.min():.0f}, {volume_array.max():.0f}]")
    
    # Generate left eye view (negative shear)
    drr_left, aspect_ratio = generate_sheared_projection(
        volume_array, spacing, projection_type, -STEREO_ANGLE
    )
    
    # Generate center view (no shear)
    drr_center, _ = generate_sheared_projection(
        volume_array, spacing, projection_type, 0.0
    )
    
    # Generate right eye view (positive shear)
    drr_right, _ = generate_sheared_projection(
        volume_array, spacing, projection_type, STEREO_ANGLE
    )
    
    return drr_left, drr_center, drr_right, aspect_ratio

def save_stereo_images(drr_left, drr_center, drr_right, view_name, patient_id, aspect_ratio):
    """Save stereo DRR images with proper aspect ratio"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate figure dimensions
    base_width = 10
    fig_height = base_width / aspect_ratio
    
    # Save individual images
    for img, suffix in [(drr_left, 'left'), (drr_center, 'center'), (drr_right, 'right')]:
        fig = plt.figure(figsize=(base_width, fig_height), facecolor='black')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.axis('off')
        
        filename = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_{suffix}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
        log_message(f"Saved {suffix} eye: {filename}")
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(20, fig_height*2), facecolor='black')
    fig.suptitle(f'Stereo DRR Pair - {view_name} View (Patient: {patient_id})', 
                 fontsize=16, y=0.98, color='white')
    
    for ax, img, title in zip(axes, 
                             [drr_left, drr_center, drr_right],
                             ['Left Eye (-3°)', 'Center (0°)', 'Right Eye (+3°)']):
        ax.imshow(img, cmap='gray', aspect='auto', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    filename_comparison = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_stereo_comparison.png"
    plt.savefig(filename_comparison, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    log_message(f"Saved comparison: {filename_comparison}")
    
    # Create anaglyph
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
    
    plt.figure(figsize=(10, 10), facecolor='black')
    plt.imshow(anaglyph)
    plt.axis('off')
    plt.title(f'Anaglyph 3D - {view_name} View (use red-cyan glasses)', color='white')
    
    filename_anaglyph = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_anaglyph.png"
    plt.savefig(filename_anaglyph, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    log_message(f"Saved anaglyph: {filename_anaglyph}")

def validate_output(image, image_name):
    """Check if generated image is valid"""
    if np.all(image == 0):
        log_message(f"❌ WARNING: {image_name} is completely black!")
        return False
    elif np.all(image == 1):
        log_message(f"❌ WARNING: {image_name} is completely white!")
        return False
    else:
        unique_values = len(np.unique(image))
        log_message(f"✅ {image_name} appears valid (unique values: {unique_values})")
        return True

def main():
    """Main execution function"""
    log_message("=== Stereo DRR Generator V2 Started ===")
    log_message(f"Using sheared projection with ±{STEREO_ANGLE}° angle")
    
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
    
    success_count = 0
    
    for dataset in datasets:
        log_message(f"\nProcessing dataset: {dataset['name']}")
        
        try:
            # Load CT volume
            ct_volume = load_ct_volume(dataset['path'])
            
            # Generate AP stereo pair
            log_message("\n--- Generating AP stereo pair ---")
            ap_left, ap_center, ap_right, ap_aspect = generate_stereo_pair(ct_volume, 'AP')
            
            # Validate outputs
            valid_ap = all([
                validate_output(ap_left, "AP left"),
                validate_output(ap_center, "AP center"),
                validate_output(ap_right, "AP right")
            ])
            
            if valid_ap:
                save_stereo_images(ap_left, ap_center, ap_right, 'AP', dataset['patient_id'], ap_aspect)
                log_message("✅ AP stereo pair generated successfully")
            else:
                log_message("❌ AP stereo pair generation failed - invalid output")
            
            # Generate Lateral stereo pair
            log_message("\n--- Generating Lateral stereo pair ---")
            lat_left, lat_center, lat_right, lat_aspect = generate_stereo_pair(ct_volume, 'Lateral')
            
            # Validate outputs
            valid_lat = all([
                validate_output(lat_left, "Lateral left"),
                validate_output(lat_center, "Lateral center"),
                validate_output(lat_right, "Lateral right")
            ])
            
            if valid_lat:
                save_stereo_images(lat_left, lat_center, lat_right, 'Lateral', dataset['patient_id'], lat_aspect)
                log_message("✅ Lateral stereo pair generated successfully")
            else:
                log_message("❌ Lateral stereo pair generation failed - invalid output")
            
            if valid_ap and valid_lat:
                success_count += 1
                log_message(f"✅ Successfully generated all stereo pairs for {dataset['name']}")
            else:
                log_message(f"❌ Partial failure for {dataset['name']}")
            
        except Exception as e:
            log_message(f"❌ Error processing {dataset['name']}: {e}")
            import traceback
            log_message(traceback.format_exc())
    
    log_message(f"\n=== Stereo DRR Generation Complete ===")
    log_message(f"Success rate: {success_count}/{len(datasets)} datasets")
    log_message(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()