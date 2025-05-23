#!/usr/bin/env python3
"""
Stereo DRR Generator V4 - Clinical Quality
Combines the working horizontal shift approach from V3 with the superior 
clinical parameters and processing pipeline from drr_clinical_final.py
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
OUTPUT_DIR = "outputs/stereo_v4_clinical"
LOG_FILE = "logs/stereo_drr_v4_clinical_generation.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def generate_clinical_quality_drr(ct_volume, projection_type='AP'):
    """
    Generate clinical-quality DRR using the proven clinical_final.py approach
    with standard X-ray film dimensions and superior processing pipeline
    """
    # Standard X-ray film/detector sizes (in mm) - from clinical_final.py
    STANDARD_SIZES = {
        'AP': {'width': 356, 'height': 432},  # 14"x17" portrait
        'Lateral': {'width': 432, 'height': 356}  # 17"x14" landscape
    }
    
    # Get volume and spacing
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    log_message(f"Volume shape (Z,Y,X): {volume.shape}")
    log_message(f"Spacing (X,Y,Z): {spacing} mm")
    log_message(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # NO windowing - preserve full HU range for bones (key improvement from clinical_final)
    
    # Convert HU to linear attenuation coefficients - clinical_final parameters
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air - zero attenuation for black background
    air_mask = volume < -900
    mu_volume[air_mask] = 0.0
    
    # Lung tissue - very low attenuation
    lung_mask = (volume >= -900) & (volume < -500)
    lung_hu = volume[lung_mask]
    mu_volume[lung_mask] = 0.0001 + (lung_hu + 900) * (0.001 / 400)
    
    # Soft tissue - standard attenuation
    soft_mask = (volume >= -500) & (volume < 200)
    soft_hu = volume[soft_mask]
    mu_volume[soft_mask] = mu_water * (1.0 + soft_hu / 1000.0)
    
    # Bone - enhanced attenuation for visibility (2.5x multiplier - clinical_final key parameter)
    bone_mask = volume >= 200
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (2.5 + bone_hu / 500.0)
    
    # Very light smoothing to reduce noise without losing detail
    mu_volume = ndimage.gaussian_filter(mu_volume, sigma=[0.2, 0.2, 0.2])
    
    log_message(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate projection with correct spacing
    if projection_type == 'AP':
        # AP view: integrate along Y axis
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)
        # Physical dimensions of projection
        proj_height_mm = projection.shape[0] * spacing[2]  # Z dimension
        proj_width_mm = projection.shape[1] * spacing[0]   # X dimension
    else:  # Lateral
        # Lateral view: integrate along X axis
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
        # Physical dimensions of projection
        proj_height_mm = projection.shape[0] * spacing[2]  # Z dimension  
        proj_width_mm = projection.shape[1] * spacing[1]   # Y dimension
    
    log_message(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    log_message(f"Projection physical size: {proj_width_mm:.1f} x {proj_height_mm:.1f} mm")
    
    # Resample to standard X-ray film dimensions (key clinical_final improvement)
    detector_size = STANDARD_SIZES[projection_type]
    detector_width_mm = detector_size['width']
    detector_height_mm = detector_size['height']
    
    # Calculate scale to fit anatomy within detector while maintaining aspect ratio
    scale_x = detector_width_mm / proj_width_mm
    scale_y = detector_height_mm / proj_height_mm
    scale = min(scale_x, scale_y) * 0.9  # 0.9 to leave some border
    
    # Calculate new dimensions
    new_width_mm = proj_width_mm * scale
    new_height_mm = proj_height_mm * scale
    new_width_px = int(detector_width_mm / 0.5)  # 0.5mm detector pixel spacing
    new_height_px = int(detector_height_mm / 0.5)
    
    # Calculate anatomy size in pixels
    anatomy_width_px = int(new_width_mm / 0.5)
    anatomy_height_px = int(new_height_mm / 0.5)
    
    # Resample projection to new size
    zoom_factors = [anatomy_height_px / projection.shape[0], 
                   anatomy_width_px / projection.shape[1]]
    projection_resampled = ndimage.zoom(projection, zoom_factors, order=3)
    
    # Create detector-sized image and center the anatomy (clinical_final approach)
    detector_image = np.zeros((new_height_px, new_width_px))
    
    # Calculate centering offsets
    y_offset = (new_height_px - anatomy_height_px) // 2
    x_offset = (new_width_px - anatomy_width_px) // 2
    
    # Place resampled projection in center
    detector_image[y_offset:y_offset+anatomy_height_px, 
                  x_offset:x_offset+anatomy_width_px] = projection_resampled
    
    log_message(f"Detector size: {detector_width_mm} x {detector_height_mm} mm")
    log_message(f"Final image: {new_width_px} x {new_height_px} pixels")
    log_message(f"Anatomy size: {anatomy_width_px} x {anatomy_height_px} pixels")
    
    # Use detector image as projection for further processing
    projection = detector_image
    
    # Apply Beer-Lambert law
    transmission = np.exp(-projection)
    
    # Convert to intensity with log transform
    epsilon = 1e-6
    intensity = -np.log10(transmission + epsilon)
    
    # Normalize using percentiles from body region (clinical_final improvement)
    body_mask = projection > 0.1
    if np.any(body_mask):
        # Use 1-99 percentiles for better dynamic range
        p1 = np.percentile(intensity[body_mask], 1)
        p99 = np.percentile(intensity[body_mask], 99)
        intensity = (intensity - p1) / (p99 - p1)
        intensity = np.clip(intensity, 0, 1)
    
    # Mild gamma correction
    gamma = 1.2
    intensity = np.power(intensity, 1.0 / gamma)
    
    # Ensure air stays black
    air_projection = projection < 0.01
    intensity[air_projection] = 0
    
    # Very subtle edge enhancement (clinical_final approach)
    blurred = ndimage.gaussian_filter(intensity, sigma=1.0)
    intensity = intensity + 0.1 * (intensity - blurred)
    intensity = np.clip(intensity, 0, 1)
    
    log_message(f"Final intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    
    return intensity

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
    """Generate stereo DRR pair using clinical quality base with horizontal shifts"""
    log_message(f"\n--- Generating clinical quality stereo pair for {projection_type} view ---")
    
    # Generate the base DRR using clinical_final quality approach
    drr_base = generate_clinical_quality_drr(ct_volume, projection_type)
    
    # Validate base image
    if np.all(drr_base == 0) or np.all(drr_base == 1):
        log_message("❌ ERROR: Base DRR is invalid (all black or all white)")
        return None, None, None
    
    # Create stereo views by shifting
    drr_left = create_stereo_shift(drr_base, STEREO_SHIFT, 'left')
    drr_right = create_stereo_shift(drr_base, STEREO_SHIFT, 'right')
    
    log_message(f"✅ Generated clinical quality stereo pair with {STEREO_SHIFT} pixel shift")
    
    return drr_left, drr_base, drr_right

def save_clinical_stereo_xray(image, filename, title=None, markers=None):
    """Save DRR with clinical X-ray film appearance - based on clinical_final approach"""
    # Standard X-ray film display size 
    h, w = image.shape
    # Convert pixel dimensions to inches (assuming 0.5mm pixel spacing = ~50 DPI)
    fig_width = w * 0.5 / 25.4  # mm to inches
    fig_height = h * 0.5 / 25.4
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display with equal aspect (already properly sized)
    im = ax.imshow(image, cmap='gray', aspect='equal', 
                   vmin=0, vmax=1, interpolation='bilinear')
    
    # Add title if provided
    if title:
        ax.text(0.5, 0.98, title, transform=ax.transAxes,
                fontsize=14, color='white', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='black', alpha=0.8))
    
    # Add markers if provided (e.g., R/L for AP view)
    if markers:
        for marker in markers:
            ax.text(marker['x'], marker['y'], marker['text'], 
                   transform=ax.transAxes,
                   fontsize=16, color='white', weight='bold',
                   ha=marker.get('ha', 'center'), 
                   va=marker.get('va', 'center'),
                   bbox=dict(boxstyle='square,pad=0.3',
                            facecolor='black', edgecolor='white',
                            linewidth=2))
    
    ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    plt.close()

def save_stereo_images(drr_left, drr_center, drr_right, view_name, patient_id):
    """Save clinical quality stereo DRR images"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if drr_left is None or drr_center is None or drr_right is None:
        log_message(f"❌ Skipping save for {view_name} - invalid images")
        return
    
    # Save individual images with clinical quality
    save_clinical_stereo_xray(drr_left, f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_left.png")
    save_clinical_stereo_xray(drr_center, f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_center.png")
    save_clinical_stereo_xray(drr_right, f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_right.png")
    
    log_message(f"Saved clinical quality individual images for {view_name}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='white')
    fig.suptitle(f'Clinical Quality Stereo DRR - {view_name} View (Patient: {patient_id})', fontsize=16)
    
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
    
    # Quality assessment
    log_message(f"Image quality - Min: {drr_center.min():.3f}, Max: {drr_center.max():.3f}, "
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
    plt.title(f'Clinical Quality Anaglyph 3D - {view_name} View (use red-cyan glasses)')
    
    filename = f"{OUTPUT_DIR}/drr_{patient_id}_{view_name}_anaglyph.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved anaglyph: {filename}")

def main():
    """Main execution function"""
    log_message("=== Clinical Quality Stereo DRR Generator V4 Started ===")
    log_message(f"Using clinical_final.py quality parameters with {STEREO_SHIFT} pixel stereo shift")
    log_message("Key improvements: Standard film dimensions, 2.5x bone enhancement, no windowing")
    
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
            ap_left, ap_center, ap_right = generate_stereo_pair(ct_volume, 'AP')
            if ap_center is not None:
                save_stereo_images(ap_left, ap_center, ap_right, 'AP', dataset['patient_id'])
                total_success += 1
            total_attempts += 1
            
            # Generate Lateral stereo pair
            lat_left, lat_center, lat_right = generate_stereo_pair(ct_volume, 'Lateral')
            if lat_center is not None:
                save_stereo_images(lat_left, lat_center, lat_right, 'Lateral', dataset['patient_id'])
                total_success += 1
            total_attempts += 1
            
        except Exception as e:
            log_message(f"❌ Error processing {dataset['name']}: {e}")
            import traceback
            log_message(traceback.format_exc())
    
    log_message(f"\n=== Clinical Quality Stereo DRR Generation Complete ===")
    log_message(f"Success rate: {total_success}/{total_attempts} projections")
    
    if total_success < total_attempts:
        log_message("❌ Some stereo pairs failed to generate")
    else:
        log_message("✅ All clinical quality stereo pairs generated successfully!")
    
    log_message(f"\nOutput directory: {OUTPUT_DIR}")
    log_message("\nKey improvements in V4:")
    log_message("  ✓ Standard X-ray film dimensions (14\"x17\" AP, 17\"x14\" Lateral)")
    log_message("  ✓ Enhanced bone visibility (2.5x attenuation multiplier)")
    log_message("  ✓ No clinical windowing (preserves full HU range)")
    log_message("  ✓ Professional film-like borders and centering")
    log_message("  ✓ Percentile-based normalization for better contrast")
    log_message("  ✓ Clinical quality processing pipeline")

if __name__ == "__main__":
    main()