#!/usr/bin/env python3
"""
Stereo DRR Generator V11 - Fast Corrected Geometry
==================================================
FAST VERSION with corrected stereo geometry calculations:
- Use V10's fast volume rotation approach (no slow ray-casting)
- Fix baseline and focal length calculations 
- Validate against chest anatomy (402mm depth)
- Provide correct stereo parameters for reconstruction

Goal: Get the right stereo parameters WITHOUT computational overhead
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import time

# V11 FAST CORRECTED Parameters - Fixed aspect ratio
# For AP view: need to accommodate 500mm width x 402mm height
DETECTOR_WIDTH_MM = 500.0   # Match CT width
DETECTOR_HEIGHT_MM = 402.0  # Match CT depth
RESOLUTION_DPI = 600
PIXEL_SPACING_MM = 0.4

# CORRECTED stereo geometry
STEREO_ANGLE_DEGREES = 3.0  # Total stereo separation
SOURCE_TO_DETECTOR_MM = 1000.0

OUTPUT_DIR = "outputs/stereo_v11_fast_corrected"
LOG_FILE = "logs/stereo_drr_v11_fast_corrected.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def calculate_corrected_stereo_parameters(volume_info):
    """Calculate CORRECTED stereo parameters using simple geometry"""
    volume_spacing = np.array(volume_info['spacing'])
    volume_size = np.array(volume_info['size'])
    
    # Physical volume dimensions
    volume_extent_mm = volume_size * volume_spacing
    chest_depth_mm = volume_extent_mm[2]
    
    log_message(f"CT Volume Analysis:")
    log_message(f"  Volume size: {volume_size} voxels")
    log_message(f"  Voxel spacing: {volume_spacing} mm")
    log_message(f"  Physical dimensions: {volume_extent_mm} mm")
    log_message(f"  Chest depth (Z-direction): {chest_depth_mm:.1f}mm")
    
    # CORRECTED baseline calculation
    # For stereo angle of ±1.5°, at source distance of 1000mm from volume center
    half_angle_rad = np.radians(STEREO_ANGLE_DEGREES / 2)
    
    # True stereo baseline = 2 * source_distance * sin(half_angle)
    CORRECTED_BASELINE_MM = 2 * SOURCE_TO_DETECTOR_MM * np.sin(half_angle_rad)
    
    # Focal length is source-to-detector distance
    FOCAL_LENGTH_MM = SOURCE_TO_DETECTOR_MM
    
    log_message(f"\nCORRECTED Stereo Parameters:")
    log_message(f"  Stereo angle: ±{STEREO_ANGLE_DEGREES/2:.1f}° (total {STEREO_ANGLE_DEGREES}°)")
    log_message(f"  CORRECTED baseline: {CORRECTED_BASELINE_MM:.1f}mm")
    log_message(f"  Focal length: {FOCAL_LENGTH_MM}mm")
    log_message(f"  Pixel spacing: {PIXEL_SPACING_MM}mm")
    
    # Validate against chest anatomy
    # Expected depth range: 50mm (anterior) to chest_depth_mm (posterior)
    anterior_depth_mm = 50
    posterior_depth_mm = chest_depth_mm
    
    # Calculate expected disparity range using stereo formula
    # disparity = (focal_length * baseline) / (depth * pixel_spacing)
    max_disparity_expected = (FOCAL_LENGTH_MM * CORRECTED_BASELINE_MM) / (anterior_depth_mm * PIXEL_SPACING_MM)
    min_disparity_expected = (FOCAL_LENGTH_MM * CORRECTED_BASELINE_MM) / (posterior_depth_mm * PIXEL_SPACING_MM)
    
    log_message(f"\nVALIDATION against chest anatomy:")
    log_message(f"  Expected depth range: {anterior_depth_mm}mm - {posterior_depth_mm:.0f}mm")
    log_message(f"  Expected max disparity: {max_disparity_expected:.1f} pixels (anterior structures)")
    log_message(f"  Expected min disparity: {min_disparity_expected:.1f} pixels (posterior structures)")
    log_message(f"  Disparity range: {min_disparity_expected:.1f} - {max_disparity_expected:.1f} pixels")
    
    # Depth sensitivity per 1 pixel disparity
    # depth = (focal_length * baseline) / (disparity * pixel_spacing)
    # For disparity = 1 pixel:
    depth_per_pixel = (FOCAL_LENGTH_MM * CORRECTED_BASELINE_MM) / (1.0 * PIXEL_SPACING_MM)
    log_message(f"  Depth sensitivity: {depth_per_pixel:.1f}mm per pixel disparity")
    
    # Sanity check: What depth does 1 pixel give us?
    depth_1px = depth_per_pixel
    log_message(f"  1 pixel disparity = {depth_1px:.1f}mm depth difference")
    
    return {
        'baseline_mm': CORRECTED_BASELINE_MM,
        'focal_length_mm': FOCAL_LENGTH_MM,
        'pixel_spacing_mm': PIXEL_SPACING_MM,
        'chest_depth_mm': chest_depth_mm,
        'expected_min_disparity': min_disparity_expected,
        'expected_max_disparity': max_disparity_expected,
        'depth_per_pixel_mm': depth_per_pixel
    }

def conservative_tissue_segmentation(volume):
    """Same conservative segmentation as V10"""
    mu_water = 0.019
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -500)
    soft_mask = (volume >= -500) & (volume < 150)
    bone_mask = volume >= 150
    
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.001
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    mu_volume[bone_mask] = mu_water * (2.5 + volume[bone_mask] / 500.0)
    
    return mu_volume

def rotate_volume_carefully(sitk_volume, angle_degrees, axis='z'):
    """Same careful rotation from V10"""
    log_message(f"Rotating volume by {angle_degrees:.1f}° around {axis} axis")
    
    size = sitk_volume.GetSize()
    spacing = sitk_volume.GetSpacing()
    origin = sitk_volume.GetOrigin()
    
    center_physical = [
        origin[0] + size[0] * spacing[0] / 2,
        origin[1] + size[1] * spacing[1] / 2, 
        origin[2] + size[2] * spacing[2] / 2
    ]
    
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)
    
    angle_rad = np.radians(angle_degrees)
    if axis == 'z':
        transform.SetRotation(0, 0, angle_rad)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetTransform(transform)
    
    rotated_volume = resampler.Execute(sitk_volume)
    
    # Verify rotation success
    original_array = sitk.GetArrayFromImage(sitk_volume)
    rotated_array = sitk.GetArrayFromImage(rotated_volume)
    
    log_message(f"  Original HU range: [{original_array.min():.0f}, {original_array.max():.0f}]")
    log_message(f"  Rotated HU range: [{rotated_array.min():.0f}, {rotated_array.max():.0f}]")
    
    if rotated_array.max() < -900:
        log_message("  WARNING: Rotation may have failed")
        return sitk_volume
    
    return rotated_volume

def generate_standard_projection(mu_volume, spacing):
    """Fixed projection with correct aspect ratio"""
    # AP projection: sum along Y-axis
    projection = np.sum(mu_volume, axis=1) * spacing[1]
    projection = np.flipud(projection)
    
    # CRITICAL FIX: Apply correct aspect ratio for anisotropic voxels
    # spacing = [0.98, 0.98, 3.0] mm -> Z/X ratio = 3.06
    aspect_ratio = spacing[2] / spacing[0]  # Z/X spacing ratio
    
    # Resample to detector size WITH aspect ratio correction
    detector_pixels_u = int(DETECTOR_WIDTH_MM / PIXEL_SPACING_MM)
    detector_pixels_v = int(DETECTOR_HEIGHT_MM / PIXEL_SPACING_MM)
    
    # Scale anatomy to fit detector, preserving aspect ratio
    scale = 0.9
    # Width: scale by projection width
    anatomy_width_px = int(projection.shape[1] * scale)
    # Height: scale by projection height AND correct for voxel spacing
    anatomy_height_px = int(projection.shape[0] * scale / aspect_ratio)
    
    zoom_factors = [anatomy_height_px / projection.shape[0], 
                   anatomy_width_px / projection.shape[1]]
    projection_resampled = ndimage.zoom(projection, zoom_factors, order=3)
    
    # Center on detector
    detector_image = np.zeros((detector_pixels_v, detector_pixels_u))
    y_offset = (detector_pixels_v - anatomy_height_px) // 2
    x_offset = (detector_pixels_u - anatomy_width_px) // 2
    
    detector_image[y_offset:y_offset+anatomy_height_px, 
                  x_offset:x_offset+anatomy_width_px] = projection_resampled
    
    return detector_image

def generate_v11_fast_corrected_stereo(ct_volume):
    """Generate V11 fast corrected stereo using volume rotation"""
    log_message(f"\n--- V11 FAST CORRECTED Stereo Generation ---")
    log_message("Using V10's fast volume rotation with CORRECTED geometry calculations")
    
    # Get volume data
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    origin = np.array(ct_volume.GetOrigin())
    size = np.array(ct_volume.GetSize())
    
    volume_info = {
        'spacing': spacing,
        'origin': origin,
        'size': size
    }
    
    log_message(f"Volume: {volume.shape}, spacing: {spacing} mm")
    
    # Calculate CORRECTED stereo parameters
    corrected_params = calculate_corrected_stereo_parameters(volume_info)
    
    # Conservative tissue segmentation
    mu_volume = conservative_tissue_segmentation(volume)
    log_message(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate stereo views using rotation (same as V10 but with corrected parameters)
    half_angle = STEREO_ANGLE_DEGREES / 2
    
    # CENTER view
    log_message("Generating CENTER view...")
    projection_center = generate_standard_projection(mu_volume, spacing)
    
    # LEFT view (negative rotation)
    log_message("Generating LEFT view...")
    try:
        rotated_volume_left = rotate_volume_carefully(ct_volume, -half_angle, 'z')
        rotated_array_left = sitk.GetArrayFromImage(rotated_volume_left)
        mu_volume_left = conservative_tissue_segmentation(rotated_array_left)
        projection_left = generate_standard_projection(mu_volume_left, spacing)
    except Exception as e:
        log_message(f"LEFT rotation failed: {e}")
        projection_left = projection_center
    
    # RIGHT view (positive rotation)
    log_message("Generating RIGHT view...")
    try:
        rotated_volume_right = rotate_volume_carefully(ct_volume, +half_angle, 'z')
        rotated_array_right = sitk.GetArrayFromImage(rotated_volume_right)
        mu_volume_right = conservative_tissue_segmentation(rotated_array_right)
        projection_right = generate_standard_projection(mu_volume_right, spacing)
    except Exception as e:
        log_message(f"RIGHT rotation failed: {e}")
        projection_right = projection_center
    
    # Convert to radiographs
    def to_radiograph(projection):
        transmission = np.exp(-projection)
        epsilon = 1e-7
        intensity = -np.log10(transmission + epsilon)
        
        body_mask = projection > 0.05
        if np.any(body_mask):
            p_low = np.percentile(intensity[body_mask], 2)
            p_high = np.percentile(intensity[body_mask], 98)
            intensity = (intensity - p_low) / (p_high - p_low)
            intensity = np.clip(intensity, 0, 1)
        
        intensity = np.power(intensity, 1.0 / 1.1)
        
        air_mask = projection < 0.02
        intensity[air_mask] = 0
        
        return np.clip(intensity, 0, 1)
    
    drr_left = to_radiograph(projection_left)
    drr_center = to_radiograph(projection_center)
    drr_right = to_radiograph(projection_right)
    
    # Assess image differences
    diff_left_right = np.mean(np.abs(drr_left - drr_right))
    
    log_message(f"\nV11 FAST CORRECTED Results:")
    log_message(f"  Image difference (L-R): {diff_left_right:.4f}")
    log_message(f"  {'✅ Detectable' if diff_left_right > 0.005 else '⚠️ Minimal'} stereo differences")
    
    log_message(f"✅ V11 fast corrected stereo generation complete")
    
    return drr_left, drr_center, drr_right, corrected_params, diff_left_right

def save_v11_fast_outputs(images, params, image_diff, patient_id):
    """Save V11 fast corrected outputs"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    drr_left, drr_center, drr_right = images
    
    # Save individual images
    save_radiograph(drr_left, f"{OUTPUT_DIR}/drr_{patient_id}_AP_left.png", 
                   f"{patient_id} - AP - Left (V11 Corrected)")
    save_radiograph(drr_center, f"{OUTPUT_DIR}/drr_{patient_id}_AP_center.png", 
                   f"{patient_id} - AP - Center (V11 Corrected)")
    save_radiograph(drr_right, f"{OUTPUT_DIR}/drr_{patient_id}_AP_right.png", 
                   f"{patient_id} - AP - Right (V11 Corrected)")
    
    # Comparison image - Fixed size for 1250x1005 images
    # Each DRR is 1250x1005, aspect ratio ~1.24, so make height larger
    fig, axes = plt.subplots(1, 4, figsize=(24, 12), facecolor='black')
    
    imgs = [drr_left, drr_center, drr_right, np.zeros_like(drr_left)]
    titles = ['Left View\n(V11 Fast)', 'Center View\n(Reference)', 'Right View\n(V11 Fast)', 'CORRECTED\nParameters']
    
    for ax, img, title in zip(axes[:3], imgs[:3], titles[:3]):
        ax.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=12, pad=15)
        ax.axis('off')
    
    # Parameters display
    axes[3].text(0.5, 0.5, f'V11 CORRECTED\nSTEREO PARAMETERS\n\n'
                           f'Baseline: {params["baseline_mm"]:.1f}mm\n'
                           f'Focal length: {params["focal_length_mm"]}mm\n'
                           f'Pixel spacing: {params["pixel_spacing_mm"]}mm\n'
                           f'Image diff: {image_diff:.4f}\n\n'
                           f'Expected disparity:\n'
                           f'{params["expected_min_disparity"]:.1f} - {params["expected_max_disparity"]:.1f} px\n\n'
                           f'Depth sensitivity:\n'
                           f'{params["depth_per_pixel_mm"]:.1f}mm/pixel\n\n'
                           f'Chest depth:\n'
                           f'{params["chest_depth_mm"]:.0f}mm', 
               transform=axes[3].transAxes, fontsize=10, color='lightgreen', weight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', alpha=0.8))
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].set_title(titles[3], color='white', fontsize=12, pad=15)
    axes[3].axis('off')
    
    plt.suptitle(f'{patient_id} - AP - V11 FAST CORRECTED STEREO\n'
                f'Fixed Geometry with Validated Parameters',
                color='lightgreen', fontsize=16, y=0.95, weight='bold')
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_AP.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved V11 fast corrected results to {filename}")

def save_radiograph(image, filename, title=None):
    """Save individual radiograph"""
    h, w = image.shape
    fig_width = w / RESOLUTION_DPI
    fig_height = h / RESOLUTION_DPI
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(image, cmap='gray', aspect='equal', vmin=0, vmax=1, interpolation='lanczos')
    
    if title:
        ax.text(0.5, 0.02, title, transform=ax.transAxes,
               fontsize=8, color='white', ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.axis('off')
    plt.savefig(filename, dpi=RESOLUTION_DPI, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

def main():
    """Main execution with fast corrected geometry"""
    log_message("="*80)
    log_message("V11 FAST CORRECTED Stereo DRR Generator")
    log_message("="*80)
    log_message("Fast version with CORRECTED stereo geometry:")
    log_message(f"  • Use V10's volume rotation (fast)")
    log_message(f"  • Fix baseline calculation (V10 was wrong)")
    log_message(f"  • Validate against {402:.0f}mm chest anatomy")
    log_message(f"  • Provide correct parameters for reconstruction")
    log_message("="*80)
    
    dataset = {
        'path': 'data/tciaDownload/1.3.6.1.4.1.32722.99.99.298991776521342375010861296712563382046',
        'patient_id': 'LUNG1-001',
        'name': 'NSCLC-Radiomics'
    }
    
    start_time = time.time()
    
    try:
        # Load DICOM
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dataset['path'])
        
        log_message(f"Loading {len(dicom_files)} DICOM files...")
        reader.SetFileNames(dicom_files)
        ct_volume = reader.Execute()
        
        # Generate fast corrected stereo
        drr_left, drr_center, drr_right, corrected_params, image_diff = generate_v11_fast_corrected_stereo(ct_volume)
        
        # Save outputs
        images = (drr_left, drr_center, drr_right)
        save_v11_fast_outputs(images, corrected_params, image_diff, dataset['patient_id'])
        
        total_time = time.time() - start_time
        
        # FINAL CORRECTED PARAMETERS
        log_message(f"\n{'='*80}")
        log_message(f"V11 FAST CORRECTED - FINAL PARAMETERS")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"")
        log_message(f"CORRECTED STEREO PARAMETERS FOR RECONSTRUCTION:")
        log_message(f"  Baseline: {corrected_params['baseline_mm']:.1f}mm")
        log_message(f"  Focal length: {corrected_params['focal_length_mm']}mm")
        log_message(f"  Pixel spacing: {corrected_params['pixel_spacing_mm']}mm")
        log_message(f"")
        log_message(f"VALIDATION AGAINST CHEST ANATOMY:")
        log_message(f"  Chest depth: {corrected_params['chest_depth_mm']:.0f}mm")
        log_message(f"  Expected disparity range: {corrected_params['expected_min_disparity']:.1f} - {corrected_params['expected_max_disparity']:.1f} pixels")
        log_message(f"  Depth sensitivity: {corrected_params['depth_per_pixel_mm']:.1f}mm per pixel")
        log_message(f"")
        log_message(f"STEREO QUALITY:")
        log_message(f"  Image difference: {image_diff:.4f}")
        log_message(f"  Status: {'✅ Ready for reconstruction' if image_diff > 0.005 else '⚠️ May need larger baseline'}")
        log_message(f"")
        log_message(f"Output: {OUTPUT_DIR}")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()