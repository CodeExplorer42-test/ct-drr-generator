#!/usr/bin/env python3
"""
Stereo DRR Generator V11 - Clinical Rotation
============================================
COMBINES PROVEN WINNERS:
- V10's volume rotation approach (first non-distorted stereo)
- V8 clinical_final's visualization parameters (best quality)
- Larger rotation angle for better depth coverage

Key features:
- Volume rotation for true geometric stereo (no post-processing tricks)
- Clinical visualization pipeline from V8
- 2.5x bone enhancement (proven optimal)
- Standard X-ray film dimensions
- Conservative rotation to avoid distortion
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import time

# V11 Parameters - Combining V10 rotation with V8 clinical quality
ROTATION_ANGLE_DEGREES = 3.0  # Increased from V10's 1.5° for better depth
DETECTOR_PIXEL_SPACING = 0.5  # mm
STANDARD_DPI = 300

# Standard X-ray film dimensions from V8
STANDARD_SIZES = {
    'AP': {'width': 356, 'height': 432},      # 14"x17" portrait
    'Lateral': {'width': 432, 'height': 356}  # 17"x14" landscape
}

OUTPUT_DIR = "outputs/stereo_v11_clinical_rotation"
LOG_FILE = "logs/stereo_drr_v11_clinical_rotation.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def clinical_tissue_segmentation(volume):
    """Clinical tissue segmentation from V8 - proven optimal"""
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Tissue-specific attenuation
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -500)
    soft_mask = (volume >= -500) & (volume < 200)
    bone_mask = volume >= 200
    
    # Apply tissue-specific coefficients
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.001
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # 2.5x bone enhancement from V8 clinical_final
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (2.5 + bone_hu / 500.0)
    
    # Clamp extreme values
    mu_volume = np.clip(mu_volume, 0, 0.16)
    
    return mu_volume

def rotate_volume_carefully(sitk_volume, angle_degrees, axis='z'):
    """Volume rotation from V10 - proven to work without distortion"""
    log_message(f"Rotating volume by {angle_degrees:.1f}° around {axis} axis")
    
    size = sitk_volume.GetSize()
    spacing = sitk_volume.GetSpacing()
    origin = sitk_volume.GetOrigin()
    
    # Calculate rotation center (volume center)
    center_physical = [
        origin[0] + size[0] * spacing[0] / 2,
        origin[1] + size[1] * spacing[1] / 2, 
        origin[2] + size[2] * spacing[2] / 2
    ]
    
    # Create rotation transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)
    
    # Apply rotation based on axis
    angle_rad = np.radians(angle_degrees)
    if axis == 'z':
        transform.SetRotation(0, 0, angle_rad)
    
    # Resample with high quality (B-spline from V10)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(-1000)  # Air value
    resampler.SetTransform(transform)
    
    rotated_volume = resampler.Execute(sitk_volume)
    
    # Verify rotation didn't corrupt data
    original_array = sitk.GetArrayFromImage(sitk_volume)
    rotated_array = sitk.GetArrayFromImage(rotated_volume)
    
    log_message(f"  Original HU range: [{original_array.min():.0f}, {original_array.max():.0f}]")
    log_message(f"  Rotated HU range: [{rotated_array.min():.0f}, {rotated_array.max():.0f}]")
    
    # Check for corruption
    if rotated_array.max() < -900:
        log_message("  WARNING: Rotation may have corrupted data")
        return sitk_volume  # Return original if corrupted
    
    return rotated_volume

def generate_clinical_projection(mu_volume, spacing, projection_type='AP'):
    """Generate projection with V8 clinical quality"""
    # Standard parallel projection
    if projection_type == 'AP':
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)
    else:  # Lateral
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
    
    log_message(f"  Raw projection range: [{projection.min():.3f}, {projection.max():.3f}]")
    
    # Resample to standard film dimensions
    detector_width_mm = STANDARD_SIZES[projection_type]['width']
    detector_height_mm = STANDARD_SIZES[projection_type]['height']
    
    # Calculate pixel dimensions
    new_width_px = int(detector_width_mm / DETECTOR_PIXEL_SPACING)
    new_height_px = int(detector_height_mm / DETECTOR_PIXEL_SPACING)
    
    # Calculate anatomy scaling (90% of detector with border)
    scale = 0.9
    new_width_mm = detector_width_mm * scale
    new_height_mm = detector_height_mm * scale
    
    # Convert to pixels for anatomy
    anatomy_width_px = int(new_width_mm / DETECTOR_PIXEL_SPACING)
    anatomy_height_px = int(new_height_mm / DETECTOR_PIXEL_SPACING)
    
    # Resample projection
    zoom_factors = [anatomy_height_px / projection.shape[0], 
                   anatomy_width_px / projection.shape[1]]
    projection_resampled = ndimage.zoom(projection, zoom_factors, order=3)
    
    # Create detector-sized image and center anatomy
    detector_image = np.zeros((new_height_px, new_width_px))
    
    y_offset = (new_height_px - anatomy_height_px) // 2
    x_offset = (new_width_px - anatomy_width_px) // 2
    
    detector_image[y_offset:y_offset+anatomy_height_px, 
                  x_offset:x_offset+anatomy_width_px] = projection_resampled
    
    return detector_image

def clinical_radiograph_conversion(projection):
    """Convert projection to radiograph using V8 clinical parameters"""
    # Apply Beer-Lambert law
    transmission = np.exp(-projection)
    
    # Convert to intensity with log transform
    epsilon = 1e-6
    intensity = -np.log10(transmission + epsilon)
    
    # Normalize using percentiles from body region (V8 method)
    body_mask = projection > 0.1
    if np.any(body_mask):
        # Use 1-99 percentiles for optimal dynamic range
        p1 = np.percentile(intensity[body_mask], 1)
        p99 = np.percentile(intensity[body_mask], 99)
        intensity = (intensity - p1) / (p99 - p1)
        intensity = np.clip(intensity, 0, 1)
    
    # Mild gamma correction (V8 value)
    gamma = 1.2
    intensity = np.power(intensity, 1.0 / gamma)
    
    # Ensure air stays black
    air_projection = projection < 0.01
    intensity[air_projection] = 0
    
    # Very subtle edge enhancement (V8 parameters)
    blurred = ndimage.gaussian_filter(intensity, sigma=1.0)
    intensity = intensity + 0.1 * (intensity - blurred)
    intensity = np.clip(intensity, 0, 1)
    
    return intensity

def generate_v11_clinical_stereo(ct_volume):
    """Generate V11 stereo using V10 rotation + V8 visualization"""
    log_message(f"\n--- V11 Clinical Rotation Stereo Generation ---")
    log_message("Combining V10 volume rotation with V8 clinical visualization")
    
    # Get volume info
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    
    log_message(f"Volume shape: {volume.shape}, spacing: {spacing} mm")
    
    # Apply clinical tissue segmentation
    log_message("Applying clinical tissue segmentation...")
    mu_volume_center = clinical_tissue_segmentation(volume)
    log_message(f"Attenuation range: [{mu_volume_center.min():.5f}, {mu_volume_center.max():.5f}] mm^-1")
    
    # Generate CENTER view (no rotation)
    log_message("\nGenerating CENTER view (no rotation)...")
    projection_center = generate_clinical_projection(mu_volume_center, spacing, 'AP')
    drr_center = clinical_radiograph_conversion(projection_center)
    
    # Generate LEFT view (negative rotation)
    log_message("\nGenerating LEFT view...")
    try:
        rotated_left = rotate_volume_carefully(ct_volume, -ROTATION_ANGLE_DEGREES/2, 'z')
        volume_left = sitk.GetArrayFromImage(rotated_left)
        mu_volume_left = clinical_tissue_segmentation(volume_left)
        projection_left = generate_clinical_projection(mu_volume_left, spacing, 'AP')
        drr_left = clinical_radiograph_conversion(projection_left)
    except Exception as e:
        log_message(f"LEFT rotation failed: {e}")
        drr_left = drr_center
    
    # Generate RIGHT view (positive rotation)
    log_message("\nGenerating RIGHT view...")
    try:
        rotated_right = rotate_volume_carefully(ct_volume, +ROTATION_ANGLE_DEGREES/2, 'z')
        volume_right = sitk.GetArrayFromImage(rotated_right)
        mu_volume_right = clinical_tissue_segmentation(volume_right)
        projection_right = generate_clinical_projection(mu_volume_right, spacing, 'AP')
        drr_right = clinical_radiograph_conversion(projection_right)
    except Exception as e:
        log_message(f"RIGHT rotation failed: {e}")
        drr_right = drr_center
    
    # Calculate stereo quality metrics
    diff_left_center = np.mean(np.abs(drr_left - drr_center))
    diff_right_center = np.mean(np.abs(drr_right - drr_center))
    diff_left_right = np.mean(np.abs(drr_left - drr_right))
    
    log_message(f"\nStereo quality assessment:")
    log_message(f"  Left-Center difference: {diff_left_center:.4f}")
    log_message(f"  Right-Center difference: {diff_right_center:.4f}")
    log_message(f"  Left-Right difference: {diff_left_right:.4f}")
    
    # Assess stereo quality
    if diff_left_right > 0.01:
        log_message("  ✅ Strong stereo separation achieved")
        stereo_quality = "strong"
    elif diff_left_right > 0.005:
        log_message("  ✅ Detectable stereo differences")
        stereo_quality = "detectable"
    else:
        log_message("  ⚠️ Minimal stereo differences")
        stereo_quality = "minimal"
    
    # Calculate effective baseline
    source_distance = 1000  # mm (approximate)
    baseline_mm = 2 * source_distance * np.sin(np.radians(ROTATION_ANGLE_DEGREES/2))
    log_message(f"  Effective baseline: {baseline_mm:.1f}mm")
    
    return drr_left, drr_center, drr_right, diff_left_right, stereo_quality, baseline_mm

def conservative_depth_estimation(left_img, right_img):
    """Conservative depth estimation from V10"""
    log_message("\nPerforming depth estimation...")
    
    h, w = left_img.shape
    disparity_map = np.zeros((h, w))
    
    # Block matching parameters
    block_size = 11
    half_block = block_size // 2
    max_disparity = 100  # Increased for larger baseline
    step = 4  # Process every 4th pixel
    
    valid_pixels = 0
    max_disp_found = 0
    
    for y in range(half_block, h - half_block, step):
        if y % 100 == 0:
            progress = (y - half_block) / (h - 2*half_block) * 100
            log_message(f"  Depth estimation progress: {progress:.1f}%")
        
        for x in range(half_block + max_disparity, w - half_block, step):
            left_block = left_img[y-half_block:y+half_block+1, 
                                 x-half_block:x+half_block+1]
            
            min_ssd = float('inf')
            best_d = 0
            
            # Search for matching block in right image
            for d in range(0, min(max_disparity, x-half_block)):
                right_block = right_img[y-half_block:y+half_block+1,
                                       x-d-half_block:x-d+half_block+1]
                
                ssd = np.sum((left_block - right_block)**2)
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_d = d
            
            # Only accept high confidence matches
            if min_ssd < 0.5:  # Threshold for confidence
                disparity_map[y:y+step, x:x+step] = best_d
                valid_pixels += step * step
                max_disp_found = max(max_disp_found, best_d)
    
    # Apply smoothing
    disparity_map = ndimage.median_filter(disparity_map, size=5)
    
    # Calculate coverage
    coverage = valid_pixels / (h * w) * 100
    
    log_message(f"  Depth estimation complete:")
    log_message(f"    Max disparity: {max_disp_found} pixels")
    log_message(f"    Coverage: {coverage:.1f}%")
    
    return disparity_map, max_disp_found, coverage

def save_v11_outputs(images, metrics, patient_id):
    """Save V11 outputs with clinical quality"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    drr_left, drr_center, drr_right = images
    diff, quality, baseline = metrics
    
    # Perform depth estimation
    depth_map, max_disp, coverage = conservative_depth_estimation(drr_left, drr_right)
    
    # Save individual images
    for img, view in zip(images, ['left', 'center', 'right']):
        save_clinical_radiograph(img, f"{OUTPUT_DIR}/drr_{patient_id}_AP_{view}.png",
                               f"{patient_id} - AP - {view.capitalize()} (V11)")
    
    # Save depth map
    plt.figure(figsize=(8, 10), facecolor='black')
    plt.imshow(depth_map, cmap='plasma', aspect='equal')
    plt.colorbar(label='Disparity (pixels)')
    plt.title(f'Depth Map - {patient_id}', color='white', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/depth_{patient_id}_AP.png", dpi=300, 
               bbox_inches='tight', facecolor='black')
    plt.close()
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 20), facecolor='black')
    
    # Show stereo views
    titles = ['Left View\n(-1.5°)', 'Right View\n(+1.5°)', 
              'Center View\n(Reference)', 'Depth Map']
    imgs = [drr_left, drr_right, drr_center, depth_map]
    
    for ax, img, title in zip(axes.flat, imgs, titles):
        if title == 'Depth Map':
            im = ax.imshow(img, cmap='plasma', aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=12, pad=10)
        ax.axis('off')
    
    # Add parameters text
    param_text = (f'V11 CLINICAL ROTATION\n\n'
                 f'Rotation: ±{ROTATION_ANGLE_DEGREES/2:.1f}°\n'
                 f'Baseline: {baseline:.1f}mm\n'
                 f'Image diff: {diff:.4f}\n'
                 f'Quality: {quality}\n\n'
                 f'Max disparity: {max_disp} px\n'
                 f'Depth coverage: {coverage:.1f}%\n\n'
                 f'Clinical params from V8\n'
                 f'Rotation method from V10')
    
    fig.text(0.5, 0.02, param_text, ha='center', va='bottom',
            fontsize=11, color='lightgreen', weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#001100', alpha=0.8))
    
    plt.suptitle(f'{patient_id} - V11 Clinical Rotation Stereo\n'
                f'Combining V10 Geometry with V8 Visualization',
                color='lightgreen', fontsize=16, y=0.98)
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_AP.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved outputs to {OUTPUT_DIR}")

def save_clinical_radiograph(image, filename, title=None):
    """Save radiograph with clinical appearance"""
    h, w = image.shape
    fig_width = w * DETECTOR_PIXEL_SPACING / 25.4
    fig_height = h * DETECTOR_PIXEL_SPACING / 25.4
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(image, cmap='gray', aspect='equal', vmin=0, vmax=1)
    
    if title:
        ax.text(0.5, 0.02, title, transform=ax.transAxes,
               fontsize=10, color='white', ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.axis('off')
    plt.savefig(filename, dpi=STANDARD_DPI, bbox_inches='tight', 
               pad_inches=0, facecolor='black')
    plt.close()

def main():
    """Main execution"""
    log_message("="*80)
    log_message("V11 Clinical Rotation Stereo DRR Generator")
    log_message("="*80)
    log_message("Combining proven approaches:")
    log_message("  • V10 volume rotation (non-distorted stereo)")
    log_message("  • V8 clinical visualization (best quality)")
    log_message(f"  • {ROTATION_ANGLE_DEGREES}° total rotation (±{ROTATION_ANGLE_DEGREES/2}°)")
    log_message("  • Conservative approach for real depth")
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
        
        # Generate clinical rotation stereo
        drr_left, drr_center, drr_right, diff, quality, baseline = generate_v11_clinical_stereo(ct_volume)
        
        # Save outputs
        images = (drr_left, drr_center, drr_right)
        metrics = (diff, quality, baseline)
        save_v11_outputs(images, metrics, dataset['patient_id'])
        
        total_time = time.time() - start_time
        
        log_message(f"\n{'='*80}")
        log_message(f"V11 COMPLETE - Clinical Quality with Real Stereo")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"Stereo difference: {diff:.4f}")
        log_message(f"Quality: {quality}")
        log_message(f"Baseline: {baseline:.1f}mm")
        log_message(f"Output: {OUTPUT_DIR}")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()