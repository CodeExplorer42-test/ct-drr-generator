#!/usr/bin/env python3
"""
Stereo DRR Generator V12 - Fast Optimized
=========================================
Faster version of V12 with same principles:
- 5° stereo angle (optimal range)
- Reduced resolution for speed (600 DPI)
- Simplified depth estimation
- Core features maintained
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import time

# V12 Fast Parameters
STEREO_ANGLE_DEGREES = 5.0  # Optimal per research
SOURCE_TO_DETECTOR_MM = 1200.0
DETECTOR_PIXEL_SPACING = 0.5  # 600 DPI for speed
OUTPUT_DPI = 600

# Standard film dimensions
STANDARD_SIZES = {
    'AP': {'width': 356, 'height': 432}  # 14"x17"
}

OUTPUT_DIR = "outputs/stereo_v12_fast_optimized"
LOG_FILE = "logs/stereo_drr_v12_fast_optimized.log"

def log_message(message):
    """Log messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def optimized_tissue_segmentation(volume):
    """Simplified but effective tissue segmentation"""
    mu_water = 0.019
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Simple thresholds
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -500)
    soft_mask = (volume >= -500) & (volume < 200)
    bone_mask = volume >= 200
    
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.001
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # 3.0x bone enhancement (balanced)
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (3.0 + bone_hu / 400.0)
    
    return mu_volume

def rotate_volume_fast(sitk_volume, angle_degrees):
    """Fast volume rotation"""
    size = sitk_volume.GetSize()
    spacing = sitk_volume.GetSpacing()
    origin = sitk_volume.GetOrigin()
    
    center = [
        origin[0] + size[0] * spacing[0] / 2,
        origin[1] + size[1] * spacing[1] / 2, 
        origin[2] + size[2] * spacing[2] / 2
    ]
    
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(0, 0, np.radians(angle_degrees))
    
    # Linear interpolation for speed
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetInterpolator(sitk.sitkLinear)  # Faster than BSpline
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetTransform(transform)
    
    return resampler.Execute(sitk_volume)

def generate_fast_projection(mu_volume, spacing):
    """Fast projection generation"""
    # Simple parallel projection
    projection = np.sum(mu_volume, axis=1) * spacing[1]
    projection = np.flipud(projection)
    
    # Light scatter simulation
    scattered = ndimage.gaussian_filter(projection, sigma=3.0)
    projection = projection * 0.9 + scattered * 0.1
    
    return projection

def resample_to_detector_fast(projection):
    """Fast resampling to detector size"""
    detector_width_mm = STANDARD_SIZES['AP']['width']
    detector_height_mm = STANDARD_SIZES['AP']['height']
    
    detector_width_px = int(detector_width_mm / DETECTOR_PIXEL_SPACING)
    detector_height_px = int(detector_height_mm / DETECTOR_PIXEL_SPACING)
    
    # 90% scale
    anatomy_width = int(detector_width_px * 0.9)
    anatomy_height = int(detector_height_px * 0.9)
    
    # Fast zoom
    zoom_y = anatomy_height / projection.shape[0]
    zoom_x = anatomy_width / projection.shape[1]
    projection_scaled = ndimage.zoom(projection, [zoom_y, zoom_x], order=1)  # Linear for speed
    
    # Center
    detector_image = np.zeros((detector_height_px, detector_width_px))
    y_offset = (detector_height_px - anatomy_height) // 2
    x_offset = (detector_width_px - anatomy_width) // 2
    
    detector_image[y_offset:y_offset+anatomy_height,
                  x_offset:x_offset+anatomy_width] = projection_scaled
    
    return detector_image

def convert_to_radiograph_fast(projection):
    """Fast radiograph conversion"""
    # Beer-Lambert
    transmission = np.exp(-projection)
    intensity = -np.log10(transmission + 1e-10)
    
    # Simple normalization
    body_mask = projection > 0.1
    if np.any(body_mask):
        p5 = np.percentile(intensity[body_mask], 5)
        p95 = np.percentile(intensity[body_mask], 95)
        intensity = (intensity - p5) / (p95 - p5)
        intensity = np.clip(intensity, 0, 1)
    
    # Single gamma
    intensity = np.power(intensity, 0.8)
    
    # Air black
    intensity[projection < 0.05] = 0
    
    return intensity

def fast_depth_estimation(left_img, right_img, max_disparity=80):
    """Simplified fast depth estimation"""
    h, w = left_img.shape
    disparity_map = np.zeros((h, w))
    
    # Coarse block matching
    block_size = 9
    half_block = block_size // 2
    step = 8  # Large steps for speed
    
    valid_count = 0
    
    for y in range(half_block, h - half_block, step):
        for x in range(half_block + max_disparity, w - half_block, step):
            left_block = left_img[y-half_block:y+half_block+1,
                                x-half_block:x+half_block+1]
            
            min_ssd = float('inf')
            best_d = 0
            
            # Coarse search
            for d in range(0, min(max_disparity, x-half_block), 2):
                right_block = right_img[y-half_block:y+half_block+1,
                                      x-d-half_block:x-d+half_block+1]
                
                ssd = np.sum((left_block - right_block)**2)
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_d = d
            
            if min_ssd < 0.5:  # Simple threshold
                disparity_map[y:y+step, x:x+step] = best_d
                valid_count += 1
    
    # Simple smoothing
    disparity_map = ndimage.median_filter(disparity_map, size=3)
    
    coverage = (disparity_map > 0).sum() / (h * w) * 100
    
    return disparity_map, coverage

def generate_v12_fast_stereo(ct_volume):
    """Generate V12 fast stereo"""
    log_message(f"\n--- V12 Fast Optimized Stereo Generation ---")
    
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    
    log_message(f"Volume: {volume.shape}, spacing: {spacing} mm")
    
    # Tissue segmentation
    log_message("Applying tissue segmentation...")
    mu_volume = optimized_tissue_segmentation(volume)
    
    # Generate views
    half_angle = STEREO_ANGLE_DEGREES / 2
    
    # CENTER
    log_message("Generating CENTER view...")
    proj_center = generate_fast_projection(mu_volume, spacing)
    det_center = resample_to_detector_fast(proj_center)
    drr_center = convert_to_radiograph_fast(det_center)
    
    # LEFT
    log_message(f"Generating LEFT view (-{half_angle}°)...")
    rot_left = rotate_volume_fast(ct_volume, -half_angle)
    vol_left = sitk.GetArrayFromImage(rot_left)
    mu_left = optimized_tissue_segmentation(vol_left)
    proj_left = generate_fast_projection(mu_left, spacing)
    det_left = resample_to_detector_fast(proj_left)
    drr_left = convert_to_radiograph_fast(det_left)
    
    # RIGHT
    log_message(f"Generating RIGHT view (+{half_angle}°)...")
    rot_right = rotate_volume_fast(ct_volume, +half_angle)
    vol_right = sitk.GetArrayFromImage(rot_right)
    mu_right = optimized_tissue_segmentation(vol_right)
    proj_right = generate_fast_projection(mu_right, spacing)
    det_right = resample_to_detector_fast(proj_right)
    drr_right = convert_to_radiograph_fast(det_right)
    
    # Metrics
    diff = np.mean(np.abs(drr_left - drr_right))
    baseline = 2 * SOURCE_TO_DETECTOR_MM * np.sin(np.radians(half_angle))
    
    log_message(f"Stereo difference: {diff:.4f}")
    log_message(f"Baseline: {baseline:.1f}mm")
    
    return drr_left, drr_center, drr_right, baseline, diff

def save_v12_fast_outputs(images, metrics, patient_id):
    """Save V12 fast outputs"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    drr_left, drr_center, drr_right = images
    baseline, diff = metrics
    
    # Fast depth estimation
    log_message("\nEstimating depth...")
    disparity_map, coverage = fast_depth_estimation(drr_left, drr_right)
    log_message(f"Depth coverage: {coverage:.1f}%")
    
    # Save images
    for img, view in zip(images, ['left', 'center', 'right']):
        filename = f"{OUTPUT_DIR}/drr_{patient_id}_AP_{view}.png"
        plt.figure(figsize=(8, 10), facecolor='black')
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.title(f'{patient_id} - {view.capitalize()}', color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight', facecolor='black')
        plt.close()
    
    # Comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 20), facecolor='black')
    
    axes[0,0].imshow(drr_left, cmap='gray')
    axes[0,0].set_title(f'Left (-{STEREO_ANGLE_DEGREES/2}°)', color='white')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(drr_right, cmap='gray')
    axes[0,1].set_title(f'Right (+{STEREO_ANGLE_DEGREES/2}°)', color='white')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(drr_center, cmap='gray')
    axes[1,0].set_title('Center', color='white')
    axes[1,0].axis('off')
    
    im = axes[1,1].imshow(disparity_map, cmap='turbo')
    axes[1,1].set_title('Disparity Map', color='white')
    axes[1,1].axis('off')
    plt.colorbar(im, ax=axes[1,1], fraction=0.046)
    
    plt.suptitle(f'{patient_id} - V12 Fast Optimized\n'
                f'5° Stereo Angle, {baseline:.1f}mm Baseline',
                color='white', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_{patient_id}_AP.png", 
               dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved outputs to {OUTPUT_DIR}")

def main():
    """Main execution"""
    log_message("="*80)
    log_message("V12 Fast Optimized Stereo DRR Generator")
    log_message("="*80)
    log_message("Parameters:")
    log_message(f"  • Stereo angle: {STEREO_ANGLE_DEGREES}°")
    log_message(f"  • Resolution: {OUTPUT_DPI} DPI")
    log_message(f"  • Optimized for speed")
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
        
        # Generate fast stereo
        drr_left, drr_center, drr_right, baseline, diff = generate_v12_fast_stereo(ct_volume)
        
        # Save outputs
        images = (drr_left, drr_center, drr_right)
        metrics = (baseline, diff)
        save_v12_fast_outputs(images, metrics, dataset['patient_id'])
        
        total_time = time.time() - start_time
        
        log_message(f"\n{'='*80}")
        log_message(f"V12 Fast COMPLETE")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()