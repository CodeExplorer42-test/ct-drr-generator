#!/usr/bin/env python3
"""
Stereo DRR Generator V10 - Volume Rotation Approach
===================================================
CONSERVATIVE approach to true stereo through small volume rotations:
- Rotate CT volume by small angles (±1.5°) to simulate different viewing positions
- Generate normal parallel projections from each rotated volume  
- Creates real geometric differences without artificial manipulation
- Preserves anatomical accuracy while achieving true parallax
- Much faster than ray-casting, more accurate than image shifting

THIS TIME: No claims of success until visually verified by user
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import time

# V10 CONSERVATIVE Parameters
ROTATION_ANGLE_DEGREES = 1.5  # Small angle to avoid distortion
DETECTOR_WIDTH_MM = 356.0
DETECTOR_HEIGHT_MM = 432.0  
RESOLUTION_DPI = 600
PIXEL_SPACING_MM = 0.4

OUTPUT_DIR = "outputs/stereo_v10_rotation"
LOG_FILE = "logs/stereo_drr_v10_rotation.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def conservative_tissue_segmentation(volume):
    """Conservative tissue segmentation - no over-enhancement"""
    mu_water = 0.019
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Conservative tissue boundaries
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -500)
    soft_mask = (volume >= -500) & (volume < 150)
    bone_mask = volume >= 150
    
    # Natural attenuation values - no artificial enhancement
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.001
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # Modest bone enhancement - much less than previous versions
    bone_hu = volume[bone_mask] 
    mu_volume[bone_mask] = mu_water * (2.5 + bone_hu / 500.0)  # Conservative
    
    return mu_volume

def rotate_volume_carefully(sitk_volume, angle_degrees, axis='z'):
    """Carefully rotate volume with proper interpolation"""
    log_message(f"Rotating volume by {angle_degrees:.1f}° around {axis} axis")
    
    # Get volume center for rotation
    size = sitk_volume.GetSize()
    spacing = sitk_volume.GetSpacing()
    origin = sitk_volume.GetOrigin()
    
    # Calculate physical center
    center_physical = [
        origin[0] + size[0] * spacing[0] / 2,
        origin[1] + size[1] * spacing[1] / 2, 
        origin[2] + size[2] * spacing[2] / 2
    ]
    
    # Create rotation transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_physical)
    
    # Set rotation based on axis
    angle_rad = np.radians(angle_degrees)
    if axis == 'z':
        transform.SetRotation(0, 0, angle_rad)  # Rotation around Z-axis
    elif axis == 'y':
        transform.SetRotation(0, angle_rad, 0)  # Rotation around Y-axis
    elif axis == 'x':
        transform.SetRotation(angle_rad, 0, 0)  # Rotation around X-axis
    
    # Set up resampler with high-quality interpolation
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_volume)
    resampler.SetInterpolator(sitk.sitkBSpline)  # High-quality interpolation
    resampler.SetDefaultPixelValue(-1000)  # Air HU value
    resampler.SetTransform(transform)
    
    # Execute rotation
    rotated_volume = resampler.Execute(sitk_volume)
    
    # Check for rotation success
    original_array = sitk.GetArrayFromImage(sitk_volume)
    rotated_array = sitk.GetArrayFromImage(rotated_volume)
    
    log_message(f"Original volume range: [{original_array.min():.0f}, {original_array.max():.0f}] HU")
    log_message(f"Rotated volume range: [{rotated_array.min():.0f}, {rotated_array.max():.0f}] HU")
    
    # Verify rotation didn't corrupt data
    if rotated_array.max() < -900:  # All air values - rotation failed
        log_message("WARNING: Rotation may have failed - all values are air")
        return sitk_volume  # Return original if rotation failed
    
    return rotated_volume

def generate_standard_projection(mu_volume, spacing, projection_type='AP'):
    """Generate standard parallel projection - no tricks"""
    log_message(f"Generating standard {projection_type} projection")
    
    if projection_type == 'AP':
        # Sum along Y-axis (anterior to posterior)
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        proj_height_mm = projection.shape[0] * spacing[2]
        proj_width_mm = projection.shape[1] * spacing[0]
    else:  # Lateral
        # Sum along X-axis (right to left)
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        proj_height_mm = projection.shape[0] * spacing[2]
        proj_width_mm = projection.shape[1] * spacing[1]
    
    # Flip for radiographic convention
    projection = np.flipud(projection)
    
    # Resample to detector size
    detector_pixels_u = int(DETECTOR_WIDTH_MM / PIXEL_SPACING_MM)
    detector_pixels_v = int(DETECTOR_HEIGHT_MM / PIXEL_SPACING_MM)
    
    # Scale to fit detector
    scale = min(DETECTOR_WIDTH_MM / proj_width_mm, DETECTOR_HEIGHT_MM / proj_height_mm) * 0.9
    
    anatomy_width_px = int((proj_width_mm * scale) / PIXEL_SPACING_MM)
    anatomy_height_px = int((proj_height_mm * scale) / PIXEL_SPACING_MM)
    
    # High-quality resampling
    zoom_factors = [anatomy_height_px / projection.shape[0], 
                   anatomy_width_px / projection.shape[1]]
    projection_resampled = ndimage.zoom(projection, zoom_factors, order=3)
    
    # Center on detector
    detector_image = np.zeros((detector_pixels_v, detector_pixels_u))
    y_offset = (detector_pixels_v - anatomy_height_px) // 2
    x_offset = (detector_pixels_u - anatomy_width_px) // 2
    
    detector_image[y_offset:y_offset+anatomy_height_px, 
                  x_offset:x_offset+anatomy_width_px] = projection_resampled
    
    log_message(f"Projection range: [{detector_image.min():.3f}, {detector_image.max():.3f}]")
    
    return detector_image

def generate_v10_rotation_stereo(ct_volume, projection_type='AP'):
    """Generate stereo through careful volume rotation"""
    log_message(f"\n--- V10 ROTATION Stereo: {projection_type} view ---")
    log_message("CONSERVATIVE approach: Small volume rotations for true geometric stereo")
    log_message(f"Rotation angle: ±{ROTATION_ANGLE_DEGREES}° (very small to avoid distortion)")
    
    # Get volume data
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    
    log_message(f"Volume: {volume.shape}, spacing: {spacing} mm")
    
    # Conservative tissue segmentation
    mu_volume = conservative_tissue_segmentation(volume)
    log_message(f"Conservative attenuation: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate CENTER view (no rotation)
    log_message("Generating CENTER view (no rotation)...")
    projection_center = generate_standard_projection(mu_volume, spacing, projection_type)
    
    # Generate LEFT view (negative rotation)
    log_message("Generating LEFT view (small negative rotation)...")
    try:
        rotated_volume_left = rotate_volume_carefully(ct_volume, -ROTATION_ANGLE_DEGREES, 'z')
        rotated_array_left = sitk.GetArrayFromImage(rotated_volume_left)
        mu_volume_left = conservative_tissue_segmentation(rotated_array_left)
        projection_left = generate_standard_projection(mu_volume_left, spacing, projection_type)
    except Exception as e:
        log_message(f"LEFT rotation failed: {e}")
        projection_left = projection_center  # Fallback to center
    
    # Generate RIGHT view (positive rotation)
    log_message("Generating RIGHT view (small positive rotation)...")
    try:
        rotated_volume_right = rotate_volume_carefully(ct_volume, +ROTATION_ANGLE_DEGREES, 'z')
        rotated_array_right = sitk.GetArrayFromImage(rotated_volume_right)
        mu_volume_right = conservative_tissue_segmentation(rotated_array_right)
        projection_right = generate_standard_projection(mu_volume_right, spacing, projection_type)
    except Exception as e:
        log_message(f"RIGHT rotation failed: {e}")
        projection_right = projection_center  # Fallback to center
    
    # Convert to radiographs with natural processing
    def to_natural_radiograph(projection):
        # Standard X-ray physics - no artificial enhancement
        transmission = np.exp(-projection)
        epsilon = 1e-7
        intensity = -np.log10(transmission + epsilon)
        
        # Natural normalization
        body_mask = projection > 0.05
        if np.any(body_mask):
            p_low = np.percentile(intensity[body_mask], 2)   # Conservative
            p_high = np.percentile(intensity[body_mask], 98) # Conservative  
            intensity = (intensity - p_low) / (p_high - p_low)
            intensity = np.clip(intensity, 0, 1)
        
        # Natural gamma - no artificial enhancement
        gamma = 1.1
        intensity = np.power(intensity, 1.0 / gamma)
        
        # Preserve air regions
        air_mask = projection < 0.02
        intensity[air_mask] = 0
        
        return np.clip(intensity, 0, 1)
    
    drr_left = to_natural_radiograph(projection_left)
    drr_center = to_natural_radiograph(projection_center)
    drr_right = to_natural_radiograph(projection_right)
    
    # HONEST assessment of differences
    diff_left_center = np.mean(np.abs(drr_left - drr_center))
    diff_right_center = np.mean(np.abs(drr_right - drr_center))
    diff_left_right = np.mean(np.abs(drr_left - drr_right))
    
    log_message(f"\nHONEST ASSESSMENT of image differences:")
    log_message(f"  Left-Center: {diff_left_center:.4f}")
    log_message(f"  Right-Center: {diff_right_center:.4f}")
    log_message(f"  Left-Right: {diff_left_right:.4f}")
    
    # Conservative thresholds for meaningful differences
    if diff_left_right > 0.005:  # Very conservative threshold
        log_message("✅ Detectable stereo differences achieved")
        stereo_quality = "detectable"
    elif diff_left_right > 0.002:
        log_message("⚠️ Minimal stereo differences - may be too subtle")
        stereo_quality = "minimal"
    else:
        log_message("❌ No meaningful stereo differences detected")
        stereo_quality = "failed"
    
    log_message(f"Stereo quality assessment: {stereo_quality}")
    
    return drr_left, drr_center, drr_right, diff_left_right, stereo_quality

def conservative_stereo_matching(left_img, right_img, max_disparity=50):
    """Conservative stereo matching - no inflated claims"""
    log_message("Conservative stereo matching for honest depth assessment...")
    
    h, w = left_img.shape
    disparity_map = np.zeros((h, w))
    confidence_map = np.zeros((h, w))
    
    # Conservative block size
    block_size = 11
    half_block = block_size // 2
    
    # Process every 4th pixel for speed
    step = 4
    
    for y in range(half_block, h - half_block, step):
        if y % 50 == 0:
            log_message(f"Stereo progress: {(y-half_block)/(h-2*half_block)*100:.1f}%")
        
        for x in range(half_block, w - half_block, step):
            left_block = left_img[y-half_block:y+half_block+1, 
                                 x-half_block:x+half_block+1]
            
            min_ssd = float('inf')
            best_d = 0
            second_best = float('inf')
            
            # Conservative search range
            for d in range(0, min(max_disparity, x-half_block)):
                if x - d - half_block < 0:
                    break
                
                right_block = right_img[y-half_block:y+half_block+1,
                                       x-d-half_block:x-d+half_block+1]
                
                ssd = np.sum((left_block - right_block)**2)
                
                if ssd < min_ssd:
                    second_best = min_ssd
                    min_ssd = ssd
                    best_d = d
                elif ssd < second_best:
                    second_best = ssd
            
            # Conservative confidence calculation
            if second_best > 0:
                confidence = (second_best - min_ssd) / second_best
            else:
                confidence = 0
            
            # Fill step x step region
            disparity_map[y:y+step, x:x+step] = best_d
            confidence_map[y:y+step, x:x+step] = confidence
    
    # Filter out low confidence matches - be strict
    disparity_map[confidence_map < 0.1] = 0
    
    # Conservative smoothing
    disparity_map = ndimage.median_filter(disparity_map, size=3)
    disparity_map = ndimage.gaussian_filter(disparity_map, sigma=1)
    
    # HONEST depth assessment
    valid_disparities = disparity_map[disparity_map > 0.5]
    if len(valid_disparities) > 0:
        max_disp = np.max(valid_disparities)
        mean_disp = np.mean(valid_disparities)
        coverage = len(valid_disparities) / disparity_map.size * 100
        
        # Conservative physical depth calculation
        effective_baseline_mm = 2 * 1000 * np.sin(np.radians(ROTATION_ANGLE_DEGREES))  # Approximate
        
        log_message(f"CONSERVATIVE depth assessment:")
        log_message(f"  Effective baseline: {effective_baseline_mm:.1f}mm") 
        log_message(f"  Max disparity: {max_disp:.1f} pixels")
        log_message(f"  Mean disparity: {mean_disp:.1f} pixels")
        log_message(f"  Coverage: {coverage:.1f}% of pixels")
        
        # Honest assessment - don't inflate numbers
        if max_disp > 5 and coverage > 1:
            depth_quality = "meaningful"
        elif max_disp > 2 and coverage > 0.5:
            depth_quality = "limited"
        else:
            depth_quality = "negligible"
            
        log_message(f"Honest depth assessment: {depth_quality}")
        
    else:
        max_disp = 0
        depth_quality = "none"
        coverage = 0
        log_message("No significant disparities detected")
    
    # Normalize depth map conservatively
    if disparity_map.max() > 0:
        depth_map = disparity_map / disparity_map.max()
    else:
        depth_map = disparity_map
    
    return depth_map, disparity_map, depth_quality, coverage, max_disp

def save_v10_honest_outputs(left_img, center_img, right_img, depth_map, disparity_map, 
                          stereo_diff, stereo_quality, depth_quality, coverage, max_disp,
                          patient_id, projection_type):
    """Save V10 outputs with HONEST assessment"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save individual images for inspection
    save_radiograph(left_img, f"{OUTPUT_DIR}/drr_{patient_id}_{projection_type}_left.png", 
                   f"{patient_id} - {projection_type} - Left (-{ROTATION_ANGLE_DEGREES}°)")
    save_radiograph(center_img, f"{OUTPUT_DIR}/drr_{patient_id}_{projection_type}_center.png", 
                   f"{patient_id} - {projection_type} - Center (0°)")
    save_radiograph(right_img, f"{OUTPUT_DIR}/drr_{patient_id}_{projection_type}_right.png", 
                   f"{patient_id} - {projection_type} - Right (+{ROTATION_ANGLE_DEGREES}°)")
    
    # HONEST comparison image
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='black')
    
    imgs = [left_img, center_img, right_img, depth_map, disparity_map, np.zeros_like(depth_map)]
    titles = [f'Left View\n(-{ROTATION_ANGLE_DEGREES}°)', f'Center View\n(0°)', f'Right View\n(+{ROTATION_ANGLE_DEGREES}°)', 
             'Depth Map', 'Disparity Map', 'HONEST\nAssessment']
    cmaps = ['gray', 'gray', 'gray', 'viridis', 'plasma', 'gray']
    
    for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        if 'HONEST' not in title:
            ax.imshow(img, cmap=cmap, aspect='equal', vmin=0, vmax=1)
        else:
            # HONEST assessment text
            status_color = 'red' if stereo_quality == 'failed' else 'orange' if stereo_quality == 'minimal' else 'lightgreen'
            
            ax.text(0.5, 0.5, f'V10 ROTATION\nHONEST RESULTS\n\n'
                             f'Rotation: ±{ROTATION_ANGLE_DEGREES}°\n'
                             f'Stereo diff: {stereo_diff:.4f}\n'
                             f'Stereo quality: {stereo_quality}\n'
                             f'Max disparity: {max_disp:.1f}px\n'
                             f'Depth coverage: {coverage:.1f}%\n'
                             f'Depth quality: {depth_quality}\n\n'
                             f'{"❌ FAILED" if stereo_quality == "failed" else "⚠️ LIMITED" if stereo_quality == "minimal" else "✅ SUCCESS"}', 
                   transform=ax.transAxes, fontsize=10, 
                   color=status_color, weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='darkred' if stereo_quality == 'failed' 
                           else 'darkorange' if stereo_quality == 'minimal' else 'darkgreen', alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.set_title(title, color='white', fontsize=12, pad=15)
        ax.axis('off')
    
    # HONEST title
    success_text = "SUCCESS" if stereo_quality == "detectable" else "LIMITED" if stereo_quality == "minimal" else "FAILED"
    title_color = 'lightgreen' if stereo_quality == "detectable" else 'orange' if stereo_quality == "minimal" else 'red'
    
    plt.suptitle(f'{patient_id} - {projection_type} - V10 ROTATION STEREO: {success_text}\n'
                f'HONEST Assessment - ±{ROTATION_ANGLE_DEGREES}° Volume Rotation',
                color=title_color, fontsize=16, y=0.95, weight='bold')
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_{projection_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved HONEST assessment to {filename}")

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
    """Main execution with HONEST assessment"""
    log_message("="*80)
    log_message("V10 ROTATION Stereo DRR Generator - HONEST APPROACH")
    log_message("="*80)
    log_message("Conservative strategy:")
    log_message(f"  • Small volume rotation: ±{ROTATION_ANGLE_DEGREES}° (avoid distortion)")
    log_message(f"  • Standard parallel projections (no tricks)")
    log_message(f"  • HONEST assessment of results")
    log_message(f"  • No claims of success without visual verification")
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
        
        # Generate rotation-based stereo
        drr_left, drr_center, drr_right, stereo_diff, stereo_quality = generate_v10_rotation_stereo(ct_volume, 'AP')
        
        # Conservative depth matching
        depth_map, disparity_map, depth_quality, coverage, max_disp = conservative_stereo_matching(drr_left, drr_right)
        
        # Save with HONEST assessment
        save_v10_honest_outputs(drr_left, drr_center, drr_right, depth_map, disparity_map,
                               stereo_diff, stereo_quality, depth_quality, coverage, max_disp,
                               dataset['patient_id'], 'AP')
        
        total_time = time.time() - start_time
        
        # FINAL HONEST SUMMARY
        log_message(f"\n{'='*80}")
        log_message(f"V10 ROTATION - HONEST FINAL ASSESSMENT")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"Rotation angle: ±{ROTATION_ANGLE_DEGREES}°")
        log_message(f"Stereo difference: {stereo_diff:.4f}")
        log_message(f"Stereo quality: {stereo_quality}")
        log_message(f"Max disparity: {max_disp:.1f} pixels")
        log_message(f"Depth coverage: {coverage:.1f}%")
        log_message(f"Depth quality: {depth_quality}")
        log_message(f"Output: {OUTPUT_DIR}")
        
        if stereo_quality == "detectable" and depth_quality in ["meaningful", "limited"]:
            log_message("✅ V10 shows promise - requires USER VISUAL VERIFICATION")
        elif stereo_quality == "minimal":
            log_message("⚠️ V10 minimal results - may need parameter adjustment")
        else:
            log_message("❌ V10 failed to achieve meaningful stereo")
            
        log_message("👁️ AWAITING USER VISUAL VERIFICATION OF RESULTS")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()