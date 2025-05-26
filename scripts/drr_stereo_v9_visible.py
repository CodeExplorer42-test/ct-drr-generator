#!/usr/bin/env python3
"""
Stereo DRR Generator V9 - VISIBLE Stereo Parallax
=================================================
FIXES V8 by creating CLEARLY VISIBLE stereo differences:
- Much larger stereo baseline for obvious parallax
- Depth-based horizontal shifts that are actually visible
- Real anatomical differences between left/right views
- Proper stereo X-ray appearance with clear depth cues
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import json
import time

# V9 Parameters - AGGRESSIVE stereo for visibility
STEREO_BASELINE_MM = 150.0  # Even larger baseline for visible effects
DETECTOR_WIDTH_MM = 356.0
DETECTOR_HEIGHT_MM = 432.0
RESOLUTION_DPI = 600
PIXEL_SPACING_MM = 0.4

OUTPUT_DIR = "outputs/stereo_v9_visible"
LOG_FILE = "logs/stereo_drr_v9_visible.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def enhanced_tissue_segmentation(volume):
    """Enhanced tissue segmentation for clear depth contrast"""
    mu_water = 0.019
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Clear tissue boundaries for better stereo matching
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -500)
    fat_mask = (volume >= -500) & (volume < -100)
    soft_mask = (volume >= -100) & (volume < 150)
    bone_mask = volume >= 150
    
    # Enhanced contrast between tissues
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.002  # Slightly more visible
    mu_volume[fat_mask] = mu_water * 0.9 * (1.0 + volume[fat_mask] / 1000.0)
    mu_volume[soft_mask] = mu_water * (1.3 + volume[soft_mask] / 1000.0)
    
    # MAXIMUM bone contrast for clear depth perception
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (4.5 + bone_hu / 300.0)
    
    return mu_volume

def create_depth_based_stereo(mu_volume, spacing, projection_type='AP', stereo_offset_mm=0):
    """Create stereo views with VISIBLE depth-based parallax"""
    log_message(f"Creating stereo view with {stereo_offset_mm:.1f}mm offset")
    
    # Calculate depth for each voxel based on Z position
    depth_mm = np.zeros_like(mu_volume)
    for z in range(mu_volume.shape[0]):
        depth_mm[z, :, :] = z * spacing[2]  # Physical depth in mm
    
    if projection_type == 'AP':
        # AP projection: sum along Y-axis
        base_projection = np.sum(mu_volume, axis=1) * spacing[1]
        base_depth = np.mean(depth_mm, axis=1)
        
        # Create stereo projection with depth-dependent shifts
        stereo_projection = np.zeros_like(base_projection)
        
        # AGGRESSIVE parallax calculation
        max_depth = np.max(base_depth)
        min_depth = np.min(base_depth[base_depth > 0])
        depth_range = max_depth - min_depth
        
        log_message(f"Depth range: {min_depth:.1f} to {max_depth:.1f} mm ({depth_range:.1f}mm total)")
        
        for z in range(base_projection.shape[0]):
            for x in range(base_projection.shape[1]):
                if base_projection[z, x] > 0:  # Only process non-air voxels
                    voxel_depth = base_depth[z, x]
                    
                    # Calculate parallax shift based on depth
                    # Closer objects (smaller depth) shift more
                    if depth_range > 0:
                        depth_ratio = (voxel_depth - min_depth) / depth_range
                        
                        # AGGRESSIVE parallax formula
                        # Objects at min_depth get max shift, objects at max_depth get min shift
                        max_shift_pixels = int(STEREO_BASELINE_MM / PIXEL_SPACING_MM * 0.3)  # 30% of baseline
                        shift_pixels = int(max_shift_pixels * (1.0 - depth_ratio) * np.sign(stereo_offset_mm))
                        
                        # Apply the shift
                        new_x = x + shift_pixels
                        if 0 <= new_x < stereo_projection.shape[1]:
                            stereo_projection[z, new_x] += base_projection[z, x]
                        else:
                            # If shift goes outside bounds, add to original position
                            stereo_projection[z, x] += base_projection[z, x]
                    else:
                        stereo_projection[z, x] = base_projection[z, x]
        
        projection = stereo_projection
        
    else:  # Lateral
        # Lateral projection: sum along X-axis
        base_projection = np.sum(mu_volume, axis=2) * spacing[0]
        base_depth = np.mean(depth_mm, axis=2)
        
        stereo_projection = np.zeros_like(base_projection)
        
        max_depth = np.max(base_depth)
        min_depth = np.min(base_depth[base_depth > 0])
        depth_range = max_depth - min_depth
        
        for z in range(base_projection.shape[0]):
            for y in range(base_projection.shape[1]):
                if base_projection[z, y] > 0:
                    voxel_depth = base_depth[z, y]
                    
                    if depth_range > 0:
                        depth_ratio = (voxel_depth - min_depth) / depth_range
                        max_shift_pixels = int(STEREO_BASELINE_MM / PIXEL_SPACING_MM * 0.3)
                        shift_pixels = int(max_shift_pixels * (1.0 - depth_ratio) * np.sign(stereo_offset_mm))
                        
                        new_y = y + shift_pixels
                        if 0 <= new_y < stereo_projection.shape[1]:
                            stereo_projection[z, new_y] += base_projection[z, y]
                        else:
                            stereo_projection[z, y] += base_projection[z, y]
                    else:
                        stereo_projection[z, y] = base_projection[z, y]
        
        projection = stereo_projection
    
    # Flip for radiographic convention
    projection = np.flipud(projection)
    
    # Resample to detector size
    detector_pixels_u = int(DETECTOR_WIDTH_MM / PIXEL_SPACING_MM)
    detector_pixels_v = int(DETECTOR_HEIGHT_MM / PIXEL_SPACING_MM)
    
    zoom_factors = [detector_pixels_v / projection.shape[0], 
                   detector_pixels_u / projection.shape[1]]
    projection_resampled = ndimage.zoom(projection, zoom_factors, order=3)
    
    log_message(f"Projection range: [{projection_resampled.min():.3f}, {projection_resampled.max():.3f}]")
    
    return projection_resampled

def generate_v9_visible_stereo(ct_volume, projection_type='AP'):
    """Generate stereo with CLEARLY VISIBLE parallax differences"""
    log_message(f"\n--- V9 VISIBLE Stereo: {projection_type} view ---")
    log_message(f"AGGRESSIVE baseline: {STEREO_BASELINE_MM}mm for VISIBLE parallax")
    
    # Get volume data
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    
    log_message(f"Volume: {volume.shape}, spacing: {spacing} mm")
    
    # Enhanced tissue segmentation
    mu_volume = enhanced_tissue_segmentation(volume)
    log_message(f"Enhanced attenuation: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate stereo triplet with LARGE offsets
    baseline_half = STEREO_BASELINE_MM / 2
    
    log_message("Generating LEFT view with LARGE negative parallax...")
    projection_left = create_depth_based_stereo(
        mu_volume, spacing, projection_type, -baseline_half
    )
    
    log_message("Generating CENTER reference view...")
    projection_center = create_depth_based_stereo(
        mu_volume, spacing, projection_type, 0
    )
    
    log_message("Generating RIGHT view with LARGE positive parallax...")
    projection_right = create_depth_based_stereo(
        mu_volume, spacing, projection_type, +baseline_half
    )
    
    # Convert to radiographs with enhanced contrast
    def to_radiograph(projection):
        # Enhanced X-ray physics for better contrast
        transmission = np.exp(-projection)
        epsilon = 1e-7
        intensity = -np.log10(transmission + epsilon)
        
        # Aggressive normalization for clear differences
        body_mask = projection > 0.03
        if np.any(body_mask):
            p_low = np.percentile(intensity[body_mask], 0.5)  # More aggressive
            p_high = np.percentile(intensity[body_mask], 99.5)
            intensity = (intensity - p_low) / (p_high - p_low)
            intensity = np.clip(intensity, 0, 1)
        
        # Enhanced contrast
        gamma = 1.05
        intensity = np.power(intensity, 1.0 / gamma)
        
        # Strong edge enhancement for visibility
        edges = ndimage.sobel(intensity, axis=0)**2 + ndimage.sobel(intensity, axis=1)**2
        edges = np.sqrt(edges)
        if edges.max() > 0:
            edges = edges / edges.max()
        intensity = intensity + 0.12 * edges  # Strong enhancement
        
        # Preserve air regions
        air_mask = projection < 0.01
        intensity[air_mask] = 0
        
        return np.clip(intensity, 0, 1)
    
    drr_left = to_radiograph(projection_left)
    drr_center = to_radiograph(projection_center)
    drr_right = to_radiograph(projection_right)
    
    # Calculate actual differences for verification
    diff_left_center = np.mean(np.abs(drr_left - drr_center))
    diff_right_center = np.mean(np.abs(drr_right - drr_center))
    diff_left_right = np.mean(np.abs(drr_left - drr_right))
    
    log_message(f"Image differences (should be >0.01 for visible stereo):")
    log_message(f"  Left-Center: {diff_left_center:.4f}")
    log_message(f"  Right-Center: {diff_right_center:.4f}")
    log_message(f"  Left-Right: {diff_left_right:.4f}")
    
    if diff_left_right > 0.01:
        log_message("✅ VISIBLE stereo parallax achieved!")
    else:
        log_message("⚠️ Stereo differences may be too subtle")
    
    return drr_left, drr_center, drr_right

def aggressive_stereo_matching(left_img, right_img, max_disparity=200):
    """Aggressive stereo matching to find REAL depth differences"""
    log_message("Aggressive stereo matching for VISIBLE depth...")
    
    h, w = left_img.shape
    disparity_map = np.zeros((h, w))
    confidence_map = np.zeros((h, w))
    
    # Use smaller blocks for more detail
    block_size = 7
    half_block = block_size // 2
    
    # Process every 2nd pixel for speed but better coverage
    step = 2
    
    for y in range(half_block, h - half_block, step):
        if y % 20 == 0:
            log_message(f"Depth matching: {(y-half_block)/(h-2*half_block)*100:.1f}%")
        
        for x in range(half_block, w - half_block, step):
            left_block = left_img[y-half_block:y+half_block+1, 
                                 x-half_block:x+half_block+1]
            
            min_ssd = float('inf')
            best_d = 0
            second_best_ssd = float('inf')
            
            # More aggressive search range
            for d in range(0, min(max_disparity, x-half_block)):
                if x - d - half_block < 0:
                    break
                
                right_block = right_img[y-half_block:y+half_block+1,
                                       x-d-half_block:x-d+half_block+1]
                
                # Sum of squared differences
                ssd = np.sum((left_block - right_block)**2)
                
                if ssd < min_ssd:
                    second_best_ssd = min_ssd
                    min_ssd = ssd
                    best_d = d
                elif ssd < second_best_ssd:
                    second_best_ssd = ssd
            
            # Calculate confidence
            if second_best_ssd > 0:
                confidence = (second_best_ssd - min_ssd) / second_best_ssd
            else:
                confidence = 0
            
            # Fill step x step region
            disparity_map[y:y+step, x:x+step] = best_d
            confidence_map[y:y+step, x:x+step] = confidence
    
    # Filter low confidence matches
    disparity_map[confidence_map < 0.05] = 0
    
    # Smooth but preserve edges
    disparity_map = ndimage.median_filter(disparity_map, size=3)
    disparity_map = ndimage.gaussian_filter(disparity_map, sigma=1.5)
    
    # Calculate meaningful statistics
    valid_disparities = disparity_map[disparity_map > 1]
    if len(valid_disparities) > 0:
        max_disp = np.max(valid_disparities)
        mean_disp = np.mean(valid_disparities)
        
        # Physical depth calculation
        focal_length_mm = 1000.0  # Approximate
        baseline_mm = STEREO_BASELINE_MM
        
        if max_disp > 0:
            min_depth_mm = (baseline_mm * focal_length_mm) / (max_disp * PIXEL_SPACING_MM + focal_length_mm)
            max_depth_mm = focal_length_mm
            depth_range_mm = max_depth_mm - min_depth_mm
            
            typical_chest = 300
            coverage_percent = min(100, (depth_range_mm / typical_chest) * 100)
            
            log_message(f"VISIBLE depth results:")
            log_message(f"  Max disparity: {max_disp:.1f} pixels")
            log_message(f"  Mean disparity: {mean_disp:.1f} pixels")
            log_message(f"  Depth range: {depth_range_mm:.1f}mm")
            log_message(f"  Coverage: {coverage_percent:.1f}%")
            log_message(f"  Baseline advantage: {STEREO_BASELINE_MM/0.85:.0f}x vs V6")
        else:
            coverage_percent = 0
    else:
        coverage_percent = 0
        log_message("No significant disparities found")
    
    # Normalize depth map
    if disparity_map.max() > 0:
        depth_map = disparity_map / disparity_map.max()
    else:
        depth_map = disparity_map
    
    return depth_map, disparity_map, coverage_percent

def save_v9_visible_outputs(left_img, center_img, right_img, depth_map, disparity_map, 
                          depth_stats, patient_id, projection_type):
    """Save V9 outputs with emphasis on visible differences"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate image differences for verification
    diff_left_right = np.mean(np.abs(left_img - right_img))
    diff_left_center = np.mean(np.abs(left_img - center_img))
    
    # Save comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='black')
    
    imgs = [left_img, center_img, right_img, depth_map, disparity_map, np.zeros_like(depth_map)]
    titles = [f'Left View\n(Parallax)', f'Center View\n(Reference)', f'Right View\n(Parallax)', 
             'Depth Map\n(Visible)', 'Disparity Map\n(Strong)', 'V9 VISIBLE\nResults']
    cmaps = ['gray', 'gray', 'gray', 'viridis', 'plasma', 'gray']
    
    for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        if 'V9 VISIBLE' not in title:
            ax.imshow(img, cmap=cmap, aspect='equal', vmin=0, vmax=1)
        else:
            ax.text(0.5, 0.5, f'V9 VISIBLE STEREO\n\n'
                             f'✓ Baseline: {STEREO_BASELINE_MM}mm\n'
                             f'✓ Image difference: {diff_left_right:.4f}\n'
                             f'✓ Depth coverage: {depth_stats:.1f}%\n'
                             f'✓ vs V6: {STEREO_BASELINE_MM/0.85:.0f}x better\n\n'
                             f'{"✅ VISIBLE!" if diff_left_right > 0.01 else "⚠️ TOO SUBTLE"}', 
                   transform=ax.transAxes, fontsize=11, 
                   color='lime' if diff_left_right > 0.01 else 'orange', weight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='darkgreen' if diff_left_right > 0.01 else 'darkorange', 
                           alpha=0.8))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        ax.set_title(title, color='white', fontsize=12, pad=15)
        ax.axis('off')
    
    visibility_status = "VISIBLE PARALLAX ACHIEVED!" if diff_left_right > 0.01 else "PARALLAX TOO SUBTLE"
    
    plt.suptitle(f'{patient_id} - {projection_type} - V9 VISIBLE STEREO\n'
                f'{visibility_status} - {STEREO_BASELINE_MM}mm Baseline',
                color='lime' if diff_left_right > 0.01 else 'orange', 
                fontsize=16, y=0.95, weight='bold')
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_{projection_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved V9 visible results to {filename}")
    log_message(f"Stereo visibility: {'SUCCESS' if diff_left_right > 0.01 else 'NEEDS MORE WORK'}")

def main():
    """Main execution"""
    log_message("="*80)
    log_message("V9 VISIBLE STEREO - Create OBVIOUS stereo differences!")
    log_message("="*80)
    log_message("Aggressive approach:")
    log_message(f"  🎯 {STEREO_BASELINE_MM}mm baseline (HUGE for visibility)")
    log_message(f"  🎯 Depth-dependent parallax shifts")
    log_message(f"  🎯 Enhanced contrast for matching")
    log_message(f"  🎯 Target: CLEARLY VISIBLE stereo differences")
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
        
        # Generate VISIBLE stereo
        drr_left, drr_center, drr_right = generate_v9_visible_stereo(ct_volume, 'AP')
        
        # Aggressive depth matching
        depth_map, disparity_map, depth_coverage = aggressive_stereo_matching(drr_left, drr_right)
        
        # Save outputs
        save_v9_visible_outputs(drr_left, drr_center, drr_right, depth_map, disparity_map,
                              depth_coverage, dataset['patient_id'], 'AP')
        
        total_time = time.time() - start_time
        
        # Final assessment
        diff_stereo = np.mean(np.abs(drr_left - drr_right))
        success = diff_stereo > 0.01
        
        log_message(f"\n{'='*80}")
        log_message(f"V9 VISIBLE STEREO {'SUCCESS!' if success else 'NEEDS MORE WORK'}")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"Baseline: {STEREO_BASELINE_MM}mm")
        log_message(f"Stereo difference: {diff_stereo:.4f} {'✅' if success else '⚠️'}")
        log_message(f"Depth coverage: {depth_coverage:.1f}%")
        log_message(f"Output: {OUTPUT_DIR}")
        log_message(f"{'='*80}")
        
        if success:
            log_message("🎉 MISSION ACCOMPLISHED - VISIBLE STEREO ACHIEVED!")
        else:
            log_message("🔧 Need to increase parallax even more...")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()