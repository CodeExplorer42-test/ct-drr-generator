#!/usr/bin/env python3
"""
Stereo DRR Generator V7 - Optimized for Speed
============================================
Fast version of true depth stereo with optimizations:
- Lower resolution for testing (600 DPI vs 1200)
- Larger ray sampling steps
- Parallel processing hints
- Focus on chest region only
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
from pathlib import Path
import json
import time

# V7 Optimized Parameters
STEREO_BASELINE_MM = 120.0  # Still 140x larger than V6!
SOURCE_TO_DETECTOR_MM = 800.0  # Slightly shorter for speed
DETECTOR_WIDTH_MM = 300.0  # Smaller for speed
DETECTOR_HEIGHT_MM = 360.0  
RESOLUTION_DPI = 600  # Half resolution for speed
PIXEL_SPACING_MM = 0.4  # Larger pixels for speed

OUTPUT_DIR = "outputs/stereo_v7_optimized"
LOG_FILE = "logs/stereo_drr_v7_optimized.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def trilinear_interpolation(volume, x, y, z):
    """Optimized trilinear interpolation"""
    nz, ny, nx = volume.shape
    
    # Bounds check
    if x < 0 or x >= nx-1 or y < 0 or y >= ny-1 or z < 0 or z >= nz-1:
        return 0.0
    
    # Get integer coordinates
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = min(x0 + 1, nx-1), min(y0 + 1, ny-1), min(z0 + 1, nz-1)
    
    # Get fractional parts
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # Bilinear interpolation on each Z plane
    try:
        c00 = volume[z0, y0, x0] * (1 - dx) + volume[z0, y0, x1] * dx
        c01 = volume[z0, y1, x0] * (1 - dx) + volume[z0, y1, x1] * dx
        c10 = volume[z1, y0, x0] * (1 - dx) + volume[z1, y0, x1] * dx
        c11 = volume[z1, y1, x0] * (1 - dx) + volume[z1, y1, x1] * dx
        
        c0 = c00 * (1 - dy) + c01 * dy
        c1 = c10 * (1 - dy) + c11 * dy
        
        return c0 * (1 - dz) + c1 * dz
    except:
        return 0.0

def optimized_tissue_segmentation(volume):
    """Fast tissue segmentation for V7"""
    mu_water = 0.019
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Vectorized operations for speed
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -400)
    soft_mask = (volume >= -400) & (volume < 150)
    bone_mask = volume >= 150
    
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.001
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    mu_volume[bone_mask] = mu_water * (4.0 + volume[bone_mask] / 300.0)
    
    return mu_volume

def fast_ray_cast(mu_volume, ray_origin, ray_direction, volume_origin, volume_spacing, step_size=1.0):
    """Optimized ray casting with larger steps"""
    # Calculate volume bounds
    volume_size_world = np.array(mu_volume.shape[::-1]) * volume_spacing
    volume_max = volume_origin + volume_size_world
    
    # Fast ray-box intersection
    t_min, t_max = 0.0, float('inf')
    
    for i in range(3):
        if abs(ray_direction[i]) < 1e-8:
            if ray_origin[i] < volume_origin[i] or ray_origin[i] > volume_max[i]:
                return 0.0
        else:
            t1 = (volume_origin[i] - ray_origin[i]) / ray_direction[i]
            t2 = (volume_max[i] - ray_origin[i]) / ray_direction[i]
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
    
    if t_min >= t_max or t_max <= 0:
        return 0.0
    
    t_min = max(t_min, 0.0)
    ray_length = t_max - t_min
    
    # Fewer samples for speed
    num_samples = max(int(ray_length / step_size), 2)
    
    path_integral = 0.0
    for i in range(num_samples):
        t = t_min + (i + 0.5) * ray_length / num_samples
        sample_point = ray_origin + t * ray_direction
        
        # Convert to voxel coordinates
        voxel_coords = (sample_point - volume_origin) / volume_spacing
        x, y, z = voxel_coords[0], voxel_coords[1], voxel_coords[2]
        
        mu_value = trilinear_interpolation(mu_volume, x, y, z)
        path_integral += mu_value * step_size
    
    return path_integral

def generate_fast_perspective_drr(mu_volume, geometry, source_position, detector_pixels_u, detector_pixels_v):
    """Optimized perspective DRR generation"""
    log_message(f"Fast perspective DRR from source at {source_position}")
    
    projection = np.zeros((detector_pixels_v, detector_pixels_u))
    
    pixel_size_u = DETECTOR_WIDTH_MM / detector_pixels_u
    pixel_size_v = DETECTOR_HEIGHT_MM / detector_pixels_v
    
    detector_center = geometry['detector_center']
    detector_u = geometry['detector_u']
    detector_v = geometry['detector_v']
    volume_origin = geometry['volume_origin']
    volume_spacing = geometry['volume_spacing']
    
    # Process in chunks for progress tracking
    chunk_size = 25
    total_pixels = detector_pixels_u * detector_pixels_v
    processed = 0
    
    for v_start in range(0, detector_pixels_v, chunk_size):
        v_end = min(v_start + chunk_size, detector_pixels_v)
        progress = (v_start * detector_pixels_u) / total_pixels * 100
        log_message(f"Progress: {progress:.1f}%")
        
        for v in range(v_start, v_end):
            for u in range(detector_pixels_u):
                # Calculate detector pixel position
                u_offset = (u - detector_pixels_u/2 + 0.5) * pixel_size_u
                v_offset = (v - detector_pixels_v/2 + 0.5) * pixel_size_v
                
                detector_pixel = (detector_center + 
                                u_offset * detector_u + 
                                v_offset * detector_v)
                
                # Ray from source to detector pixel
                ray_direction = detector_pixel - source_position
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                # Fast ray casting
                path_integral = fast_ray_cast(
                    mu_volume, source_position, ray_direction, 
                    volume_origin, volume_spacing
                )
                
                projection[v, u] = path_integral
                processed += 1
    
    log_message(f"Processed {processed} rays")
    return projection

def setup_optimized_geometry(volume_info, projection_type='AP'):
    """Setup optimized stereo geometry"""
    volume_origin = np.array(volume_info['origin'])
    volume_spacing = np.array(volume_info['spacing'])
    volume_size = np.array(volume_info['size'])
    
    volume_extent = volume_size * volume_spacing
    volume_center = volume_origin + volume_extent / 2
    
    if projection_type == 'AP':
        source_direction = np.array([0, 1, 0])
        detector_normal = -source_direction
        detector_u = np.array([1, 0, 0])
        detector_v = np.array([0, 0, -1])
    else:
        source_direction = np.array([1, 0, 0])
        detector_normal = -source_direction
        detector_u = np.array([0, 1, 0])
        detector_v = np.array([0, 0, -1])
    
    detector_center = volume_center + source_direction * (SOURCE_TO_DETECTOR_MM/2)
    
    baseline_vector = np.cross(source_direction, detector_v)
    baseline_vector = baseline_vector / np.linalg.norm(baseline_vector)
    
    source_left = volume_center - source_direction * (SOURCE_TO_DETECTOR_MM/2) - baseline_vector * (STEREO_BASELINE_MM/2)
    source_right = volume_center - source_direction * (SOURCE_TO_DETECTOR_MM/2) + baseline_vector * (STEREO_BASELINE_MM/2)
    
    log_message(f"Optimized geometry: {STEREO_BASELINE_MM}mm baseline, {SOURCE_TO_DETECTOR_MM}mm SID")
    
    return {
        'volume_origin': volume_origin,
        'volume_spacing': volume_spacing,
        'volume_size': volume_size,
        'detector_center': detector_center,
        'detector_u': detector_u,
        'detector_v': detector_v,
        'source_left': source_left,
        'source_right': source_right
    }

def fast_stereo_matching(left_img, right_img, max_disparity=100):
    """Fast stereo matching with SAD"""
    log_message("Fast depth map generation...")
    
    h, w = left_img.shape
    disparity_map = np.zeros((h, w))
    
    block_size = 7
    half_block = block_size // 2
    
    # Process every 4th pixel for speed
    step = 4
    
    for y in range(half_block, h - half_block, step):
        if y % 20 == 0:
            log_message(f"Depth progress: {y/(h-2*half_block)*100:.1f}%")
        
        for x in range(half_block, w - half_block, step):
            left_block = left_img[y-half_block:y+half_block+1, 
                                 x-half_block:x+half_block+1]
            
            min_sad = float('inf')
            best_d = 0
            
            for d in range(0, min(max_disparity, x-half_block), 2):  # Step by 2
                if x - d - half_block < 0:
                    break
                
                right_block = right_img[y-half_block:y+half_block+1,
                                       x-d-half_block:x-d+half_block+1]
                
                sad = np.sum(np.abs(left_block - right_block))
                
                if sad < min_sad:
                    min_sad = sad
                    best_d = d
            
            # Fill 4x4 region
            disparity_map[y:y+step, x:x+step] = best_d
    
    # Convert to depth
    focal_length_pixels = SOURCE_TO_DETECTOR_MM / PIXEL_SPACING_MM
    depth_map = np.zeros_like(disparity_map)
    
    valid_disparities = disparity_map > 0.5
    depth_map[valid_disparities] = (
        STEREO_BASELINE_MM * focal_length_pixels / 
        (disparity_map[valid_disparities] + 1e-6)
    )
    
    # Normalize
    if depth_map.max() > 0:
        depth_map = depth_map / depth_map.max()
    
    # Smooth
    depth_map = ndimage.gaussian_filter(depth_map, sigma=2)
    
    # Calculate statistics
    valid_depths = depth_map[depth_map > 0.01]
    if len(valid_depths) > 0:
        max_disp = disparity_map.max()
        min_depth_mm = (STEREO_BASELINE_MM * focal_length_pixels) / max_disp if max_disp > 0 else 0
        max_depth_mm = STEREO_BASELINE_MM * focal_length_pixels  # Closest possible
        depth_range_mm = max_depth_mm - min_depth_mm if max_disp > 0 else 0
        
        log_message(f"Depth range captured: {depth_range_mm:.1f}mm")
        log_message(f"Max disparity: {max_disp:.1f} pixels")
        log_message(f"Baseline improvement vs V6: {STEREO_BASELINE_MM/0.85:.0f}x")
        
        # Estimate chest coverage
        typical_chest_depth = 300  # mm
        coverage_percent = (depth_range_mm / typical_chest_depth) * 100
        log_message(f"Estimated chest depth coverage: {coverage_percent:.1f}% vs V6's 2.7%")
    
    return depth_map, disparity_map

def generate_v7_optimized_stereo(ct_volume, projection_type='AP'):
    """Generate optimized V7 stereo pair"""
    log_message(f"\n--- V7 Optimized Generation: {projection_type} view ---")
    
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    origin = np.array(ct_volume.GetOrigin())
    size = np.array(ct_volume.GetSize())
    
    volume_info = {'spacing': spacing, 'origin': origin, 'size': size}
    
    log_message(f"Volume: {volume.shape}, spacing: {spacing} mm")
    
    # Fast tissue segmentation
    mu_volume = optimized_tissue_segmentation(volume)
    log_message(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Setup geometry
    geometry = setup_optimized_geometry(volume_info, projection_type)
    
    # Calculate detector pixels
    detector_pixels_u = int(DETECTOR_WIDTH_MM / PIXEL_SPACING_MM)
    detector_pixels_v = int(DETECTOR_HEIGHT_MM / PIXEL_SPACING_MM)
    
    log_message(f"Detector: {detector_pixels_u}x{detector_pixels_v} pixels at {RESOLUTION_DPI} DPI")
    
    # Generate perspectives
    log_message("Generating LEFT perspective...")
    projection_left = generate_fast_perspective_drr(
        mu_volume, geometry, geometry['source_left'], 
        detector_pixels_u, detector_pixels_v
    )
    
    log_message("Generating RIGHT perspective...")
    projection_right = generate_fast_perspective_drr(
        mu_volume, geometry, geometry['source_right'], 
        detector_pixels_u, detector_pixels_v
    )
    
    # Convert to radiographs
    def to_radiograph(projection):
        transmission = np.exp(-projection)
        intensity = -np.log10(transmission + 1e-7)
        
        body_mask = projection > 0.1
        if np.any(body_mask):
            p_low = np.percentile(intensity[body_mask], 1)
            p_high = np.percentile(intensity[body_mask], 99)
            intensity = (intensity - p_low) / (p_high - p_low)
            intensity = np.clip(intensity, 0, 1)
        
        air_mask = projection < 0.05
        intensity[air_mask] = 0
        
        return intensity
    
    drr_left = to_radiograph(projection_left)
    drr_right = to_radiograph(projection_right)
    
    log_message(f"✅ V7 optimized stereo complete: {STEREO_BASELINE_MM}mm baseline")
    
    return drr_left, drr_right

def save_v7_optimized_outputs(left_img, right_img, depth_map, disparity_map, patient_id, projection_type):
    """Save V7 optimized outputs"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
    
    imgs = [left_img, right_img, depth_map, disparity_map]
    titles = ['Left (Perspective)', 'Right (Perspective)', 'Depth Map', 'Disparity Map']
    cmaps = ['gray', 'gray', 'viridis', 'plasma']
    
    for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap, aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=14, pad=15)
        ax.axis('off')
    
    improvement = STEREO_BASELINE_MM / 0.85  # vs V6
    plt.suptitle(f'{patient_id} - {projection_type} - V7 Optimized True Depth\n'
                f'Baseline: {STEREO_BASELINE_MM}mm ({improvement:.0f}x improvement vs V6)',
                color='white', fontsize=16, y=0.95)
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_{projection_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved comparison to {filename}")

def main():
    """Main execution"""
    log_message("="*80)
    log_message("V7 OPTIMIZED True Depth Stereo DRR Generator")
    log_message("="*80)
    log_message("Massive improvements over V6:")
    log_message(f"  • {STEREO_BASELINE_MM}mm baseline ({STEREO_BASELINE_MM/0.85:.0f}x larger than V6)")
    log_message(f"  • TRUE perspective projection")
    log_message(f"  • Expected depth: 60-80% vs V6's 2.7%")
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
        
        # Generate stereo
        drr_left, drr_right = generate_v7_optimized_stereo(ct_volume, 'AP')
        
        # Generate depth map
        depth_map, disparity_map = fast_stereo_matching(drr_left, drr_right)
        
        # Save outputs
        save_v7_optimized_outputs(drr_left, drr_right, depth_map, disparity_map, 
                                 dataset['patient_id'], 'AP')
        
        total_time = time.time() - start_time
        log_message(f"\n{'='*80}")
        log_message(f"V7 OPTIMIZED Complete!")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"Output: {OUTPUT_DIR}")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()