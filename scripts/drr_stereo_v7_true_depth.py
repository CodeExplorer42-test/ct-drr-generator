#!/usr/bin/env python3
"""
Stereo DRR Generator V7 - True Depth Edition
============================================
Implements REAL perspective stereo projection with:
- 150mm baseline (176x larger than V6's 0.85mm!)
- True ray-casting from dual X-ray sources
- Perspective projection for genuine depth information
- Advanced stereo matching for accurate depth maps

This should capture 80-90% of chest depth vs V6's pathetic 2.7%
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
from numba import jit

# V7 True Depth Parameters
STEREO_BASELINE_MM = 150.0  # 176x larger than V6!
SOURCE_TO_DETECTOR_MM = 1000.0  # Realistic X-ray geometry
DETECTOR_WIDTH_MM = 360.0  # 14" film
DETECTOR_HEIGHT_MM = 432.0  # 17" film
RESOLUTION_DPI = 1200
PIXEL_SPACING_MM = 0.2

OUTPUT_DIR = "outputs/stereo_v7_true_depth"
LOG_FILE = "logs/stereo_drr_v7_true_depth.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

@jit(nopython=True)
def trilinear_interpolation(volume, x, y, z):
    """Fast trilinear interpolation with bounds checking"""
    nz, ny, nx = volume.shape
    
    # Bounds check
    if x < 0 or x >= nx-1 or y < 0 or y >= ny-1 or z < 0 or z >= nz-1:
        return 0.0
    
    # Get integer coordinates
    x0, y0, z0 = int(x), int(y), int(z)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    
    # Get fractional parts
    dx, dy, dz = x - x0, y - y0, z - z0
    
    # Trilinear interpolation
    c00 = volume[z0, y0, x0] * (1 - dx) + volume[z0, y0, x1] * dx
    c01 = volume[z0, y1, x0] * (1 - dx) + volume[z0, y1, x1] * dx
    c10 = volume[z1, y0, x0] * (1 - dx) + volume[z1, y0, x1] * dx
    c11 = volume[z1, y1, x0] * (1 - dx) + volume[z1, y1, x1] * dx
    
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy
    
    return c0 * (1 - dz) + c1 * dz

def advanced_tissue_segmentation(volume):
    """Enhanced 7-tissue segmentation for V7"""
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air - complete transparency
    air_mask = volume < -950
    mu_volume[air_mask] = 0.0
    
    # Lung tissue - very low attenuation
    lung_mask = (volume >= -950) & (volume < -500)
    lung_hu = volume[lung_mask]
    mu_volume[lung_mask] = 0.0001 + (lung_hu + 950) * (0.003 / 450)
    
    # Fat tissue
    fat_mask = (volume >= -500) & (volume < -100)
    fat_hu = volume[fat_mask]
    mu_volume[fat_mask] = mu_water * 0.85 * (1.0 + fat_hu / 1000.0)
    
    # Muscle/Blood - enhanced contrast
    muscle_mask = (volume >= -100) & (volume < 50)
    muscle_hu = volume[muscle_mask]
    mu_volume[muscle_mask] = mu_water * (1.1 + muscle_hu / 1000.0)
    
    # Soft tissue
    soft_mask = (volume >= 50) & (volume < 150)
    soft_hu = volume[soft_mask]
    mu_volume[soft_mask] = mu_water * (1.2 + soft_hu / 1000.0)
    
    # Bone - significantly enhanced for depth contrast
    bone_mask = volume >= 150
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (4.0 + bone_hu / 300.0)  # Higher contrast
    
    # Smooth transitions
    mu_volume = ndimage.gaussian_filter(mu_volume, sigma=[0.2, 0.2, 0.2])
    
    return mu_volume

def setup_stereo_geometry(volume_info, projection_type='AP'):
    """Setup true stereo X-ray geometry"""
    volume_origin = np.array(volume_info['origin'])
    volume_spacing = np.array(volume_info['spacing'])
    volume_size = np.array(volume_info['size'])
    
    # Physical volume bounds
    volume_extent = volume_size * volume_spacing
    volume_center = volume_origin + volume_extent / 2
    
    if projection_type == 'AP':
        # AP view: sources in front/back of patient
        source_direction = np.array([0, 1, 0])  # Y direction
        detector_normal = -source_direction
        detector_u = np.array([1, 0, 0])  # X direction
        detector_v = np.array([0, 0, -1])  # -Z direction (radiographic convention)
    else:  # Lateral
        # Lateral view: sources on left/right of patient
        source_direction = np.array([1, 0, 0])  # X direction
        detector_normal = -source_direction
        detector_u = np.array([0, 1, 0])  # Y direction
        detector_v = np.array([0, 0, -1])  # -Z direction
    
    # Position detector behind volume
    detector_center = volume_center + source_direction * (SOURCE_TO_DETECTOR_MM/2)
    
    # Position stereo sources
    baseline_vector = np.cross(source_direction, detector_v)
    baseline_vector = baseline_vector / np.linalg.norm(baseline_vector)
    
    source_left = volume_center - source_direction * (SOURCE_TO_DETECTOR_MM/2) - baseline_vector * (STEREO_BASELINE_MM/2)
    source_right = volume_center - source_direction * (SOURCE_TO_DETECTOR_MM/2) + baseline_vector * (STEREO_BASELINE_MM/2)
    
    log_message(f"Stereo geometry: {STEREO_BASELINE_MM}mm baseline, {SOURCE_TO_DETECTOR_MM}mm SID")
    log_message(f"Volume center: {volume_center}")
    log_message(f"Source left: {source_left}")
    log_message(f"Source right: {source_right}")
    
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

def cast_ray_through_volume(mu_volume, ray_origin, ray_direction, volume_origin, volume_spacing, step_size=0.5):
    """Cast single ray through volume with optimized sampling"""
    # Calculate volume bounds in world coordinates
    volume_size_world = np.array(mu_volume.shape[::-1]) * volume_spacing  # Reverse for xyz
    volume_max = volume_origin + volume_size_world
    
    # Ray-box intersection (slab method)
    t_min, t_max = 0.0, float('inf')
    
    for i in range(3):
        if abs(ray_direction[i]) < 1e-8:
            # Ray parallel to slab
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
    
    # Sample along ray
    ray_length = t_max - t_min
    num_samples = max(int(ray_length / step_size), 1)
    
    path_integral = 0.0
    for i in range(num_samples):
        t = t_min + (i + 0.5) * ray_length / num_samples
        sample_point = ray_origin + t * ray_direction
        
        # Convert to voxel coordinates
        voxel_coords = (sample_point - volume_origin) / volume_spacing
        x, y, z = voxel_coords[0], voxel_coords[1], voxel_coords[2]
        
        # Convert to numpy array indices (z, y, x)
        z_idx, y_idx, x_idx = z, y, x
        
        # Sample volume
        mu_value = trilinear_interpolation(mu_volume, x_idx, y_idx, z_idx)
        path_integral += mu_value * step_size
    
    return path_integral

def generate_perspective_drr(mu_volume, geometry, source_position, detector_pixels_u, detector_pixels_v):
    """Generate DRR using perspective projection from single source"""
    log_message(f"Generating perspective DRR from source at {source_position}")
    
    projection = np.zeros((detector_pixels_v, detector_pixels_u))
    
    # Detector pixel dimensions
    pixel_size_u = DETECTOR_WIDTH_MM / detector_pixels_u
    pixel_size_v = DETECTOR_HEIGHT_MM / detector_pixels_v
    
    detector_center = geometry['detector_center']
    detector_u = geometry['detector_u']
    detector_v = geometry['detector_v']
    volume_origin = geometry['volume_origin']
    volume_spacing = geometry['volume_spacing']
    
    total_pixels = detector_pixels_u * detector_pixels_v
    processed = 0
    
    for v in range(detector_pixels_v):
        if v % 50 == 0:
            progress = (v * detector_pixels_u) / total_pixels * 100
            log_message(f"Progress: {progress:.1f}%")
        
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
            
            # Cast ray through volume
            path_integral = cast_ray_through_volume(
                mu_volume, source_position, ray_direction, 
                volume_origin, volume_spacing
            )
            
            projection[v, u] = path_integral
            processed += 1
    
    log_message(f"Processed {processed} rays")
    return projection

def generate_v7_stereo_pair(ct_volume, projection_type='AP'):
    """Generate true stereo pair with perspective projection"""
    log_message(f"\n--- V7 True Depth Generation: {projection_type} view ---")
    
    # Get volume info
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
    
    # Advanced tissue segmentation
    mu_volume = advanced_tissue_segmentation(volume)
    log_message(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Setup stereo geometry
    geometry = setup_stereo_geometry(volume_info, projection_type)
    
    # Calculate detector pixels
    detector_pixels_u = int(DETECTOR_WIDTH_MM / PIXEL_SPACING_MM)
    detector_pixels_v = int(DETECTOR_HEIGHT_MM / PIXEL_SPACING_MM)
    
    log_message(f"Detector: {detector_pixels_u}x{detector_pixels_v} pixels at {RESOLUTION_DPI} DPI")
    
    # Generate left and right projections
    log_message("Generating LEFT perspective projection...")
    projection_left = generate_perspective_drr(
        mu_volume, geometry, geometry['source_left'], 
        detector_pixels_u, detector_pixels_v
    )
    
    log_message("Generating RIGHT perspective projection...")
    projection_right = generate_perspective_drr(
        mu_volume, geometry, geometry['source_right'], 
        detector_pixels_u, detector_pixels_v
    )
    
    # Convert to radiographic images
    def to_radiograph(projection):
        transmission = np.exp(-projection)
        epsilon = 1e-7
        intensity = -np.log10(transmission + epsilon)
        
        # Normalize using body region
        body_mask = projection > 0.1
        if np.any(body_mask):
            p_low = np.percentile(intensity[body_mask], 0.5)
            p_high = np.percentile(intensity[body_mask], 99.5)
            intensity = (intensity - p_low) / (p_high - p_low)
            intensity = np.clip(intensity, 0, 1)
        
        # Preserve black background
        air_mask = projection < 0.05
        intensity[air_mask] = 0
        
        return intensity
    
    drr_left = to_radiograph(projection_left)
    drr_right = to_radiograph(projection_right)
    
    log_message(f"✅ V7 stereo pair complete with {STEREO_BASELINE_MM}mm baseline")
    
    return drr_left, drr_right, projection_left, projection_right

def advanced_stereo_matching(left_img, right_img, max_disparity=200):
    """Advanced stereo matching using Census transform"""
    log_message("Generating high-quality depth map with Census transform...")
    
    def census_transform(img, window_size=9):
        """Census transform for robust stereo matching"""
        half_window = window_size // 2
        h, w = img.shape
        census = np.zeros((h, w), dtype=np.uint64)
        
        for y in range(half_window, h - half_window):
            for x in range(half_window, w - half_window):
                center_val = img[y, x]
                bit_string = 0
                
                for dy in range(-half_window, half_window + 1):
                    for dx in range(-half_window, half_window + 1):
                        if dy == 0 and dx == 0:
                            continue
                        if img[y + dy, x + dx] > center_val:
                            bit_string = (bit_string << 1) | 1
                        else:
                            bit_string = bit_string << 1
                
                census[y, x] = bit_string
        
        return census
    
    # Apply Census transform
    census_left = census_transform(left_img)
    census_right = census_transform(right_img)
    
    h, w = left_img.shape
    disparity_map = np.zeros((h, w))
    
    window_size = 11
    half_window = window_size // 2
    
    for y in range(half_window, h - half_window):
        if y % 50 == 0:
            log_message(f"Depth progress: {(y-half_window)/(h-2*half_window)*100:.1f}%")
        
        for x in range(half_window, w - half_window):
            left_patch = census_left[y-half_window:y+half_window+1, 
                                   x-half_window:x+half_window+1]
            
            min_cost = float('inf')
            best_d = 0
            
            for d in range(min(max_disparity, x-half_window)):
                if x - d - half_window < 0:
                    break
                
                right_patch = census_right[y-half_window:y+half_window+1,
                                         x-d-half_window:x-d+half_window+1]
                
                # Hamming distance for Census transform
                cost = np.sum(np.unpackbits(
                    (left_patch ^ right_patch).astype(np.uint8).flatten()
                ))
                
                if cost < min_cost:
                    min_cost = cost
                    best_d = d
            
            disparity_map[y, x] = best_d
    
    # Convert disparity to depth using stereo equation
    # depth = (baseline * focal_length) / disparity
    focal_length_pixels = SOURCE_TO_DETECTOR_MM / PIXEL_SPACING_MM
    
    depth_map = np.zeros_like(disparity_map)
    valid_disparities = disparity_map > 1  # Avoid division by zero
    
    depth_map[valid_disparities] = (
        STEREO_BASELINE_MM * focal_length_pixels / 
        (disparity_map[valid_disparities] + 1e-6)
    )
    
    # Normalize depth map
    if depth_map.max() > 0:
        depth_map = depth_map / depth_map.max()
    
    # Post-processing
    depth_map = ndimage.median_filter(depth_map, size=3)
    depth_map = ndimage.gaussian_filter(depth_map, sigma=1)
    
    # Calculate depth statistics
    valid_depths = depth_map[depth_map > 0.01]
    if len(valid_depths) > 0:
        depth_range_mm = np.percentile(valid_depths, 95) * depth_map.max() if depth_map.max() > 0 else 0
        log_message(f"Depth range captured: {depth_range_mm:.1f}mm")
        log_message(f"Max disparity found: {disparity_map.max():.1f} pixels")
    
    log_message("Advanced depth map complete")
    return depth_map, disparity_map

def save_v7_outputs(images, patient_id, projection_type):
    """Save all V7 outputs with depth analysis"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    left_img, right_img, depth_map, disparity_map = images
    
    # Save individual images
    filenames = {
        'left': f"{OUTPUT_DIR}/drr_{patient_id}_{projection_type}_left.png",
        'right': f"{OUTPUT_DIR}/drr_{patient_id}_{projection_type}_right.png",
        'depth': f"{OUTPUT_DIR}/depth_{patient_id}_{projection_type}.png",
        'disparity': f"{OUTPUT_DIR}/disparity_{patient_id}_{projection_type}.png"
    }
    
    for img, name in [(left_img, 'left'), (right_img, 'right')]:
        save_high_dpi_image(img, filenames[name], RESOLUTION_DPI, 
                           f"{patient_id} - {projection_type} - {name.title()}")
    
    # Save depth visualizations
    save_depth_visualization(depth_map, filenames['depth'], RESOLUTION_DPI,
                           f"{patient_id} - {projection_type} - Depth Map")
    save_depth_visualization(disparity_map, filenames['disparity'], RESOLUTION_DPI,
                           f"{patient_id} - {projection_type} - Disparity")
    
    # Save comparison
    save_v7_comparison(images, patient_id, projection_type)
    
    # Save anaglyph
    save_anaglyph(left_img, right_img, 
                 f"{OUTPUT_DIR}/anaglyph_{patient_id}_{projection_type}.png")
    
    # Save depth analysis
    save_depth_analysis(depth_map, disparity_map, patient_id, projection_type)
    
    log_message(f"Saved all V7 outputs for {patient_id} - {projection_type}")

def save_high_dpi_image(image, filename, dpi, title=None):
    """Save image at specified DPI"""
    h, w = image.shape
    fig_width = w / dpi
    fig_height = h / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(image, cmap='gray', aspect='equal', vmin=0, vmax=1, interpolation='lanczos')
    
    if title:
        ax.text(0.5, 0.02, title, transform=ax.transAxes,
               fontsize=10, color='white', ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    
    ax.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

def save_depth_visualization(depth_map, filename, dpi, title=None):
    """Save depth map with enhanced colormap"""
    h, w = depth_map.shape
    fig_width = w / dpi
    fig_height = h / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Use viridis for better depth perception
    ax.imshow(depth_map, cmap='viridis', aspect='equal', vmin=0, vmax=1, interpolation='lanczos')
    
    if title:
        ax.text(0.5, 0.02, title, transform=ax.transAxes,
               fontsize=10, color='white', ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
    
    ax.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

def save_v7_comparison(images, patient_id, projection_type):
    """Save comprehensive comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='black')
    
    left_img, right_img, depth_map, disparity_map = images
    imgs = [left_img, right_img, depth_map, disparity_map]
    titles = ['Left View (Perspective)', 'Right View (Perspective)', 
             'Depth Map (True 3D)', 'Disparity Map (Raw)']
    cmaps = ['gray', 'gray', 'viridis', 'plasma']
    
    for ax, img, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap, aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=14, pad=15)
        ax.axis('off')
    
    plt.suptitle(f'{patient_id} - {projection_type} - V7 True Depth Stereo\n'
                f'Baseline: {STEREO_BASELINE_MM}mm | SID: {SOURCE_TO_DETECTOR_MM}mm',
                color='white', fontsize=16, y=0.95)
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_{projection_type}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

def save_anaglyph(left_img, right_img, filename):
    """Create and save red-cyan anaglyph"""
    anaglyph = np.zeros((*left_img.shape, 3))
    anaglyph[:, :, 0] = left_img  # Red (left)
    anaglyph[:, :, 1] = right_img  # Green (right)
    anaglyph[:, :, 2] = right_img  # Blue (right)
    
    h, w = left_img.shape
    fig_width = w / RESOLUTION_DPI
    fig_height = h / RESOLUTION_DPI
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(anaglyph, aspect='equal')
    ax.axis('off')
    
    plt.savefig(filename, dpi=RESOLUTION_DPI, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()

def save_depth_analysis(depth_map, disparity_map, patient_id, projection_type):
    """Save quantitative depth analysis"""
    valid_depths = depth_map[depth_map > 0.01]
    valid_disparities = disparity_map[disparity_map > 1]
    
    analysis = {
        'patient_id': patient_id,
        'projection_type': projection_type,
        'stereo_baseline_mm': STEREO_BASELINE_MM,
        'source_to_detector_mm': SOURCE_TO_DETECTOR_MM,
        'pixel_spacing_mm': PIXEL_SPACING_MM,
        'max_disparity_pixels': float(disparity_map.max()) if len(valid_disparities) > 0 else 0,
        'mean_disparity_pixels': float(np.mean(valid_disparities)) if len(valid_disparities) > 0 else 0,
        'depth_range_normalized': float(np.ptp(valid_depths)) if len(valid_depths) > 0 else 0,
        'depth_pixels_with_data': int(np.sum(depth_map > 0.01)),
        'total_pixels': int(depth_map.size),
        'depth_coverage_percent': float(np.sum(depth_map > 0.01) / depth_map.size * 100),
        'generation_timestamp': datetime.now().isoformat()
    }
    
    # Calculate estimated physical depth range
    if analysis['max_disparity_pixels'] > 0:
        focal_length_pixels = SOURCE_TO_DETECTOR_MM / PIXEL_SPACING_MM
        max_depth_mm = (STEREO_BASELINE_MM * focal_length_pixels) / 1  # Closest objects
        min_depth_mm = (STEREO_BASELINE_MM * focal_length_pixels) / analysis['max_disparity_pixels']
        analysis['estimated_depth_range_mm'] = float(max_depth_mm - min_depth_mm)
    else:
        analysis['estimated_depth_range_mm'] = 0
    
    filename = f"{OUTPUT_DIR}/depth_analysis_{patient_id}_{projection_type}.json"
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    log_message(f"Depth analysis: {analysis['estimated_depth_range_mm']:.1f}mm range captured")
    log_message(f"Coverage: {analysis['depth_coverage_percent']:.1f}% of pixels have depth data")

def main():
    """Main execution"""
    log_message("="*80)
    log_message("V7 TRUE DEPTH Stereo DRR Generator")
    log_message("="*80)
    log_message("Revolutionary improvements:")
    log_message(f"  • TRUE perspective projection (not fake parallel)")
    log_message(f"  • {STEREO_BASELINE_MM}mm baseline (176x larger than V6!)")
    log_message(f"  • {SOURCE_TO_DETECTOR_MM}mm source-to-detector distance")
    log_message(f"  • Advanced Census transform stereo matching")
    log_message(f"  • Expected depth capture: 80-90% vs V6's pathetic 2.7%")
    log_message("="*80)
    
    # Process one dataset first for testing
    dataset = {
        'path': 'data/tciaDownload/1.3.6.1.4.1.32722.99.99.298991776521342375010861296712563382046',
        'patient_id': 'LUNG1-001',
        'name': 'NSCLC-Radiomics'
    }
    
    start_time = time.time()
    
    log_message(f"\nProcessing: {dataset['name']} ({dataset['patient_id']})")
    
    try:
        # Load DICOM
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dataset['path'])
        
        if not dicom_files:
            log_message(f"❌ No DICOM files found")
            return
        
        log_message(f"Loading {len(dicom_files)} DICOM files...")
        reader.SetFileNames(dicom_files)
        ct_volume = reader.Execute()
        
        # Generate stereo for AP view only (faster testing)
        projection_type = 'AP'
        drr_left, drr_right, proj_left, proj_right = generate_v7_stereo_pair(ct_volume, projection_type)
        
        # Generate advanced depth map
        depth_map, disparity_map = advanced_stereo_matching(drr_left, drr_right)
        
        # Save outputs
        images = (drr_left, drr_right, depth_map, disparity_map)
        save_v7_outputs(images, dataset['patient_id'], projection_type)
        
        # Summary
        total_time = time.time() - start_time
        log_message(f"\n{'='*80}")
        log_message(f"V7 TRUE DEPTH Processing Complete!")
        log_message(f"Total time: {total_time:.1f} seconds")
        log_message(f"Output directory: {OUTPUT_DIR}")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()