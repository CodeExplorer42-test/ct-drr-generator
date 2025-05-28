#!/usr/bin/env python3
"""
Stereo DRR Generator V13 - True Stereo Geometry
===============================================
Implements true dual-source stereo based on research:
- Two X-ray sources with convergent geometry
- Simplified ray casting (not full Siddon's but proper rays)
- Correct attenuation model from research
- Controlled ray step size (0.75mm)
- Proper stereo baseline calculation

This is a practical implementation balancing accuracy and speed.
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
import time

# Research-based parameters
STEREO_ANGLE_DEGREES = 5.0  # Within optimal 3-7° range
SOURCE_TO_DETECTOR_MM = 1200.0  # Middle of 1000-1500mm range
RAY_STEP_SIZE_MM = 0.75  # Within optimal 0.5-1.0mm range

# Detector specifications
DETECTOR_WIDTH_MM = 356.0   # 14"
DETECTOR_HEIGHT_MM = 432.0  # 17"
DETECTOR_PIXELS_U = 712     # Practical resolution
DETECTOR_PIXELS_V = 864

OUTPUT_DIR = "outputs/stereo_v13_true_geometry"
LOG_FILE = "logs/stereo_drr_v13_true_geometry.log"

def log_message(message):
    """Log messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(log_entry + "\n")

def correct_hu_to_attenuation(hu_values):
    """Research-based HU to attenuation conversion"""
    # From research: attenuation = (HU + 1000) / 1000
    # But we need physical attenuation coefficients in mm^-1
    
    # Water attenuation at ~70 keV
    mu_water = 0.019  # mm^-1
    
    # Correct conversion preserving tissue differences
    # Linear attenuation coefficient relative to water
    relative_attenuation = (hu_values + 1000) / 1000
    
    # Convert to physical attenuation
    mu_values = mu_water * relative_attenuation
    
    # Tissue-specific adjustments for better contrast
    # Air/lung
    mu_values[hu_values < -900] = 0.0  # Air
    mu_values[(hu_values >= -900) & (hu_values < -500)] *= 0.1  # Lung
    
    # Bone enhancement (research suggests 300-500 HU threshold)
    bone_mask = hu_values > 300
    mu_values[bone_mask] *= 2.5  # Enhance bone visibility
    
    # Clamp to reasonable range
    mu_values = np.clip(mu_values, 0, 0.15)
    
    return mu_values

def calculate_stereo_source_positions(volume_center, baseline_mm):
    """Calculate true stereo source positions"""
    # Sources are positioned symmetrically around the center
    # For convergent geometry, sources point toward volume center
    
    # Source distance from volume center
    source_distance = SOURCE_TO_DETECTOR_MM
    
    # Left source: -X offset
    source_left = volume_center.copy()
    source_left[0] -= baseline_mm / 2  # X offset
    source_left[1] -= source_distance   # Y offset (behind patient)
    
    # Right source: +X offset
    source_right = volume_center.copy()
    source_right[0] += baseline_mm / 2  # X offset
    source_right[1] -= source_distance  # Y offset (behind patient)
    
    # Center source for reference
    source_center = volume_center.copy()
    source_center[1] -= source_distance
    
    return source_left, source_center, source_right

def create_detector_positions(volume_center):
    """Create detector pixel positions in world coordinates"""
    # Detector is positioned in front of patient
    detector_center = volume_center.copy()
    detector_center[1] += SOURCE_TO_DETECTOR_MM / 2  # Front of patient
    
    # Detector basis vectors
    detector_u = np.array([1, 0, 0])  # X direction
    detector_v = np.array([0, 0, -1])  # -Z for radiographic convention
    
    # Pixel spacing
    pixel_spacing_u = DETECTOR_WIDTH_MM / DETECTOR_PIXELS_U
    pixel_spacing_v = DETECTOR_HEIGHT_MM / DETECTOR_PIXELS_V
    
    # Create mesh of detector positions
    u_coords = np.linspace(-DETECTOR_WIDTH_MM/2, DETECTOR_WIDTH_MM/2, DETECTOR_PIXELS_U)
    v_coords = np.linspace(-DETECTOR_HEIGHT_MM/2, DETECTOR_HEIGHT_MM/2, DETECTOR_PIXELS_V)
    
    detector_positions = np.zeros((DETECTOR_PIXELS_V, DETECTOR_PIXELS_U, 3))
    
    for i, v in enumerate(v_coords):
        for j, u in enumerate(u_coords):
            detector_positions[i, j] = detector_center + u * detector_u + v * detector_v
    
    return detector_positions, detector_center

def ray_box_intersection(ray_origin, ray_direction, box_min, box_max):
    """Calculate ray-box intersection parameters"""
    # Avoid division by zero
    inv_dir = np.where(np.abs(ray_direction) > 1e-8, 
                      1.0 / ray_direction, 
                      np.sign(ray_direction) * 1e8)
    
    # Calculate intersection parameters
    t1 = (box_min - ray_origin) * inv_dir
    t2 = (box_max - ray_origin) * inv_dir
    
    t_min = np.maximum.reduce([np.minimum(t1[0], t2[0]),
                               np.minimum(t1[1], t2[1]),
                               np.minimum(t1[2], t2[2]),
                               0.0])
    
    t_max = np.minimum.reduce([np.maximum(t1[0], t2[0]),
                               np.maximum(t1[1], t2[1]),
                               np.maximum(t1[2], t2[2])])
    
    return t_min, t_max

def cast_ray_through_volume(mu_volume, volume_origin, volume_spacing, 
                           ray_origin, ray_direction):
    """Cast a single ray through the volume with controlled step size"""
    # Volume bounds in world coordinates
    volume_size_world = np.array(mu_volume.shape[::-1]) * volume_spacing
    box_min = volume_origin
    box_max = volume_origin + volume_size_world
    
    # Find ray-volume intersection
    t_min, t_max = ray_box_intersection(ray_origin, ray_direction, box_min, box_max)
    
    if t_min >= t_max:
        return 0.0
    
    # Number of steps based on ray step size
    ray_length = t_max - t_min
    num_steps = max(int(ray_length / RAY_STEP_SIZE_MM), 2)
    actual_step_size = ray_length / num_steps
    
    # Sample points along ray
    t_values = np.linspace(t_min, t_max, num_steps)
    sample_points = ray_origin + t_values[:, np.newaxis] * ray_direction
    
    # Convert to voxel coordinates (accounting for coordinate order)
    voxel_coords = (sample_points - volume_origin) / volume_spacing
    voxel_coords = voxel_coords[:, [2, 1, 0]]  # Convert to ZYX order
    
    # Accumulate attenuation values
    path_integral = 0.0
    
    for coord in voxel_coords:
        z, y, x = coord
        
        # Bounds check
        if (0 <= z < mu_volume.shape[0] - 1 and
            0 <= y < mu_volume.shape[1] - 1 and
            0 <= x < mu_volume.shape[2] - 1):
            
            # Simple trilinear interpolation
            z0, y0, x0 = int(z), int(y), int(x)
            dz, dy, dx = z - z0, y - y0, x - x0
            
            # Get eight corner values
            v000 = mu_volume[z0, y0, x0]
            v001 = mu_volume[z0, y0, x0+1]
            v010 = mu_volume[z0, y0+1, x0]
            v011 = mu_volume[z0, y0+1, x0+1]
            v100 = mu_volume[z0+1, y0, x0]
            v101 = mu_volume[z0+1, y0, x0+1]
            v110 = mu_volume[z0+1, y0+1, x0]
            v111 = mu_volume[z0+1, y0+1, x0+1]
            
            # Trilinear interpolation
            v00 = v000 * (1-dx) + v001 * dx
            v01 = v010 * (1-dx) + v011 * dx
            v10 = v100 * (1-dx) + v101 * dx
            v11 = v110 * (1-dx) + v111 * dx
            
            v0 = v00 * (1-dy) + v01 * dy
            v1 = v10 * (1-dy) + v11 * dy
            
            mu_value = v0 * (1-dz) + v1 * dz
            
            path_integral += mu_value * actual_step_size
    
    return path_integral

def generate_true_stereo_projection(ct_volume, source_position, view_name):
    """Generate a single projection from a source position"""
    log_message(f"\nGenerating {view_name} projection...")
    log_message(f"Source position: {source_position}")
    
    # Get volume data
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    origin = np.array(ct_volume.GetOrigin())
    
    # Convert HU to attenuation
    mu_volume = correct_hu_to_attenuation(volume)
    log_message(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Get volume center for detector positioning
    volume_size_world = np.array(volume.shape[::-1]) * spacing
    volume_center = origin + volume_size_world / 2
    
    # Create detector positions
    detector_positions, detector_center = create_detector_positions(volume_center)
    log_message(f"Detector center: {detector_center}")
    
    # Initialize projection
    projection = np.zeros((DETECTOR_PIXELS_V, DETECTOR_PIXELS_U))
    
    # Cast rays from source through each detector pixel
    total_rays = DETECTOR_PIXELS_V * DETECTOR_PIXELS_U
    ray_count = 0
    
    log_message(f"Casting {total_rays} rays...")
    
    for i in range(0, DETECTOR_PIXELS_V, 2):  # Skip every other row for speed
        if i % 50 == 0:
            log_message(f"  Progress: {ray_count/total_rays*100:.1f}%")
        
        for j in range(0, DETECTOR_PIXELS_U, 2):  # Skip every other column
            # Ray from source to detector pixel
            detector_pos = detector_positions[i, j]
            ray_direction = detector_pos - source_position
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # Cast ray through volume
            path_integral = cast_ray_through_volume(
                mu_volume, origin, spacing, source_position, ray_direction
            )
            
            # Apply Beer-Lambert law
            transmission = np.exp(-path_integral)
            
            # Fill 2x2 block for speed
            projection[i:i+2, j:j+2] = transmission
            
            ray_count += 4
    
    log_message(f"Ray casting complete. Projection range: [{projection.min():.3f}, {projection.max():.3f}]")
    
    # Convert transmission to radiograph
    # Logarithmic response
    epsilon = 1e-10
    intensity = -np.log10(projection + epsilon)
    
    # Normalize
    body_mask = projection < 0.9  # Where there's attenuation
    if np.any(body_mask):
        p5 = np.percentile(intensity[body_mask], 5)
        p95 = np.percentile(intensity[body_mask], 95)
        intensity = (intensity - p5) / (p95 - p5)
        intensity = np.clip(intensity, 0, 1)
    
    # Gamma correction
    intensity = np.power(intensity, 0.8)
    
    # Ensure air is black
    intensity[projection > 0.95] = 0
    
    return intensity

def generate_v13_true_stereo(ct_volume):
    """Generate true stereo DRR with dual sources"""
    log_message(f"\n--- V13 True Stereo Geometry Generation ---")
    log_message("Implementing research-based true dual-source stereo")
    
    # Get volume info
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())
    origin = np.array(ct_volume.GetOrigin())
    
    # Calculate volume center
    volume_size_world = np.array(volume.shape[::-1]) * spacing
    volume_center = origin + volume_size_world / 2
    
    # Calculate stereo baseline
    half_angle_rad = np.radians(STEREO_ANGLE_DEGREES / 2)
    baseline_mm = 2 * SOURCE_TO_DETECTOR_MM * np.sin(half_angle_rad)
    
    log_message(f"Volume center: {volume_center}")
    log_message(f"Stereo angle: {STEREO_ANGLE_DEGREES}°")
    log_message(f"Baseline: {baseline_mm:.1f}mm")
    log_message(f"Ray step size: {RAY_STEP_SIZE_MM}mm")
    
    # Calculate source positions
    source_left, source_center, source_right = calculate_stereo_source_positions(
        volume_center, baseline_mm
    )
    
    # Generate projections from each source
    drr_left = generate_true_stereo_projection(ct_volume, source_left, "LEFT")
    drr_center = generate_true_stereo_projection(ct_volume, source_center, "CENTER")
    drr_right = generate_true_stereo_projection(ct_volume, source_right, "RIGHT")
    
    # Calculate metrics
    diff_left_right = np.mean(np.abs(drr_left - drr_right))
    log_message(f"\nStereo difference (L-R): {diff_left_right:.4f}")
    
    return drr_left, drr_center, drr_right, baseline_mm, diff_left_right

def simple_depth_from_stereo(drr_left, drr_right, baseline_mm):
    """Simple block matching for depth estimation"""
    h, w = drr_left.shape
    disparity_map = np.zeros((h, w))
    
    block_size = 11
    half_block = block_size // 2
    max_disparity = 100
    
    valid_count = 0
    
    for y in range(half_block, h - half_block, 4):
        for x in range(half_block + max_disparity, w - half_block, 4):
            left_block = drr_left[y-half_block:y+half_block+1,
                                 x-half_block:x+half_block+1]
            
            min_ssd = float('inf')
            best_d = 0
            
            for d in range(0, min(max_disparity, x-half_block), 2):
                right_block = drr_right[y-half_block:y+half_block+1,
                                       x-d-half_block:x-d+half_block+1]
                
                ssd = np.sum((left_block - right_block)**2)
                
                if ssd < min_ssd:
                    min_ssd = ssd
                    best_d = d
            
            if min_ssd < 0.5:
                disparity_map[y:y+4, x:x+4] = best_d
                valid_count += 1
    
    disparity_map = ndimage.gaussian_filter(disparity_map, sigma=2)
    coverage = (disparity_map > 0).sum() / (h * w) * 100
    
    return disparity_map, coverage

def save_v13_outputs(images, metrics, patient_id):
    """Save V13 outputs"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    drr_left, drr_center, drr_right = images
    baseline_mm, diff = metrics
    
    # Estimate depth
    log_message("\nEstimating depth from stereo...")
    disparity_map, coverage = simple_depth_from_stereo(drr_left, drr_right, baseline_mm)
    log_message(f"Depth coverage: {coverage:.1f}%")
    
    # Save images
    for img, view in zip(images, ['left', 'center', 'right']):
        filename = f"{OUTPUT_DIR}/drr_{patient_id}_AP_{view}.png"
        plt.figure(figsize=(8, 10), facecolor='black')
        plt.imshow(img, cmap='gray', aspect='equal')
        plt.title(f'{patient_id} - {view.capitalize()} (True Stereo)', color='white')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    # Comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 20), facecolor='black')
    
    axes[0,0].imshow(drr_left, cmap='gray')
    axes[0,0].set_title('Left Source', color='white', fontsize=14)
    axes[0,0].axis('off')
    
    axes[0,1].imshow(drr_right, cmap='gray')
    axes[0,1].set_title('Right Source', color='white', fontsize=14)
    axes[0,1].axis('off')
    
    axes[1,0].imshow(drr_center, cmap='gray')
    axes[1,0].set_title('Center Reference', color='white', fontsize=14)
    axes[1,0].axis('off')
    
    im = axes[1,1].imshow(disparity_map, cmap='turbo')
    axes[1,1].set_title('Disparity Map', color='white', fontsize=14)
    axes[1,1].axis('off')
    plt.colorbar(im, ax=axes[1,1], fraction=0.046)
    
    plt.suptitle(f'{patient_id} - V13 True Stereo Geometry\n'
                f'Dual X-ray Sources, {baseline_mm:.1f}mm Baseline',
                color='white', fontsize=18)
    
    # Add text box with parameters
    textstr = (f'True Stereo Parameters:\n'
               f'• Dual sources (not volume rotation)\n'
               f'• {STEREO_ANGLE_DEGREES}° convergence angle\n'
               f'• {RAY_STEP_SIZE_MM}mm ray steps\n'
               f'• Proper ray casting\n'
               f'• Stereo diff: {diff:.4f}\n'
               f'• Coverage: {coverage:.1f}%')
    
    fig.text(0.02, 0.02, textstr, fontsize=11, color='lightgreen',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_{patient_id}_AP.png", 
               dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved outputs to {OUTPUT_DIR}")

def main():
    """Main execution"""
    log_message("="*80)
    log_message("V13 True Stereo Geometry DRR Generator")
    log_message("="*80)
    log_message("Research-based implementation:")
    log_message("  • True dual X-ray sources (not volume rotation)")
    log_message("  • Convergent ray geometry")
    log_message(f"  • {RAY_STEP_SIZE_MM}mm ray step size")
    log_message("  • Correct HU to attenuation conversion")
    log_message("="*80)
    
    dataset = {
        'path': 'data/tciaDownload/1.3.6.1.4.1.14519.5.2.1.6834.5010.189721824525842725510380467695',
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
        
        # Generate true stereo
        drr_left, drr_center, drr_right, baseline, diff = generate_v13_true_stereo(ct_volume)
        
        # Save outputs
        images = (drr_left, drr_center, drr_right)
        metrics = (baseline, diff)
        save_v13_outputs(images, metrics, dataset['patient_id'])
        
        total_time = time.time() - start_time
        
        log_message(f"\n{'='*80}")
        log_message(f"V13 COMPLETE - True Stereo Geometry")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"This is true dual-source stereo, not volume rotation")
        log_message(f"{'='*80}")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()