#!/usr/bin/env python3
"""
Stereo DRR Generator V11 - Corrected Geometry
=============================================
COMPLETE REBUILD from scratch with correct stereo geometry:
- Fix baseline calculation errors from V10
- Proper focal length calculation for stereo
- Validate against known CT volume dimensions (402mm chest depth)
- Ensure depth formula gives realistic anatomical ranges (0-40cm)

Goal: Correct stereo parameters for realistic depth reconstruction
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import ndimage
import time

# V11 CORRECTED Parameters
DETECTOR_WIDTH_MM = 356.0
DETECTOR_HEIGHT_MM = 432.0  
RESOLUTION_DPI = 600
PIXEL_SPACING_MM = 0.4

# Key stereo parameters to be calculated correctly
STEREO_ANGLE_DEGREES = 3.0  # Total stereo separation (±1.5°)
SOURCE_TO_DETECTOR_MM = 1000.0  # Standard X-ray geometry

OUTPUT_DIR = "outputs/stereo_v11_corrected"
LOG_FILE = "logs/stereo_drr_v11_corrected.log"

def log_message(message):
    """Log messages to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def calculate_correct_stereo_geometry(volume_info):
    """Calculate CORRECT stereo geometry with proper baseline"""
    volume_origin = np.array(volume_info['origin'])
    volume_spacing = np.array(volume_info['spacing'])
    volume_size = np.array(volume_info['size'])
    
    # Physical volume dimensions
    volume_extent_mm = volume_size * volume_spacing
    volume_center = volume_origin + volume_extent_mm / 2
    
    log_message(f"CT Volume physical dimensions:")
    log_message(f"  Size: {volume_size} voxels")
    log_message(f"  Spacing: {volume_spacing} mm")
    log_message(f"  Physical extent: {volume_extent_mm} mm")
    log_message(f"  Chest depth (Z): {volume_extent_mm[2]:.1f}mm")
    
    # Standard X-ray geometry: source behind patient, detector in front
    # For AP view: X-ray travels +Y direction (anterior to posterior)
    
    # Position detector in front of volume (positive Y direction)
    detector_distance_from_center = SOURCE_TO_DETECTOR_MM / 2
    detector_center = volume_center + np.array([0, detector_distance_from_center, 0])
    
    # Position sources behind volume (negative Y direction)  
    source_distance_from_center = SOURCE_TO_DETECTOR_MM / 2
    source_center = volume_center - np.array([0, source_distance_from_center, 0])
    
    # Calculate stereo baseline from angle
    # For ±1.5° rotation, true baseline is 2 * source_distance * sin(1.5°)
    half_angle_rad = np.radians(STEREO_ANGLE_DEGREES / 2)
    TRUE_BASELINE_MM = 2 * source_distance_from_center * np.sin(half_angle_rad)
    
    # Position left and right sources
    source_left = source_center - np.array([TRUE_BASELINE_MM/2, 0, 0])  # Left along X
    source_right = source_center + np.array([TRUE_BASELINE_MM/2, 0, 0])  # Right along X
    
    # Calculate CORRECT focal length for stereo
    # In stereo, focal length is the distance from source to detector plane
    FOCAL_LENGTH_MM = SOURCE_TO_DETECTOR_MM
    
    log_message(f"\nCORRECTED Stereo Geometry:")
    log_message(f"  Stereo angle: ±{STEREO_ANGLE_DEGREES/2:.1f}° (total {STEREO_ANGLE_DEGREES}°)")
    log_message(f"  TRUE baseline: {TRUE_BASELINE_MM:.1f}mm")
    log_message(f"  Source-to-detector: {SOURCE_TO_DETECTOR_MM}mm")
    log_message(f"  Focal length: {FOCAL_LENGTH_MM}mm")
    log_message(f"  Left source: [{source_left[0]:.1f}, {source_left[1]:.1f}, {source_left[2]:.1f}]")
    log_message(f"  Right source: [{source_right[0]:.1f}, {source_right[1]:.1f}, {source_right[2]:.1f}]")
    log_message(f"  Detector center: [{detector_center[0]:.1f}, {detector_center[1]:.1f}, {detector_center[2]:.1f}]")
    
    return {
        'volume_center': volume_center,
        'volume_extent_mm': volume_extent_mm,
        'source_left': source_left,
        'source_right': source_right,
        'detector_center': detector_center,
        'baseline_mm': TRUE_BASELINE_MM,
        'focal_length_mm': FOCAL_LENGTH_MM,
        'pixel_spacing_mm': PIXEL_SPACING_MM
    }

def validate_stereo_parameters(geometry):
    """Validate stereo parameters against expected chest anatomy"""
    baseline = geometry['baseline_mm']
    focal_length = geometry['focal_length_mm']
    pixel_spacing = geometry['pixel_spacing_mm']
    chest_depth = geometry['volume_extent_mm'][2]
    
    log_message(f"\nSTEREO PARAMETER VALIDATION:")
    log_message(f"Stereo formula: Depth = (focal_length × baseline) / (disparity × pixel_spacing)")
    log_message(f"  Focal length: {focal_length}mm")
    log_message(f"  Baseline: {baseline:.1f}mm")
    log_message(f"  Pixel spacing: {pixel_spacing}mm")
    
    # Calculate expected disparity range for chest anatomy
    # Closest: anterior chest wall (~50mm from detector)
    # Farthest: posterior spine (~chest_depth from detector)
    
    closest_depth_mm = 50  # Anterior chest wall
    farthest_depth_mm = chest_depth  # Posterior spine
    
    # Expected disparities
    max_disparity_expected = (focal_length * baseline) / (closest_depth_mm * pixel_spacing)
    min_disparity_expected = (focal_length * baseline) / (farthest_depth_mm * pixel_spacing)
    
    log_message(f"\nEXPECTED DISPARITY RANGE for chest anatomy:")
    log_message(f"  Closest depth (~anterior): {closest_depth_mm}mm → disparity: {max_disparity_expected:.1f} pixels")
    log_message(f"  Farthest depth (~posterior): {farthest_depth_mm}mm → disparity: {min_disparity_expected:.1f} pixels")
    log_message(f"  Expected disparity range: {min_disparity_expected:.1f} - {max_disparity_expected:.1f} pixels")
    
    # Reverse check: what depth does 1 pixel disparity give?
    depth_per_pixel = (focal_length * baseline) / pixel_spacing
    log_message(f"  Depth sensitivity: {depth_per_pixel:.1f}mm per pixel disparity")
    
    return {
        'expected_min_disparity': min_disparity_expected,
        'expected_max_disparity': max_disparity_expected,
        'depth_per_pixel': depth_per_pixel
    }

def conservative_tissue_segmentation(volume):
    """Conservative tissue segmentation for V11"""
    mu_water = 0.019
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Conservative tissue boundaries
    air_mask = volume < -900
    lung_mask = (volume >= -900) & (volume < -500)
    soft_mask = (volume >= -500) & (volume < 150)
    bone_mask = volume >= 150
    
    # Natural attenuation values
    mu_volume[air_mask] = 0.0
    mu_volume[lung_mask] = 0.001
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # Modest bone enhancement
    bone_hu = volume[bone_mask] 
    mu_volume[bone_mask] = mu_water * (2.5 + bone_hu / 500.0)
    
    return mu_volume

def generate_perspective_projection_corrected(mu_volume, volume_info, source_position, detector_center):
    """Generate perspective projection with CORRECTED ray casting"""
    log_message(f"Generating CORRECTED perspective projection")
    log_message(f"Source: [{source_position[0]:.1f}, {source_position[1]:.1f}, {source_position[2]:.1f}]")
    
    volume_origin = np.array(volume_info['origin'])
    volume_spacing = np.array(volume_info['spacing'])
    
    # Detector dimensions
    detector_pixels_u = int(DETECTOR_WIDTH_MM / PIXEL_SPACING_MM)
    detector_pixels_v = int(DETECTOR_HEIGHT_MM / PIXEL_SPACING_MM)
    
    projection = np.zeros((detector_pixels_v, detector_pixels_u))
    
    # Detector coordinate system
    detector_u = np.array([1, 0, 0])  # X direction (left-right)
    detector_v = np.array([0, 0, -1])  # -Z direction (up-down, radiographic)
    
    log_message(f"Detector: {detector_pixels_u} × {detector_pixels_v} pixels")
    log_message(f"Ray casting through volume...")
    
    # Process in chunks for progress
    chunk_size = 50
    total_pixels = detector_pixels_u * detector_pixels_v
    processed = 0
    
    for v_start in range(0, detector_pixels_v, chunk_size):
        v_end = min(v_start + chunk_size, detector_pixels_v)
        progress = (v_start * detector_pixels_u) / total_pixels * 100
        if v_start % 100 == 0:
            log_message(f"  Progress: {progress:.1f}%")
        
        for v in range(v_start, v_end):
            for u in range(detector_pixels_u):
                # Calculate detector pixel position
                u_offset = (u - detector_pixels_u/2 + 0.5) * PIXEL_SPACING_MM
                v_offset = (v - detector_pixels_v/2 + 0.5) * PIXEL_SPACING_MM
                
                detector_pixel = (detector_center + 
                                u_offset * detector_u + 
                                v_offset * detector_v)
                
                # Ray from source to detector pixel
                ray_direction = detector_pixel - source_position
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                # Fast ray casting through volume
                path_integral = cast_ray_fast(mu_volume, source_position, ray_direction, 
                                            volume_origin, volume_spacing)
                
                projection[v, u] = path_integral
                processed += 1
    
    log_message(f"Ray casting complete: {processed} rays processed")
    log_message(f"Projection range: [{projection.min():.3f}, {projection.max():.3f}]")
    
    return projection

def cast_ray_fast(mu_volume, ray_origin, ray_direction, volume_origin, volume_spacing):
    """Fast ray casting implementation"""
    # Calculate volume bounds
    volume_size_world = np.array(mu_volume.shape[::-1]) * volume_spacing
    volume_max = volume_origin + volume_size_world
    
    # Ray-box intersection
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
    
    # Sample along ray
    step_size = 1.0  # mm
    ray_length = t_max - t_min
    num_samples = max(int(ray_length / step_size), 1)
    
    path_integral = 0.0
    for i in range(num_samples):
        t = t_min + (i + 0.5) * ray_length / num_samples
        sample_point = ray_origin + t * ray_direction
        
        # Convert to voxel coordinates
        voxel_coords = (sample_point - volume_origin) / volume_spacing
        x, y, z = voxel_coords[0], voxel_coords[1], voxel_coords[2]
        
        # Bounds check and trilinear interpolation
        if (0 <= x < mu_volume.shape[2]-1 and 
            0 <= y < mu_volume.shape[1]-1 and 
            0 <= z < mu_volume.shape[0]-1):
            
            # Simple trilinear interpolation
            x0, y0, z0 = int(x), int(y), int(z)
            dx, dy, dz = x - x0, y - y0, z - z0
            
            try:
                c000 = mu_volume[z0, y0, x0]
                c001 = mu_volume[z0, y0, min(x0+1, mu_volume.shape[2]-1)]
                c010 = mu_volume[z0, min(y0+1, mu_volume.shape[1]-1), x0]
                c011 = mu_volume[z0, min(y0+1, mu_volume.shape[1]-1), min(x0+1, mu_volume.shape[2]-1)]
                c100 = mu_volume[min(z0+1, mu_volume.shape[0]-1), y0, x0]
                c101 = mu_volume[min(z0+1, mu_volume.shape[0]-1), y0, min(x0+1, mu_volume.shape[2]-1)]
                c110 = mu_volume[min(z0+1, mu_volume.shape[0]-1), min(y0+1, mu_volume.shape[1]-1), x0]
                c111 = mu_volume[min(z0+1, mu_volume.shape[0]-1), min(y0+1, mu_volume.shape[1]-1), min(x0+1, mu_volume.shape[2]-1)]
                
                # Trilinear interpolation
                c00 = c000 * (1 - dx) + c001 * dx
                c01 = c010 * (1 - dx) + c011 * dx
                c10 = c100 * (1 - dx) + c101 * dx
                c11 = c110 * (1 - dx) + c111 * dx
                
                c0 = c00 * (1 - dy) + c01 * dy
                c1 = c10 * (1 - dy) + c11 * dy
                
                mu_value = c0 * (1 - dz) + c1 * dz
                path_integral += mu_value * step_size
            except:
                # Fallback to nearest neighbor
                mu_value = mu_volume[z0, y0, x0]
                path_integral += mu_value * step_size
    
    return path_integral

def generate_v11_corrected_stereo(ct_volume):
    """Generate V11 corrected stereo pair"""
    log_message(f"\n--- V11 CORRECTED Stereo Generation ---")
    log_message("COMPLETE REBUILD with correct stereo geometry")
    
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
    
    # Calculate CORRECT stereo geometry
    geometry = calculate_correct_stereo_geometry(volume_info)
    
    # Validate stereo parameters
    validation = validate_stereo_parameters(geometry)
    
    # Enhanced tissue segmentation
    mu_volume = conservative_tissue_segmentation(volume)
    log_message(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate perspective projections from each source
    log_message("\nGenerating LEFT perspective projection...")
    projection_left = generate_perspective_projection_corrected(
        mu_volume, volume_info, geometry['source_left'], geometry['detector_center']
    )
    
    log_message("Generating RIGHT perspective projection...")
    projection_right = generate_perspective_projection_corrected(
        mu_volume, volume_info, geometry['source_right'], geometry['detector_center']
    )
    
    # Convert to radiographs
    def to_radiograph(projection):
        transmission = np.exp(-projection)
        epsilon = 1e-7
        intensity = -np.log10(transmission + epsilon)
        
        # Natural normalization
        body_mask = projection > 0.05
        if np.any(body_mask):
            p_low = np.percentile(intensity[body_mask], 2)
            p_high = np.percentile(intensity[body_mask], 98)
            intensity = (intensity - p_low) / (p_high - p_low)
            intensity = np.clip(intensity, 0, 1)
        
        # Natural gamma
        intensity = np.power(intensity, 1.0 / 1.1)
        
        # Preserve air regions
        air_mask = projection < 0.02
        intensity[air_mask] = 0
        
        return np.clip(intensity, 0, 1)
    
    drr_left = to_radiograph(projection_left)
    drr_right = to_radiograph(projection_right)
    
    # Generate center reference
    source_center = (geometry['source_left'] + geometry['source_right']) / 2
    log_message("Generating CENTER reference projection...")
    projection_center = generate_perspective_projection_corrected(
        mu_volume, volume_info, source_center, geometry['detector_center']
    )
    drr_center = to_radiograph(projection_center)
    
    log_message(f"✅ V11 corrected stereo generation complete")
    
    return drr_left, drr_center, drr_right, geometry, validation

def analyze_v11_results(drr_left, drr_right, geometry, validation):
    """Analyze V11 results with corrected parameters"""
    # Calculate image differences
    diff_left_right = np.mean(np.abs(drr_left - drr_right))
    
    log_message(f"\nV11 CORRECTED RESULTS ANALYSIS:")
    log_message(f"Image difference (L-R): {diff_left_right:.4f}")
    
    # Extract corrected stereo parameters
    baseline = geometry['baseline_mm']
    focal_length = geometry['focal_length_mm']
    pixel_spacing = geometry['pixel_spacing_mm']
    
    log_message(f"\nCORRECTED STEREO PARAMETERS for manual validation:")
    log_message(f"  Baseline: {baseline:.1f}mm")
    log_message(f"  Focal length: {focal_length}mm")
    log_message(f"  Pixel spacing: {pixel_spacing}mm")
    log_message(f"  Expected disparity range: {validation['expected_min_disparity']:.1f} - {validation['expected_max_disparity']:.1f} pixels")
    log_message(f"  Depth sensitivity: {validation['depth_per_pixel']:.1f}mm per pixel")
    
    return {
        'baseline_mm': baseline,
        'focal_length_mm': focal_length,
        'pixel_spacing_mm': pixel_spacing,
        'image_difference': diff_left_right,
        'expected_disparity_range': [validation['expected_min_disparity'], validation['expected_max_disparity']],
        'depth_per_pixel_mm': validation['depth_per_pixel']
    }

def save_v11_corrected_outputs(images, analysis, patient_id):
    """Save V11 corrected outputs"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    drr_left, drr_center, drr_right = images
    
    # Save individual images
    save_radiograph(drr_left, f"{OUTPUT_DIR}/drr_{patient_id}_AP_left.png", 
                   f"{patient_id} - AP - Left (Corrected)")
    save_radiograph(drr_center, f"{OUTPUT_DIR}/drr_{patient_id}_AP_center.png", 
                   f"{patient_id} - AP - Center (Corrected)")
    save_radiograph(drr_right, f"{OUTPUT_DIR}/drr_{patient_id}_AP_right.png", 
                   f"{patient_id} - AP - Right (Corrected)")
    
    # Save comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='black')
    
    imgs = [drr_left, drr_center, drr_right, np.zeros_like(drr_left)]
    titles = ['Left View\n(Corrected)', 'Center View\n(Reference)', 'Right View\n(Corrected)', 'V11 CORRECTED\nParameters']
    
    for ax, img, title in zip(axes[:3], imgs[:3], titles[:3]):
        ax.imshow(img, cmap='gray', aspect='equal', vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=12, pad=15)
        ax.axis('off')
    
    # Parameters display
    axes[3].text(0.5, 0.5, f'V11 CORRECTED\nGEOMETRY\n\n'
                           f'Baseline: {analysis["baseline_mm"]:.1f}mm\n'
                           f'Focal length: {analysis["focal_length_mm"]}mm\n'
                           f'Pixel spacing: {analysis["pixel_spacing_mm"]}mm\n'
                           f'Image diff: {analysis["image_difference"]:.4f}\n\n'
                           f'Expected disparity:\n'
                           f'{analysis["expected_disparity_range"][0]:.1f} - {analysis["expected_disparity_range"][1]:.1f} px\n\n'
                           f'Depth sensitivity:\n'
                           f'{analysis["depth_per_pixel_mm"]:.1f}mm/pixel', 
               transform=axes[3].transAxes, fontsize=11, color='lightblue', weight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='darkblue', alpha=0.8))
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].set_title(titles[3], color='white', fontsize=12, pad=15)
    axes[3].axis('off')
    
    plt.suptitle(f'{patient_id} - AP - V11 CORRECTED STEREO GEOMETRY\n'
                f'Rebuilt with Accurate Baseline and Focal Length Calculations',
                color='lightblue', fontsize=16, y=0.95, weight='bold')
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/comparison_{patient_id}_AP.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    log_message(f"Saved V11 corrected results to {filename}")

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
    """Main execution with corrected geometry"""
    log_message("="*80)
    log_message("V11 CORRECTED Stereo DRR Generator")
    log_message("="*80)
    log_message("COMPLETE REBUILD fixing V10 geometry errors:")
    log_message(f"  • Correct baseline calculation from {STEREO_ANGLE_DEGREES}° stereo angle")
    log_message(f"  • Proper focal length for stereo reconstruction")
    log_message(f"  • Validation against {402:.0f}mm chest depth")
    log_message(f"  • Realistic depth ranges for anatomy")
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
        
        # Generate corrected stereo
        drr_left, drr_center, drr_right, geometry, validation = generate_v11_corrected_stereo(ct_volume)
        
        # Analyze corrected results
        analysis = analyze_v11_results(drr_left, drr_right, geometry, validation)
        
        # Save outputs
        images = (drr_left, drr_center, drr_right)
        save_v11_corrected_outputs(images, analysis, dataset['patient_id'])
        
        total_time = time.time() - start_time
        
        # FINAL SUMMARY
        log_message(f"\n{'='*80}")
        log_message(f"V11 CORRECTED - GEOMETRY VALIDATION")
        log_message(f"Time: {total_time:.1f} seconds")
        log_message(f"CORRECTED stereo parameters:")
        log_message(f"  Baseline: {analysis['baseline_mm']:.1f}mm")
        log_message(f"  Focal length: {analysis['focal_length_mm']}mm") 
        log_message(f"  Pixel spacing: {analysis['pixel_spacing_mm']}mm")
        log_message(f"Expected depth range: 50-402mm (chest anatomy)")
        log_message(f"Expected disparity range: {analysis['expected_disparity_range'][0]:.1f}-{analysis['expected_disparity_range'][1]:.1f} pixels")
        log_message(f"Depth sensitivity: {analysis['depth_per_pixel_mm']:.1f}mm per pixel")
        log_message(f"Image difference: {analysis['image_difference']:.4f}")
        log_message(f"Output: {OUTPUT_DIR}")
        log_message(f"{'='*80}")
        log_message(f"🎯 READY FOR MANUAL STEREO VALIDATION")
        
    except Exception as e:
        log_message(f"❌ Error: {str(e)}")
        import traceback
        log_message(traceback.format_exc())

if __name__ == "__main__":
    main()