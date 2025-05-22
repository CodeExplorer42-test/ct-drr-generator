#!/usr/bin/env python3
"""
Diagnostic DRR generation to identify and fix visualization issues.
Focus on bone visibility and aspect ratio correction.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def generate_diagnostic_drr(ct_volume, projection_type='AP', debug=True):
    """
    Generate DRR with extensive diagnostic logging.
    """
    # Get volume and spacing
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    print(f"\n=== DIAGNOSTIC INFO ===")
    print(f"Volume shape (Z,Y,X): {volume.shape}")
    print(f"Spacing (X,Y,Z): {spacing} mm")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Analyze HU distribution
    bone_threshold = 300
    bone_voxels = np.sum(volume >= bone_threshold)
    total_voxels = volume.size
    bone_percentage = (bone_voxels / total_voxels) * 100
    
    print(f"\nBone analysis:")
    print(f"  Bone voxels (HU >= {bone_threshold}): {bone_voxels:,}")
    print(f"  Percentage of volume: {bone_percentage:.2f}%")
    print(f"  Max bone HU: {volume[volume >= bone_threshold].max():.0f}")
    
    # DON'T apply windowing - we need full HU range for bones
    print("\nNOTE: Not applying windowing to preserve bone HU values")
    
    # Convert HU to attenuation with MORE AGGRESSIVE bone coefficients
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air - zero attenuation
    air_mask = volume < -900
    mu_volume[air_mask] = 0.0
    
    # Lung tissue
    lung_mask = (volume >= -900) & (volume < -500)
    lung_hu = volume[lung_mask]
    mu_volume[lung_mask] = 0.0001 + (lung_hu + 900) * (0.001 / 400)
    
    # Soft tissue
    soft_mask = (volume >= -500) & (volume < 200)
    soft_hu = volume[soft_mask]
    mu_volume[soft_mask] = mu_water * (1.0 + soft_hu / 1000.0)
    
    # Bone - MUCH MORE AGGRESSIVE
    # Real bone attenuation is 2-3x water at diagnostic energies
    bone_mask = volume >= 200
    bone_hu = volume[bone_mask]
    # More aggressive formula for better contrast
    mu_volume[bone_mask] = mu_water * (2.5 + bone_hu / 500.0)
    
    # Log attenuation statistics
    print(f"\nAttenuation statistics:")
    print(f"  Air: {mu_volume[air_mask].mean():.5f} mm^-1")
    if lung_mask.any():
        print(f"  Lung: {mu_volume[lung_mask].mean():.5f} mm^-1")
    print(f"  Soft tissue: {mu_volume[soft_mask].mean():.5f} mm^-1")
    print(f"  Bone: min={mu_volume[bone_mask].min():.3f}, max={mu_volume[bone_mask].max():.3f}, mean={mu_volume[bone_mask].mean():.3f} mm^-1")
    
    # NO smoothing to preserve fine detail
    print("\nNOTE: Skipping Gaussian smoothing to preserve bone detail")
    
    # Generate projection
    if projection_type == 'AP':
        # AP view: integrate along Y axis
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)
        # Correct aspect ratio calculation
        pixel_aspect = spacing[0] / spacing[2]  # X/Z for display
        print(f"\nAP view pixel aspect ratio: {pixel_aspect:.3f}")
    else:  # Lateral
        # Lateral view: integrate along X axis
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
        pixel_aspect = spacing[1] / spacing[2]  # Y/Z for display
        print(f"\nLateral view pixel aspect ratio: {pixel_aspect:.3f}")
    
    print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    
    # Analyze projection distribution
    proj_percentiles = np.percentile(projection[projection > 0], [10, 25, 50, 75, 90, 95, 99])
    print(f"\nProjection percentiles (non-zero):")
    for i, p in enumerate([10, 25, 50, 75, 90, 95, 99]):
        print(f"  {p}%: {proj_percentiles[i]:.2f}")
    
    # Apply Beer-Lambert law
    transmission = np.exp(-projection)
    
    # Simple intensity mapping to preserve dynamic range
    # Use log transformation but with better scaling
    epsilon = 1e-6
    intensity = -np.log10(transmission + epsilon)
    
    # Find dynamic range in body region only
    body_mask = projection > 0.1
    if np.any(body_mask):
        # Use wider percentile range to preserve bone detail
        p1 = np.percentile(intensity[body_mask], 1)
        p99 = np.percentile(intensity[body_mask], 99)
        print(f"\nIntensity percentiles (1%, 99%): [{p1:.3f}, {p99:.3f}]")
        
        # Linear scaling with clipping
        intensity = (intensity - p1) / (p99 - p1)
        intensity = np.clip(intensity, 0, 1)
    
    # Very mild gamma correction
    gamma = 1.1  # Almost linear
    intensity = np.power(intensity, 1.0 / gamma)
    
    # Ensure air stays black
    air_projection = projection < 0.01
    intensity[air_projection] = 0
    
    print(f"\nFinal intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    
    # Check bone visibility
    high_intensity = intensity > 0.7
    print(f"High intensity pixels (potential bones): {np.sum(high_intensity):,} ({np.sum(high_intensity)/intensity.size*100:.1f}%)")
    
    return intensity, pixel_aspect


def save_diagnostic_xray(image, filename, pixel_aspect=1.0, title=None):
    """Save DRR with correct pixel aspect ratio."""
    # Standard figure width
    fig_width = 10  # inches
    
    # Calculate height based on image shape and pixel aspect
    h, w = image.shape
    image_aspect = (w * pixel_aspect) / h
    fig_height = fig_width / image_aspect
    
    print(f"\nSaving {title}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Pixel aspect: {pixel_aspect:.3f}")
    print(f"  Figure size: {fig_width:.1f} x {fig_height:.1f} inches")
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display with equal aspect to respect pixel aspect ratio
    im = ax.imshow(image, cmap='gray', aspect='equal', 
                   vmin=0, vmax=1, interpolation='bilinear')
    
    # Add title
    if title:
        ax.text(0.5, 0.98, title, transform=ax.transAxes,
                fontsize=14, color='white', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='black', alpha=0.8))
    
    ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    plt.close()


def main():
    """Generate diagnostic DRRs with enhanced bone visualization."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CT series
    series_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not series_dirs:
        print("No CT series found!")
        return
    
    series_dir = series_dirs[0]
    print(f"Processing series: {series_dir.name}")
    
    # Load DICOM series
    print("\nLoading DICOM series...")
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(series_dir))
    
    if not dicom_files:
        print("No DICOM files found!")
        return
    
    reader.SetFileNames(dicom_files)
    ct_volume = reader.Execute()
    
    # Generate AP view with diagnostics
    print("\n=== Generating Diagnostic AP View ===")
    drr_ap, aspect_ap = generate_diagnostic_drr(ct_volume, 'AP', debug=True)
    
    # Save AP view
    save_diagnostic_xray(drr_ap, output_dir / 'diagnostic_chest_xray_ap.png', 
                        pixel_aspect=aspect_ap, title='CHEST PA - DIAGNOSTIC')
    
    # Generate lateral view
    print("\n=== Generating Diagnostic Lateral View ===")
    drr_lateral, aspect_lateral = generate_diagnostic_drr(ct_volume, 'Lateral', debug=True)
    
    # Save lateral view
    save_diagnostic_xray(drr_lateral, output_dir / 'diagnostic_chest_xray_lateral.png',
                        pixel_aspect=aspect_lateral, title='LATERAL - DIAGNOSTIC')
    
    # Create comparison of different bone enhancement levels
    print("\n=== Creating Bone Enhancement Comparison ===")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    
    # Different bone multipliers
    multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    for idx, mult in enumerate(multipliers):
        row = idx // 3
        col = idx % 3
        
        # Create custom attenuation with this multiplier
        mu_volume = np.zeros_like(sitk.GetArrayFromImage(ct_volume), dtype=np.float32)
        volume = sitk.GetArrayFromImage(ct_volume)
        
        # Standard air/lung/soft tissue
        mu_volume[volume < -900] = 0.0
        mu_volume[(volume >= -900) & (volume < -500)] = 0.001
        mu_volume[(volume >= -500) & (volume < 200)] = 0.019 * (1.0 + volume[(volume >= -500) & (volume < 200)] / 1000.0)
        
        # Variable bone
        bone_mask = volume >= 200
        mu_volume[bone_mask] = 0.019 * (mult + volume[bone_mask] / 500.0)
        
        # Generate projection
        projection = np.sum(mu_volume, axis=1) * 0.9765625  # Y spacing
        projection = np.flipud(projection)
        
        # Convert to intensity
        transmission = np.exp(-projection)
        intensity = -np.log10(transmission + 1e-6)
        
        # Normalize
        body_mask = projection > 0.1
        if np.any(body_mask):
            p1 = np.percentile(intensity[body_mask], 1)
            p99 = np.percentile(intensity[body_mask], 99)
            intensity = (intensity - p1) / (p99 - p1)
            intensity = np.clip(intensity, 0, 1)
        
        # Display
        axes[row, col].imshow(intensity, cmap='gray', aspect=aspect_ap)
        axes[row, col].set_title(f'Bone multiplier: {mult}x', fontsize=10)
        axes[row, col].axis('off')
    
    plt.suptitle('Bone Enhancement Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'bone_enhancement_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDiagnostic DRRs saved to {output_dir}")
    print("Files created:")
    print("  - diagnostic_chest_xray_ap.png (Enhanced bone visibility)")
    print("  - diagnostic_chest_xray_lateral.png")
    print("  - bone_enhancement_comparison.png (Different bone multipliers)")
    print("\nKey diagnostics:")
    print("  ✓ Removed windowing to preserve bone HU values")
    print("  ✓ Increased bone attenuation coefficient (2.5x)")
    print("  ✓ Fixed pixel aspect ratio calculation")
    print("  ✓ Removed smoothing to preserve detail")
    print("  ✓ Used wider percentile range for normalization")


if __name__ == "__main__":
    main()