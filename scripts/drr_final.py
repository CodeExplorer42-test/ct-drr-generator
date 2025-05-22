#!/usr/bin/env python3
"""
Final corrected DRR generation with proper projection directions and visualization.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def generate_drr_correct(ct_volume, projection_type='AP'):
    """
    Generate anatomically correct DRR projections.
    AP: Front-to-back projection (along Y axis)
    Lateral: Side-to-side projection (along X axis)
    """
    # Get volume as numpy array (z, y, x order)
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    # Convert HU to attenuation coefficients
    # Bone: HU ~1000 -> high attenuation
    # Soft tissue: HU ~40 -> medium attenuation  
    # Lung: HU ~-500 -> low attenuation
    # Air: HU ~-1000 -> very low attenuation
    
    # More realistic attenuation model
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air and lung
    mu_volume[volume < -500] = 0.0001  # Air/lung - minimal attenuation
    
    # Soft tissue  
    soft_mask = (volume >= -500) & (volume < 200)
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # Bone
    bone_mask = volume >= 200
    mu_volume[bone_mask] = mu_water * (1.5 + volume[bone_mask] / 1000.0)
    
    print(f"Volume shape (Z,Y,X): {volume.shape}")
    print(f"Spacing (X,Y,Z): {spacing} mm")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    print(f"Attenuation range: [{mu_volume.min():.4f}, {mu_volume.max():.4f}] mm^-1")
    
    if projection_type == 'AP':
        # AP view: rays travel along Y axis (anterior to posterior)
        # Sum over Y axis (axis 1)
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        # Result shape: (Z, X) - need to flip Z for proper orientation
        projection = np.flipud(projection)
        print(f"AP projection shape (after flip): {projection.shape}")
    else:  # Lateral
        # Lateral view: rays travel along X axis (right to left)
        # Sum over X axis (axis 2)
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        # Result shape: (Z, Y) - need to flip Z for proper orientation
        projection = np.flipud(projection)
        print(f"Lateral projection shape (after flip): {projection.shape}")
    
    print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}] mm")
    
    # Apply Beer-Lambert law with proper scaling
    # Typical chest thickness: 200-300mm
    # We want good contrast without saturation
    
    # Normalize based on expected chest thickness
    expected_thickness = 250.0  # mm
    scale_factor = expected_thickness * mu_water * 0.5  # Adjust for good contrast
    
    projection_scaled = projection / scale_factor
    
    # Apply exponential attenuation
    transmission = np.exp(-projection_scaled)
    
    # Convert to radiographic density (darker = more absorption)
    # Real X-rays: more dense tissue = less transmission = darker on film
    drr = 1.0 - transmission
    
    # Apply logarithmic response (mimics film characteristics)
    # This enhances contrast in the middle ranges
    epsilon = 0.001
    drr = -np.log10(transmission + epsilon) / 3.0  # Scale to ~[0,1]
    drr = np.clip(drr, 0, 1)
    
    # Apply gamma correction for better soft tissue visibility
    drr = np.power(drr, 0.5)
    
    # Invert for conventional radiograph appearance
    # (dense structures appear white)
    drr = 1.0 - drr
    
    print(f"DRR range: [{drr.min():.3f}, {drr.max():.3f}]")
    
    return drr


def main():
    """Generate anatomically correct DRRs."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/final_correct")
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
    
    # Generate AP view
    print("\n=== Generating AP (Frontal) View ===")
    drr_ap = generate_drr_correct(ct_volume, 'AP')
    
    # Generate lateral view
    print("\n=== Generating Lateral View ===")
    drr_lateral = generate_drr_correct(ct_volume, 'Lateral')
    
    # Save individual views with proper aspect ratio
    # AP view
    fig = plt.figure(figsize=(8, 10), facecolor='black')
    plt.imshow(drr_ap, cmap='gray', aspect='auto')
    plt.title('Chest X-Ray - AP View', fontsize=14, color='white', pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'chest_xray_ap.png', dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    plt.close()
    
    # Lateral view
    fig = plt.figure(figsize=(8, 10), facecolor='black')
    plt.imshow(drr_lateral, cmap='gray', aspect='auto')
    plt.title('Chest X-Ray - Lateral View', fontsize=14, color='white', pad=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'chest_xray_lateral.png', dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    
    # Create comparison with annotations
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), facecolor='black')
    
    # AP view with annotations
    axes[0].imshow(drr_ap, cmap='gray', aspect='auto')
    axes[0].set_title('Anterior-Posterior (AP) View', fontsize=14, color='white', pad=10)
    axes[0].text(0.02, 0.98, 'R', transform=axes[0].transAxes, fontsize=16, 
                 color='white', ha='left', va='top')
    axes[0].text(0.98, 0.98, 'L', transform=axes[0].transAxes, fontsize=16,
                 color='white', ha='right', va='top')
    axes[0].axis('off')
    
    # Lateral view
    axes[1].imshow(drr_lateral, cmap='gray', aspect='auto')
    axes[1].set_title('Lateral View', fontsize=14, color='white', pad=10)
    axes[1].axis('off')
    
    plt.suptitle('Digitally Reconstructed Radiographs (DRR)', fontsize=18, color='white')
    plt.tight_layout()
    plt.savefig(output_dir / 'chest_xray_both_views.png', dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()
    
    # Create diagnostic comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    
    # Show CT slices for reference
    ct_array = sitk.GetArrayFromImage(ct_volume)
    
    # Axial slice (middle)
    axes[0, 0].imshow(ct_array[ct_array.shape[0]//2, :, :], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 0].set_title('CT - Axial Slice (Middle)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Coronal slice (middle) - this is what AP projection integrates
    axes[0, 1].imshow(ct_array[:, ct_array.shape[1]//2, :], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 1].set_title('CT - Coronal Slice', fontsize=12)
    axes[0, 1].axis('off')
    
    # Sagittal slice (middle) - this is what lateral projection integrates
    axes[0, 2].imshow(ct_array[:, :, ct_array.shape[2]//2], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 2].set_title('CT - Sagittal Slice', fontsize=12)
    axes[0, 2].axis('off')
    
    # Show projections
    axes[1, 0].axis('off')  # Empty
    
    axes[1, 1].imshow(drr_ap, cmap='gray', aspect='auto')
    axes[1, 1].set_title('DRR - AP View', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(drr_lateral, cmap='gray', aspect='auto')
    axes[1, 2].set_title('DRR - Lateral View', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle('CT to DRR Transformation', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'ct_to_drr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nChest X-ray DRRs saved to {output_dir}")
    print("Files created:")
    print("  - chest_xray_ap.png")
    print("  - chest_xray_lateral.png") 
    print("  - chest_xray_both_views.png")
    print("  - ct_to_drr_comparison.png")


if __name__ == "__main__":
    main()