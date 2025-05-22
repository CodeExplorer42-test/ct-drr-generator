#!/usr/bin/env python3
"""
Fixed DRR generation with proper coordinate handling and diagnostics.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import time


def generate_drr_parallel(ct_volume, projection_type='AP'):
    """
    Generate DRR using parallel projection (simpler, more robust).
    """
    # Get volume as numpy array (z, y, x order)
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    # Convert HU to attenuation coefficients
    # Using simple linear model: mu = mu_water * (1 + HU/1000)
    mu_water = 0.02  # mm^-1 at ~70 keV
    mu_volume = mu_water * (1.0 + volume / 1000.0)
    mu_volume = np.maximum(mu_volume, 0.0)  # No negative attenuation
    
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")
    print(f"HU range: [{volume.min():.1f}, {volume.max():.1f}]")
    print(f"Attenuation range: [{mu_volume.min():.4f}, {mu_volume.max():.4f}] mm^-1")
    
    if projection_type == 'AP':
        # Project along x-axis (sum over axis 2)
        # Intensity integral along each ray
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        print(f"AP projection shape: {projection.shape}")
        print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    else:  # Lateral
        # Project along y-axis (sum over axis 1)
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        print(f"Lateral projection shape: {projection.shape}")
        print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    
    # Apply Beer-Lambert law: I = I0 * exp(-integral)
    # Use logarithmic transformation for better visualization
    # This mimics the film response
    max_integral = projection.max()
    if max_integral > 0:
        # Normalize to prevent overflow
        projection_normalized = projection / max_integral * 10.0  # Scale factor
        transmission = np.exp(-projection_normalized)
        
        # Convert to optical density (what we see on X-ray)
        # Higher density = darker on film = whiter in our display
        drr = 1.0 - transmission
        
        # Apply window/level adjustment for chest X-ray appearance
        # Enhance contrast in soft tissue range
        drr = np.clip(drr, 0.3, 0.9)  # Clip to useful range
        drr = (drr - 0.3) / 0.6  # Normalize to [0, 1]
        
        # Apply gamma correction for better visualization
        drr = np.power(drr, 0.7)
    else:
        drr = np.zeros_like(projection)
    
    print(f"DRR range: [{drr.min():.3f}, {drr.max():.3f}]")
    
    return drr


def main():
    """Generate DRRs with simplified approach."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/fixed")
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
    print("\n=== Generating AP View ===")
    start_time = time.time()
    drr_ap = generate_drr_parallel(ct_volume, 'AP')
    print(f"AP generation time: {time.time() - start_time:.1f} seconds")
    
    # Generate lateral view
    print("\n=== Generating Lateral View ===")
    start_time = time.time()
    drr_lateral = generate_drr_parallel(ct_volume, 'Lateral')
    print(f"Lateral generation time: {time.time() - start_time:.1f} seconds")
    
    # Save individual views
    plt.figure(figsize=(10, 10))
    plt.imshow(drr_ap, cmap='gray', vmin=0, vmax=1)
    plt.title('DRR - Anterior-Posterior View (Fixed)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_fixed_ap.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(drr_lateral, cmap='gray', vmin=0, vmax=1)
    plt.title('DRR - Lateral View (Fixed)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_fixed_lateral.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
    
    axes[0].imshow(drr_ap, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Anterior-Posterior View', fontsize=16, color='white')
    axes[0].axis('off')
    
    axes[1].imshow(drr_lateral, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Lateral View', fontsize=16, color='white')
    axes[1].axis('off')
    
    plt.suptitle('Fixed DRR Generation (Parallel Projection)', fontsize=20, color='white')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_fixed_combined.png', dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"\nDRRs saved to {output_dir}")
    
    # Also create a diagnostic figure showing the process
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Show middle slice of CT
    middle_slice = ct_volume.GetSize()[2] // 2
    ct_array = sitk.GetArrayFromImage(ct_volume)
    
    axes[0, 0].imshow(ct_array[middle_slice, :, :], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 0].set_title('CT Axial Slice (Middle)')
    axes[0, 0].axis('off')
    
    # Show coronal slice
    axes[0, 1].imshow(ct_array[:, ct_array.shape[1]//2, :], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 1].set_title('CT Coronal Slice')
    axes[0, 1].axis('off')
    
    # Show sagittal slice
    axes[0, 2].imshow(ct_array[:, :, ct_array.shape[2]//2], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 2].set_title('CT Sagittal Slice')
    axes[0, 2].axis('off')
    
    # Show projection integrals before Beer-Lambert
    mu_water = 0.02  # mm^-1
    mu_volume = mu_water * (1.0 + ct_array / 1000.0)
    mu_volume = np.maximum(mu_volume, 0.0)
    
    projection_ap = np.sum(mu_volume, axis=2)
    axes[1, 0].imshow(projection_ap, cmap='hot')
    axes[1, 0].set_title('AP Path Integral (before Beer-Lambert)')
    axes[1, 0].axis('off')
    
    # Show final DRRs
    axes[1, 1].imshow(drr_ap, cmap='gray')
    axes[1, 1].set_title('Final AP DRR')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(drr_lateral, cmap='gray')
    axes[1, 2].set_title('Final Lateral DRR')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Diagnostic images saved!")


if __name__ == "__main__":
    main()