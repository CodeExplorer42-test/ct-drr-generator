#!/usr/bin/env python3
"""
Clinical-quality DRR generation optimized for chest X-ray appearance.
Designed to match real chest X-rays used for lung infection detection.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def apply_clinical_window(image, window_center=-600, window_width=1500):
    """Apply DICOM-style windowing for chest visualization."""
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(image, lower, upper)
    windowed = (windowed - lower) / window_width
    return windowed


def generate_clinical_drr(ct_volume, projection_type='AP'):
    """
    Generate clinical-quality DRR that matches real chest X-ray appearance.
    Optimized for detecting lung pathology including infections.
    """
    # Get volume and spacing
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    print(f"Volume shape (Z,Y,X): {volume.shape}")
    print(f"Spacing (X,Y,Z): {spacing} mm")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Apply chest window first to enhance contrast
    volume_windowed = apply_clinical_window(volume, window_center=-400, window_width=1500)
    
    # Convert HU to linear attenuation coefficients
    # Using energy-dependent coefficients (~60-80 keV effective for chest)
    mu_water = 0.0195  # mm^-1 at ~70 keV
    
    # Create attenuation map with tissue-specific values
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air and lung parenchyma - very low attenuation
    lung_mask = volume < -500
    mu_volume[lung_mask] = 0.00005 + (volume[lung_mask] + 1000) * 0.00001
    
    # Fat tissue
    fat_mask = (volume >= -500) & (volume < -100)
    mu_volume[fat_mask] = mu_water * 0.9 * (1.0 + volume[fat_mask] / 1000.0)
    
    # Soft tissue (muscle, organs)
    soft_mask = (volume >= -100) & (volume < 300)
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # Bone and calcifications - high attenuation
    bone_mask = volume >= 300
    # Enhanced bone contrast for clinical appearance
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (1.8 + bone_hu / 500.0)
    
    # Apply smoothing to reduce noise
    mu_volume = ndimage.gaussian_filter(mu_volume, sigma=[0.5, 0.5, 0.5])
    
    print(f"Attenuation range: [{mu_volume.min():.4f}, {mu_volume.max():.4f}] mm^-1")
    
    # Generate projection based on view
    if projection_type == 'AP':
        # AP view: integrate along Y axis (anterior-posterior)
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)  # Correct orientation
    else:  # Lateral
        # Lateral view: integrate along X axis
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
    
    print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    
    # Apply Beer-Lambert law with clinical scaling
    # Chest thickness typically 200-300mm, but we need to account for the actual data
    avg_thickness = np.percentile(projection, 75) / mu_water
    print(f"Estimated average chest thickness: {avg_thickness:.1f} mm")
    
    # Normalize projection for consistent appearance
    projection_normalized = projection / avg_thickness / mu_water
    
    # Apply exponential attenuation
    transmission = np.exp(-projection_normalized * 3.0)  # Scale factor for contrast
    
    # Convert to optical density (logarithmic response)
    # This mimics film/detector response
    od = -np.log10(transmission + 1e-6)
    
    # Apply clinical contrast curve
    # S-curve (sigmoid) transformation for film-like appearance
    od_normalized = (od - np.mean(od)) / (np.std(od) + 1e-6)
    contrast_enhanced = 1.0 / (1.0 + np.exp(-3.0 * od_normalized))
    
    # Additional contrast adjustments for clinical appearance
    # Enhance dynamic range in lung fields
    lung_region = contrast_enhanced < 0.3
    contrast_enhanced[lung_region] = contrast_enhanced[lung_region] * 0.7
    
    # Enhance bone contrast
    bone_region = contrast_enhanced > 0.7
    contrast_enhanced[bone_region] = 0.7 + (contrast_enhanced[bone_region] - 0.7) * 1.5
    
    # Apply gamma correction for proper display
    gamma = 1.8  # Higher gamma for more clinical appearance
    drr = np.power(contrast_enhanced, 1.0 / gamma)
    
    # Final adjustments
    # Ensure full dynamic range usage
    drr = (drr - drr.min()) / (drr.max() - drr.min())
    
    # Apply subtle edge enhancement (unsharp masking)
    blurred = ndimage.gaussian_filter(drr, sigma=2)
    drr = drr + 0.3 * (drr - blurred)
    drr = np.clip(drr, 0, 1)
    
    # Invert for radiographic convention (dense = white)
    drr = 1.0 - drr
    
    print(f"DRR range: [{drr.min():.3f}, {drr.max():.3f}]")
    
    return drr


def add_clinical_annotations(ax, projection_type='AP'):
    """Add clinical annotations to the X-ray image."""
    if projection_type == 'AP':
        # Add R/L markers
        ax.text(0.02, 0.98, 'R', transform=ax.transAxes, 
                fontsize=20, color='white', ha='left', va='top', 
                weight='bold', family='sans-serif')
        ax.text(0.98, 0.98, 'L', transform=ax.transAxes, 
                fontsize=20, color='white', ha='right', va='top',
                weight='bold', family='sans-serif')
    
    # Add technique factors (simulated)
    technique = "120 kVp, 3 mAs, 180 cm"
    ax.text(0.02, 0.02, technique, transform=ax.transAxes,
            fontsize=10, color='white', ha='left', va='bottom',
            family='monospace', alpha=0.7)


def main():
    """Generate clinical-quality chest X-ray DRRs."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/clinical")
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
    
    # Generate clinical AP view
    print("\n=== Generating Clinical AP View ===")
    drr_ap = generate_clinical_drr(ct_volume, 'AP')
    
    # Generate clinical lateral view
    print("\n=== Generating Clinical Lateral View ===")
    drr_lateral = generate_clinical_drr(ct_volume, 'Lateral')
    
    # Save AP view with clinical appearance
    fig = plt.figure(figsize=(10, 12), facecolor='black')
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(drr_ap, cmap='gray', aspect='auto', interpolation='bilinear')
    add_clinical_annotations(ax, 'AP')
    ax.axis('off')
    plt.savefig(output_dir / 'clinical_chest_xray_ap.png', 
                dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black', edgecolor='none')
    plt.close()
    
    # Save lateral view
    fig = plt.figure(figsize=(10, 12), facecolor='black')
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(drr_lateral, cmap='gray', aspect='auto', interpolation='bilinear')
    ax.axis('off')
    plt.savefig(output_dir / 'clinical_chest_xray_lateral.png',
                dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black', edgecolor='none')
    plt.close()
    
    # Create clinical presentation (both views)
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), facecolor='black')
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.05)
    
    # AP view
    axes[0].imshow(drr_ap, cmap='gray', aspect='auto', interpolation='bilinear')
    axes[0].text(0.5, 1.02, 'PA CHEST', transform=axes[0].transAxes,
                 fontsize=14, color='white', ha='center', va='bottom')
    add_clinical_annotations(axes[0], 'AP')
    axes[0].axis('off')
    
    # Lateral view  
    axes[1].imshow(drr_lateral, cmap='gray', aspect='auto', interpolation='bilinear')
    axes[1].text(0.5, 1.02, 'LATERAL', transform=axes[1].transAxes,
                 fontsize=14, color='white', ha='center', va='bottom')
    axes[1].axis('off')
    
    plt.savefig(output_dir / 'clinical_chest_xray_presentation.png',
                dpi=300, bbox_inches='tight', pad_inches=0.1,
                facecolor='black', edgecolor='none')
    plt.close()
    
    # Create comparison with enhanced lung detail view
    # This helps visualize subtle lung pathology
    lung_window_volume = apply_clinical_window(
        sitk.GetArrayFromImage(ct_volume), 
        window_center=-600, 
        window_width=1500
    )
    
    # Generate lung-optimized DRR
    print("\n=== Generating Lung-Optimized View ===")
    ct_lung = sitk.GetImageFromArray(lung_window_volume)
    ct_lung.CopyInformation(ct_volume)
    drr_lung = generate_clinical_drr(ct_lung, 'AP')
    
    # Save lung-optimized view
    fig = plt.figure(figsize=(10, 12), facecolor='black')
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(drr_lung, cmap='gray', aspect='auto', interpolation='bilinear')
    ax.text(0.5, 1.02, 'LUNG WINDOW', transform=ax.transAxes,
            fontsize=14, color='white', ha='center', va='bottom')
    add_clinical_annotations(ax, 'AP')
    ax.axis('off')
    plt.savefig(output_dir / 'clinical_chest_xray_lung_window.png',
                dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black', edgecolor='none')
    plt.close()
    
    print(f"\nClinical chest X-rays saved to {output_dir}")
    print("Files created:")
    print("  - clinical_chest_xray_ap.png (Standard PA chest)")
    print("  - clinical_chest_xray_lateral.png") 
    print("  - clinical_chest_xray_presentation.png (Both views)")
    print("  - clinical_chest_xray_lung_window.png (Optimized for lung pathology)")


if __name__ == "__main__":
    main()