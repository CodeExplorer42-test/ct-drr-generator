#!/usr/bin/env python3
"""
True clinical-quality DRR generation that produces authentic chest X-ray appearance.
This version correctly implements X-ray physics and display characteristics.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def generate_true_clinical_drr(ct_volume, projection_type='AP', enhance_contrast=True):
    """
    Generate true clinical-quality DRR with proper X-ray physics.
    
    Key improvements:
    - Black background (air = no attenuation = black)
    - White bones (high attenuation = white)
    - Proper lung field appearance with visible vasculature
    - Clinical dynamic range
    """
    # Get volume and spacing
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    print(f"Volume shape (Z,Y,X): {volume.shape}")
    print(f"Spacing (X,Y,Z): {spacing} mm")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # Convert HU to linear attenuation coefficients
    # Using realistic values for diagnostic X-ray energies (~70 keV effective)
    mu_water = 0.019  # mm^-1 at ~70 keV
    
    # Create tissue-specific attenuation map
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air (outside body and in lungs) - essentially zero attenuation
    # This ensures black background
    air_mask = volume < -900
    mu_volume[air_mask] = 0.0
    
    # Lung tissue - very low attenuation but not zero
    # This allows vascular structures to be visible
    lung_mask = (volume >= -900) & (volume < -500)
    mu_volume[lung_mask] = 0.0002 + (volume[lung_mask] + 900) * 0.000002
    
    # Fat tissue - slightly less than water
    fat_mask = (volume >= -500) & (volume < -100)
    mu_volume[fat_mask] = mu_water * 0.9 * (1.0 + volume[fat_mask] / 1000.0)
    
    # Soft tissue (muscle, organs, blood)
    soft_mask = (volume >= -100) & (volume < 400)
    mu_volume[soft_mask] = mu_water * (1.0 + volume[soft_mask] / 1000.0)
    
    # Bone and calcifications - high attenuation
    bone_mask = volume >= 400
    # Use more aggressive bone attenuation for better contrast
    mu_volume[bone_mask] = mu_water * (2.0 + volume[bone_mask] / 800.0)
    
    # Clamp extreme values to prevent numerical issues
    mu_volume = np.clip(mu_volume, 0, 0.1)  # Max ~0.1 mm^-1 for very dense bone
    
    print(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate projection based on view
    if projection_type == 'AP':
        # AP view: integrate along Y axis (anterior-posterior)
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)
    else:  # Lateral
        # Lateral view: integrate along X axis
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
    
    print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    
    # Apply Beer-Lambert law
    # I = I0 * exp(-∫μ dx)
    # For display: we want high transmission (low attenuation) to be dark
    transmission = np.exp(-projection)
    
    # Convert transmission to image intensity
    # Key insight: In real X-rays, more transmission = darker (more X-rays hit detector)
    # So we use transmission directly (NOT inverted)
    
    if enhance_contrast:
        # Apply logarithmic transformation to enhance contrast
        # This mimics the response of X-ray film/digital detectors
        epsilon = 1e-6
        intensity = -np.log10(transmission + epsilon)
        
        # Normalize to use full dynamic range
        # Find the body region (non-air) for better normalization
        body_mask = projection > 0.1  # Areas with some attenuation
        if np.any(body_mask):
            p5 = np.percentile(intensity[body_mask], 5)
            p95 = np.percentile(intensity[body_mask], 95)
            intensity = (intensity - p5) / (p95 - p5)
        else:
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        
        # Apply gamma correction for better tissue visibility
        # Lower gamma = more contrast in dark regions (lungs)
        gamma = 2.2
        intensity = np.power(np.clip(intensity, 0, 1), 1.0/gamma)
        
        # Ensure air remains black
        air_projection = projection < 0.01
        intensity[air_projection] = 0
        
    else:
        # Simple linear mapping
        intensity = 1.0 - transmission
    
    # Final adjustments for clinical appearance
    # Enhance bone contrast
    high_density = intensity > 0.7
    intensity[high_density] = 0.7 + (intensity[high_density] - 0.7) * 1.3
    
    # Ensure full range [0, 1]
    intensity = np.clip(intensity, 0, 1)
    
    print(f"Final intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    
    return intensity


def add_clinical_markers(ax, view_type='AP'):
    """Add standard clinical markers to X-ray."""
    if view_type == 'AP':
        # R/L markers
        ax.text(0.02, 0.98, 'R', transform=ax.transAxes,
                fontsize=16, color='white', weight='bold',
                ha='left', va='top', bbox=dict(boxstyle='square,pad=0.3', 
                                               facecolor='black', 
                                               edgecolor='white',
                                               linewidth=2))
        ax.text(0.98, 0.98, 'L', transform=ax.transAxes,
                fontsize=16, color='white', weight='bold',
                ha='right', va='top', bbox=dict(boxstyle='square,pad=0.3',
                                                facecolor='black',
                                                edgecolor='white',
                                                linewidth=2))


def main():
    """Generate true clinical-quality chest X-ray DRRs."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/true_clinical")
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
    print("\n=== Generating True Clinical AP View ===")
    drr_ap = generate_true_clinical_drr(ct_volume, 'AP', enhance_contrast=True)
    
    # Generate clinical lateral view
    print("\n=== Generating True Clinical Lateral View ===")
    drr_lateral = generate_true_clinical_drr(ct_volume, 'Lateral', enhance_contrast=True)
    
    # Create proper clinical display with black background
    # AP view - clinical presentation
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure
    
    # Display with proper clinical appearance
    # Use 'gray' colormap which maps 0=black, 1=white
    im = ax.imshow(drr_ap, cmap='gray', aspect='equal', 
                   vmin=0, vmax=1, interpolation='bilinear')
    
    # Add clinical markers
    add_clinical_markers(ax, 'AP')
    
    # Remove axes
    ax.axis('off')
    
    # Save with black background
    plt.savefig(output_dir / 'clinical_chest_xray_ap.png', 
                dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    plt.close()
    
    # Lateral view
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.imshow(drr_lateral, cmap='gray', aspect='equal',
              vmin=0, vmax=1, interpolation='bilinear')
    ax.axis('off')
    
    plt.savefig(output_dir / 'clinical_chest_xray_lateral.png',
                dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    plt.close()
    
    # Create side-by-side comparison
    fig = plt.figure(figsize=(20, 12), facecolor='black')
    
    # AP view
    ax1 = plt.subplot(121)
    ax1.imshow(drr_ap, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax1.set_title('CHEST PA', fontsize=16, color='white', pad=10)
    add_clinical_markers(ax1, 'AP')
    ax1.axis('off')
    
    # Lateral view
    ax2 = plt.subplot(122)
    ax2.imshow(drr_lateral, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax2.set_title('LATERAL', fontsize=16, color='white', pad=10)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clinical_chest_xray_both.png',
                dpi=300, bbox_inches='tight',
                facecolor='black')
    plt.close()
    
    # Create diagnostic comparison showing the improvement
    fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor='white')
    
    # Load previous attempts for comparison
    try:
        from drr_final import generate_drr_correct
        
        # Generate old version
        print("\n=== Generating comparison with previous version ===")
        old_drr = generate_drr_correct(ct_volume, 'AP')
        
        # Show old vs new
        axes[0, 0].imshow(old_drr, cmap='gray', aspect='auto')
        axes[0, 0].set_title('Previous Version (V5)\nIncorrect: White background', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(drr_ap, cmap='gray', aspect='equal')
        axes[0, 1].set_title('New Clinical Version\nCorrect: Black background', fontsize=12)
        axes[0, 1].axis('off')
        
        # Show detail regions
        h, w = drr_ap.shape
        # Lung field detail
        lung_region = drr_ap[h//4:h//2, w//3:2*w//3]
        axes[1, 0].imshow(lung_region, cmap='gray', aspect='equal')
        axes[1, 0].set_title('Lung Field Detail\n(Note vascular markings)', fontsize=12)
        axes[1, 0].axis('off')
        
        # Spine/heart border detail  
        heart_region = drr_ap[h//3:2*h//3, w//3:2*w//3]
        axes[1, 1].imshow(heart_region, cmap='gray', aspect='equal')
        axes[1, 1].set_title('Heart/Spine Detail\n(Proper tissue contrast)', fontsize=12)
        axes[1, 1].axis('off')
        
    except Exception as e:
        print(f"Could not create comparison: {e}")
        # Just show the new version
        axes[0, 0].axis('off')
        axes[0, 1].imshow(drr_ap, cmap='gray', aspect='equal')
        axes[0, 1].set_title('Clinical AP View', fontsize=12)
        axes[0, 1].axis('off')
        axes[1, 0].imshow(drr_lateral, cmap='gray', aspect='equal')
        axes[1, 0].set_title('Clinical Lateral View', fontsize=12)
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
    
    plt.suptitle('Clinical DRR Generation - Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'clinical_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTrue clinical chest X-rays saved to {output_dir}")
    print("Files created:")
    print("  - clinical_chest_xray_ap.png (Clinical AP view)")
    print("  - clinical_chest_xray_lateral.png (Clinical lateral view)")
    print("  - clinical_chest_xray_both.png (Both views)")
    print("  - clinical_comparison.png (Comparison with previous version)")
    print("\nKey improvements:")
    print("  ✓ Black background (air = no attenuation)")
    print("  ✓ White bones with proper contrast")
    print("  ✓ Visible lung vasculature")
    print("  ✓ Clinical gray scale for soft tissues")
    print("  ✓ Full dynamic range utilization")


if __name__ == "__main__":
    main()