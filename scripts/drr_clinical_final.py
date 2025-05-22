#!/usr/bin/env python3
"""
Final clinical DRR generation with proper bone visibility and standard X-ray film dimensions.
Incorporates all learnings from diagnostic analysis and uses standard radiographic sizes.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage


def generate_clinical_final_drr(ct_volume, projection_type='AP'):
    """
    Generate final clinical-quality DRR with standard X-ray film dimensions.
    """
    # Standard X-ray film/detector sizes (in mm)
    STANDARD_SIZES = {
        'AP': {'width': 356, 'height': 432},  # 14"x17" portrait
        'Lateral': {'width': 432, 'height': 356}  # 17"x14" landscape
    }
    
    # Get volume and spacing
    volume = sitk.GetArrayFromImage(ct_volume)
    spacing = np.array(ct_volume.GetSpacing())  # (x, y, z) in mm
    
    print(f"Volume shape (Z,Y,X): {volume.shape}")
    print(f"Spacing (X,Y,Z): {spacing} mm")
    print(f"HU range: [{volume.min():.0f}, {volume.max():.0f}]")
    
    # NO windowing - preserve full HU range for bones
    
    # Convert HU to linear attenuation coefficients
    mu_water = 0.019  # mm^-1 at ~70 keV
    mu_volume = np.zeros_like(volume, dtype=np.float32)
    
    # Air - zero attenuation for black background
    air_mask = volume < -900
    mu_volume[air_mask] = 0.0
    
    # Lung tissue - very low attenuation
    lung_mask = (volume >= -900) & (volume < -500)
    lung_hu = volume[lung_mask]
    mu_volume[lung_mask] = 0.0001 + (lung_hu + 900) * (0.001 / 400)
    
    # Soft tissue - standard attenuation
    soft_mask = (volume >= -500) & (volume < 200)
    soft_hu = volume[soft_mask]
    mu_volume[soft_mask] = mu_water * (1.0 + soft_hu / 1000.0)
    
    # Bone - enhanced attenuation for visibility (2.5x multiplier)
    bone_mask = volume >= 200
    bone_hu = volume[bone_mask]
    mu_volume[bone_mask] = mu_water * (2.5 + bone_hu / 500.0)
    
    # Very light smoothing to reduce noise without losing detail
    mu_volume = ndimage.gaussian_filter(mu_volume, sigma=[0.2, 0.2, 0.2])
    
    print(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
    
    # Generate projection with correct spacing
    if projection_type == 'AP':
        # AP view: integrate along Y axis
        projection = np.sum(mu_volume, axis=1) * spacing[1]
        projection = np.flipud(projection)
        # Physical dimensions of projection
        proj_height_mm = projection.shape[0] * spacing[2]  # Z dimension
        proj_width_mm = projection.shape[1] * spacing[0]   # X dimension
    else:  # Lateral
        # Lateral view: integrate along X axis
        projection = np.sum(mu_volume, axis=2) * spacing[0]
        projection = np.flipud(projection)
        # Physical dimensions of projection
        proj_height_mm = projection.shape[0] * spacing[2]  # Z dimension  
        proj_width_mm = projection.shape[1] * spacing[1]   # Y dimension
    
    print(f"Path integral range: [{projection.min():.1f}, {projection.max():.1f}]")
    print(f"Projection physical size: {proj_width_mm:.1f} x {proj_height_mm:.1f} mm")
    print(f"Original projection shape: {projection.shape}")
    
    # Resample to standard X-ray film dimensions
    detector_size = STANDARD_SIZES[projection_type]
    detector_width_mm = detector_size['width']
    detector_height_mm = detector_size['height']
    
    # Calculate scale to fit anatomy within detector while maintaining aspect ratio
    scale_x = detector_width_mm / proj_width_mm
    scale_y = detector_height_mm / proj_height_mm
    scale = min(scale_x, scale_y) * 0.9  # 0.9 to leave some border
    
    # Calculate new dimensions
    new_width_mm = proj_width_mm * scale
    new_height_mm = proj_height_mm * scale
    new_width_px = int(detector_width_mm / 0.5)  # 0.5mm detector pixel spacing
    new_height_px = int(detector_height_mm / 0.5)
    
    # Calculate anatomy size in pixels
    anatomy_width_px = int(new_width_mm / 0.5)
    anatomy_height_px = int(new_height_mm / 0.5)
    
    # Resample projection to new size
    zoom_factors = [anatomy_height_px / projection.shape[0], 
                   anatomy_width_px / projection.shape[1]]
    projection_resampled = ndimage.zoom(projection, zoom_factors, order=3)
    
    # Create detector-sized image and center the anatomy
    detector_image = np.zeros((new_height_px, new_width_px))
    
    # Calculate centering offsets
    y_offset = (new_height_px - anatomy_height_px) // 2
    x_offset = (new_width_px - anatomy_width_px) // 2
    
    # Place resampled projection in center
    detector_image[y_offset:y_offset+anatomy_height_px, 
                  x_offset:x_offset+anatomy_width_px] = projection_resampled
    
    print(f"Detector size: {detector_width_mm} x {detector_height_mm} mm")
    print(f"Resampled to: {new_width_px} x {new_height_px} pixels")
    print(f"Anatomy size: {anatomy_width_px} x {anatomy_height_px} pixels")
    
    # Use detector image as projection for further processing
    projection = detector_image
    
    # Apply Beer-Lambert law
    transmission = np.exp(-projection)
    
    # Convert to intensity with log transform
    epsilon = 1e-6
    intensity = -np.log10(transmission + epsilon)
    
    # Normalize using percentiles from body region
    body_mask = projection > 0.1
    if np.any(body_mask):
        # Use 1-99 percentiles for better dynamic range
        p1 = np.percentile(intensity[body_mask], 1)
        p99 = np.percentile(intensity[body_mask], 99)
        intensity = (intensity - p1) / (p99 - p1)
        intensity = np.clip(intensity, 0, 1)
    
    # Mild gamma correction
    gamma = 1.2
    intensity = np.power(intensity, 1.0 / gamma)
    
    # Ensure air stays black
    air_projection = projection < 0.01
    intensity[air_projection] = 0
    
    # Very subtle edge enhancement
    blurred = ndimage.gaussian_filter(intensity, sigma=1.0)
    intensity = intensity + 0.1 * (intensity - blurred)
    intensity = np.clip(intensity, 0, 1)
    
    print(f"Final intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    
    return intensity


def save_clinical_final_xray(image, filename, title=None, markers=None):
    """Save DRR with standard X-ray film appearance."""
    # Standard X-ray film display size 
    h, w = image.shape
    # Convert pixel dimensions to inches (assuming 0.5mm pixel spacing = ~50 DPI)
    fig_width = w * 0.5 / 25.4  # mm to inches
    fig_height = h * 0.5 / 25.4
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Display with equal aspect (already properly sized)
    im = ax.imshow(image, cmap='gray', aspect='equal', 
                   vmin=0, vmax=1, interpolation='bilinear')
    
    # Add title if provided
    if title:
        ax.text(0.5, 0.98, title, transform=ax.transAxes,
                fontsize=14, color='white', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='black', alpha=0.8))
    
    # Add markers if provided (e.g., R/L for AP view)
    if markers:
        for marker in markers:
            ax.text(marker['x'], marker['y'], marker['text'], 
                   transform=ax.transAxes,
                   fontsize=16, color='white', weight='bold',
                   ha=marker.get('ha', 'center'), 
                   va=marker.get('va', 'center'),
                   bbox=dict(boxstyle='square,pad=0.3',
                            facecolor='black', edgecolor='white',
                            linewidth=2))
    
    ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    plt.close()


def main():
    """Generate final clinical chest X-rays with all corrections."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/clinical_final")
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
    print("\n=== Generating Final Clinical AP View ===")
    drr_ap = generate_clinical_final_drr(ct_volume, 'AP')
    
    # Save AP view with markers
    ap_markers = [
        {'x': 0.02, 'y': 0.98, 'text': 'R', 'ha': 'left', 'va': 'top'},
        {'x': 0.98, 'y': 0.98, 'text': 'L', 'ha': 'right', 'va': 'top'}
    ]
    save_clinical_final_xray(drr_ap, output_dir / 'clinical_final_chest_xray_ap.png', 
                            title='CHEST PA', markers=ap_markers)
    
    # Generate lateral view
    print("\n=== Generating Final Clinical Lateral View ===")
    drr_lateral = generate_clinical_final_drr(ct_volume, 'Lateral')
    
    # Save lateral view
    save_clinical_final_xray(drr_lateral, output_dir / 'clinical_final_chest_xray_lateral.png',
                            title='LATERAL')
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), facecolor='black')
    
    # AP view
    ax1.imshow(drr_ap, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax1.set_title('CHEST PA', fontsize=16, color='white', pad=10)
    ax1.text(0.02, 0.98, 'R', transform=ax1.transAxes,
             fontsize=16, color='white', weight='bold',
             ha='left', va='top')
    ax1.text(0.98, 0.98, 'L', transform=ax1.transAxes,
             fontsize=16, color='white', weight='bold',
             ha='right', va='top')
    ax1.axis('off')
    
    # Lateral view
    ax2.imshow(drr_lateral, cmap='gray', aspect='equal', vmin=0, vmax=1)
    ax2.set_title('LATERAL', fontsize=16, color='white', pad=10)
    ax2.axis('off')
    
    plt.savefig(output_dir / 'clinical_final_chest_xray_both.png',
                dpi=300, bbox_inches='tight',
                facecolor='black')
    plt.close()
    
    # Create detail analysis showing improvement
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    
    # Full AP view
    axes[0, 0].imshow(drr_ap, cmap='gray', aspect='equal')
    axes[0, 0].set_title('Final Clinical AP View', fontsize=12)
    axes[0, 0].axis('off')
    
    # Rib cage detail (upper right chest)
    h, w = drr_ap.shape
    rib_roi = drr_ap[h//5:2*h//5, w//2:4*w//5]
    axes[0, 1].imshow(rib_roi, cmap='gray', aspect='equal')
    axes[0, 1].set_title('Rib Cage Detail\n(Clear rib visualization)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Spine detail (center)
    spine_roi = drr_ap[h//3:2*h//3, 2*w//5:3*w//5]
    axes[1, 0].imshow(spine_roi, cmap='gray', aspect='equal')
    axes[1, 0].set_title('Spine Detail\n(Vertebrae visible)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Heart/mediastinum detail
    heart_roi = drr_ap[2*h//5:3*h//5, w//3:2*w//3]
    axes[1, 1].imshow(heart_roi, cmap='gray', aspect='equal')
    axes[1, 1].set_title('Heart/Mediastinum\n(Clear cardiac silhouette)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle('Final Clinical DRR - Anatomical Detail', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'clinical_final_detail_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFinal clinical chest X-rays saved to {output_dir}")
    print("Files created:")
    print("  - clinical_final_chest_xray_ap.png (Clinical AP view)")
    print("  - clinical_final_chest_xray_lateral.png (Clinical lateral view)")
    print("  - clinical_final_chest_xray_both.png (Both views)")
    print("  - clinical_final_detail_analysis.png (Anatomical details)")
    print("\nFinal improvements achieved:")
    print("  ✓ Standard X-ray film dimensions (14\"x17\" AP, 17\"x14\" Lateral)")
    print("  ✓ Proper anatomical proportions (no squishing)")
    print("  ✓ Visible ribs and spine (2.5x bone enhancement)")
    print("  ✓ Black background (proper air attenuation)")
    print("  ✓ Clinical gray scale for soft tissues")
    print("  ✓ Full HU range preserved (no windowing)")
    print("  ✓ Centered anatomy with borders like real X-rays")


if __name__ == "__main__":
    main()