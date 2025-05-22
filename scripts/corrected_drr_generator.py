#!/usr/bin/env python3
"""
Corrected High-Quality DRR Generator

This fixes the numerical issues and creates clinically-realistic DRRs
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_ct_volume():
    """Load CT volume and return properly formatted data"""
    try:
        import SimpleITK as sitk
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SimpleITK"])
        import SimpleITK as sitk
    
    # Find DICOM directory
    download_dirs = glob.glob("data/tciaDownload/*")
    dicom_dirs = [d for d in download_dirs if os.path.isdir(d)]
    
    dicom_dir = dicom_dirs[0]
    print(f"📁 Loading CT: {os.path.basename(dicom_dir)}")
    
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Convert to numpy
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    
    print(f"📐 Volume: {volume.shape}, Spacing: {spacing} mm")
    print(f"🎯 HU range: {volume.min():.0f} to {volume.max():.0f}")
    
    return volume, spacing

def create_realistic_attenuation_map(volume):
    """Create realistic attenuation map for different tissue types"""
    print("🔧 Creating realistic attenuation map...")
    
    # Clip extreme values
    hu_clipped = np.clip(volume, -1000, 3000)
    
    # Realistic attenuation coefficients at 120 kVp (in mm^-1)
    mu_air = 0.0
    mu_water = 0.019
    mu_soft_tissue = 0.020
    mu_bone = 0.048
    
    # Create attenuation map based on HU values
    mu_map = np.zeros_like(hu_clipped, dtype=np.float32)
    
    # Air/lung (HU < -500): Approximately exponential from air to water
    air_mask = hu_clipped < -500
    air_fraction = (hu_clipped[air_mask] + 1000) / 500.0  # 0 to 1
    mu_map[air_mask] = mu_air + (mu_water - mu_air) * air_fraction
    
    # Soft tissue (-500 <= HU <= 100): Linear from water to soft tissue
    soft_mask = (hu_clipped >= -500) & (hu_clipped <= 100)
    soft_fraction = (hu_clipped[soft_mask] + 500) / 600.0  # 0 to 1
    mu_map[soft_mask] = mu_water + (mu_soft_tissue - mu_water) * soft_fraction
    
    # Bone (HU > 100): Progressive increase with HU
    bone_mask = hu_clipped > 100
    # Cap bone density to prevent extreme values
    bone_hu_capped = np.minimum(hu_clipped[bone_mask], 2000)
    bone_fraction = bone_hu_capped / 2000.0  # 0 to 1 for HU 0-2000
    mu_map[bone_mask] = mu_soft_tissue + (mu_bone - mu_soft_tissue) * bone_fraction
    
    # Ensure all values are non-negative
    mu_map = np.maximum(mu_map, 0.0)
    
    print(f"📊 Attenuation range: {mu_map.min():.4f} to {mu_map.max():.4f} mm^-1")
    
    return mu_map

def generate_high_quality_drr(mu_volume, spacing, projection_angle='AP'):
    """Generate high-quality DRR with proper physics"""
    print(f"🎯 Generating {projection_angle} DRR...")
    
    if projection_angle == 'AP':
        # Anterior-Posterior: sum along Y-axis (axis=1)
        path_length = np.sum(mu_volume, axis=1) * spacing[1]
    elif projection_angle == 'Lateral':
        # Lateral: sum along X-axis (axis=2) 
        path_length = np.sum(mu_volume, axis=2) * spacing[0]
    else:
        raise ValueError("projection_angle must be 'AP' or 'Lateral'")
    
    print(f"📐 Projection shape: {path_length.shape}")
    print(f"📊 Path length range: {path_length.min():.3f} to {path_length.max():.3f} mm")
    
    # Apply Beer-Lambert law: I = I0 * exp(-μ*t)
    # Use I0 = 1000 for realistic X-ray source intensity
    I0 = 1000.0
    
    # Cap path lengths to prevent numerical overflow
    path_length_capped = np.minimum(path_length, 15.0)  # Cap at 15 mm equivalent
    
    # Calculate transmitted intensity
    intensity = I0 * np.exp(-path_length_capped)
    
    print(f"💡 Intensity range: {intensity.min():.1f} to {intensity.max():.1f}")
    
    return intensity

def enhance_for_xray_display(intensity):
    """Convert intensity to X-ray-like display values"""
    print("🎨 Converting to X-ray display...")
    
    # X-ray images typically show log of inverse intensity
    # Add small offset to prevent log(0)
    epsilon = 1.0
    
    # Calculate attenuation from intensity: A = log(I0/I)
    I0 = intensity.max()  # Use max intensity as reference
    attenuation = np.log((I0 + epsilon) / (intensity + epsilon))
    
    # Normalize to 0-1 range
    if attenuation.max() > attenuation.min():
        normalized = (attenuation - attenuation.min()) / (attenuation.max() - attenuation.min())
    else:
        normalized = np.zeros_like(attenuation)
    
    # Apply gamma correction for better contrast
    gamma = 0.7  # Slightly brighten shadows
    enhanced = np.power(normalized, gamma)
    
    # Convert to 8-bit
    display_image = (255 * enhanced).astype(np.uint8)
    
    print(f"✨ Display range: {display_image.min()} to {display_image.max()}")
    
    return display_image

def create_maximum_intensity_projection(volume):
    """Create MIP for bone visualization"""
    print("🦴 Creating Maximum Intensity Projection for bone...")
    
    # Apply bone window
    bone_windowed = np.clip(volume, 200, 1500)
    
    # MIP projections
    mip_ap = np.max(bone_windowed, axis=1)
    mip_lateral = np.max(bone_windowed, axis=2)
    
    # Normalize and enhance
    def enhance_mip(mip_data):
        normalized = (mip_data - mip_data.min()) / (mip_data.max() - mip_data.min())
        # Invert so bone appears bright
        inverted = 1.0 - normalized
        enhanced = (255 * inverted).astype(np.uint8)
        return enhanced
    
    mip_ap_enhanced = enhance_mip(mip_ap)
    mip_lateral_enhanced = enhance_mip(mip_lateral)
    
    return mip_ap_enhanced, mip_lateral_enhanced

def save_high_quality_drr(drr_ap, drr_lateral, mip_ap, mip_lateral):
    """Save high-quality DRR images"""
    print("💾 Saving high-quality DRR images...")
    
    # Individual DRR images
    def save_single_view(image_data, filename, title, is_ap=True):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(np.flipud(image_data), cmap='gray', aspect='auto')
        ax.set_title(f'{title}\nHigh-Quality Chest DRR (LUNG1-001)', fontsize=14, pad=20)
        ax.axis('off')
        
        # Add orientation labels
        if is_ap:
            ax.text(0.02, 0.98, 'R', transform=ax.transAxes, fontsize=16, 
                   color='white', weight='bold', ha='left', va='top')
            ax.text(0.98, 0.98, 'L', transform=ax.transAxes, fontsize=16, 
                   color='white', weight='bold', ha='right', va='top')
        else:
            ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=16,
                   color='white', weight='bold', ha='left', va='top')
            ax.text(0.98, 0.98, 'P', transform=ax.transAxes, fontsize=16,
                   color='white', weight='bold', ha='right', va='top')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        plt.close()
    
    # Save individual views
    save_single_view(drr_ap, "outputs/final/drr_corrected_ap.png", "Anterior-Posterior DRR", True)
    save_single_view(drr_lateral, "outputs/final/drr_corrected_lateral.png", "Lateral DRR", False)
    save_single_view(mip_ap, "outputs/final/drr_mip_ap.png", "MIP Anterior-Posterior (Bone)", True)
    save_single_view(mip_lateral, "outputs/final/drr_mip_lateral.png", "MIP Lateral (Bone)", False)
    
    # Create comprehensive comparison
    fig = plt.figure(figsize=(20, 12))
    
    # DRR AP
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(np.flipud(drr_ap), cmap='gray', aspect='auto')
    ax1.set_title('DRR - Anterior-Posterior', fontsize=14)
    ax1.axis('off')
    ax1.text(0.02, 0.98, 'R', transform=ax1.transAxes, fontsize=14, 
            color='white', weight='bold', ha='left', va='top')
    ax1.text(0.98, 0.98, 'L', transform=ax1.transAxes, fontsize=14, 
            color='white', weight='bold', ha='right', va='top')
    
    # DRR Lateral
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(np.flipud(drr_lateral), cmap='gray', aspect='auto')
    ax2.set_title('DRR - Lateral', fontsize=14)
    ax2.axis('off')
    ax2.text(0.02, 0.98, 'A', transform=ax2.transAxes, fontsize=14,
            color='white', weight='bold', ha='left', va='top')
    ax2.text(0.98, 0.98, 'P', transform=ax2.transAxes, fontsize=14,
            color='white', weight='bold', ha='right', va='top')
    
    # MIP AP
    ax3 = plt.subplot(2, 3, 4)
    ax3.imshow(np.flipud(mip_ap), cmap='gray', aspect='auto')
    ax3.set_title('MIP - Anterior-Posterior\n(Bone Structures)', fontsize=14)
    ax3.axis('off')
    ax3.text(0.02, 0.98, 'R', transform=ax3.transAxes, fontsize=14,
            color='white', weight='bold', ha='left', va='top')
    ax3.text(0.98, 0.98, 'L', transform=ax3.transAxes, fontsize=14,
            color='white', weight='bold', ha='right', va='top')
    
    # MIP Lateral
    ax4 = plt.subplot(2, 3, 5)
    ax4.imshow(np.flipud(mip_lateral), cmap='gray', aspect='auto')
    ax4.set_title('MIP - Lateral\n(Bone Structures)', fontsize=14)
    ax4.axis('off')
    ax4.text(0.02, 0.98, 'A', transform=ax4.transAxes, fontsize=14,
            color='white', weight='bold', ha='left', va='top')
    ax4.text(0.98, 0.98, 'P', transform=ax4.transAxes, fontsize=14,
            color='white', weight='bold', ha='right', va='top')
    
    # Combined view (side by side DRRs)
    ax5 = plt.subplot(1, 3, 3)
    combined = np.hstack([drr_ap, drr_lateral])
    ax5.imshow(np.flipud(combined), cmap='gray', aspect='auto')
    ax5.set_title('Combined DRR Views\nAP + Lateral', fontsize=14)
    ax5.axis('off')
    
    plt.suptitle('High-Quality DRR Suite - Chest CT (LUNG1-001, NSCLC-Radiomics)', 
                fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig("outputs/final/drr_final_suite.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print("✅ High-quality DRR suite saved to outputs/final/:")
    print("   • drr_corrected_ap.png - Realistic AP DRR")
    print("   • drr_corrected_lateral.png - Realistic lateral DRR")
    print("   • drr_mip_ap.png - Bone structures AP")
    print("   • drr_mip_lateral.png - Bone structures lateral")
    print("   • drr_final_suite.png - Complete comparison")

def main():
    """Generate corrected high-quality DRRs"""
    print("🏥 === High-Quality DRR Generator (Corrected) ===\n")
    
    # Load CT data
    volume, spacing = load_ct_volume()
    
    # Create realistic attenuation map
    mu_volume = create_realistic_attenuation_map(volume)
    
    # Generate DRRs
    print("\n🎯 === Generating High-Quality DRRs ===")
    intensity_ap = generate_high_quality_drr(mu_volume, spacing, 'AP')
    intensity_lateral = generate_high_quality_drr(mu_volume, spacing, 'Lateral')
    
    # Convert to display format
    drr_ap = enhance_for_xray_display(intensity_ap)
    drr_lateral = enhance_for_xray_display(intensity_lateral)
    
    # Generate MIP for bone visualization
    mip_ap, mip_lateral = create_maximum_intensity_projection(volume)
    
    # Save all results
    save_high_quality_drr(drr_ap, drr_lateral, mip_ap, mip_lateral)
    
    print(f"\n🎉 === High-Quality DRR Generation Complete ===")
    print("The corrected DRRs should now look much more like real chest X-rays!")
    print("- Better contrast and dynamic range")
    print("- Realistic tissue differentiation") 
    print("- Proper bone visualization in MIP images")
    print("- Clinical-quality appearance")

if __name__ == "__main__":
    main() 