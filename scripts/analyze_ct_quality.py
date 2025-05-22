#!/usr/bin/env python3
"""
Analyze CT Data Quality and Compare with DRR Output
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_and_analyze_ct():
    """Load and analyze individual CT slices"""
    print("🔍 === Analyzing CT Data Quality ===")
    
    try:
        import SimpleITK as sitk
    except ImportError:
        print("Installing SimpleITK...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "SimpleITK"])
        import SimpleITK as sitk
    
    # Find DICOM directory
    download_dirs = glob.glob("data/tciaDownload/*")
    dicom_dirs = [d for d in download_dirs if os.path.isdir(d)]
    
    if not dicom_dirs:
        print("❌ No DICOM directories found")
        return None, None
    
    dicom_dir = dicom_dirs[0]
    print(f"📁 Analyzing: {os.path.basename(dicom_dir)}")
    
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Convert to numpy
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    
    print(f"📐 Volume shape: {volume.shape}")
    print(f"📏 Spacing: {spacing} mm")
    print(f"🎯 HU range: {volume.min():.0f} to {volume.max():.0f}")
    
    # Analyze individual slices
    print(f"\n🔬 === Analyzing Individual Slices ===")
    
    # Show sample slices from different regions
    num_slices = volume.shape[0]
    sample_indices = [
        num_slices // 4,      # Upper chest
        num_slices // 2,      # Mid chest  
        3 * num_slices // 4   # Lower chest
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, slice_idx in enumerate(sample_indices):
        slice_data = volume[slice_idx]
        
        print(f"Slice {slice_idx}: HU range {slice_data.min():.0f} to {slice_data.max():.0f}")
        
        # Original slice
        axes[0, i].imshow(slice_data, cmap='gray', vmin=-1000, vmax=1000)
        axes[0, i].set_title(f'CT Slice {slice_idx}\n(Raw HU values)')
        axes[0, i].axis('off')
        
        # Window/Level adjusted for chest
        axes[1, i].imshow(slice_data, cmap='gray', vmin=-600, vmax=1500)  # Chest window
        axes[1, i].set_title(f'CT Slice {slice_idx}\n(Chest Window: -600 to 1500 HU)')
        axes[1, i].axis('off')
    
    plt.suptitle('CT Slice Analysis - Different Anatomical Levels', fontsize=16)
    plt.tight_layout()
    plt.savefig("outputs/analysis/ct_slice_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ CT slice analysis saved as outputs/analysis/ct_slice_analysis.png")
    
    return volume, spacing

def compare_projection_methods(volume, spacing):
    """Compare different projection methods"""
    print(f"\n⚗️ === Comparing Projection Methods ===")
    
    # Method 1: Simple summation (current approach)
    print("Method 1: Simple summation...")
    drr_sum_ap = np.sum(volume, axis=1) * spacing[1]
    drr_sum_lateral = np.sum(volume, axis=2) * spacing[0]
    
    # Method 2: Maximum Intensity Projection (MIP)
    print("Method 2: Maximum Intensity Projection...")
    drr_mip_ap = np.max(volume, axis=1)
    drr_mip_lateral = np.max(volume, axis=2)
    
    # Method 3: Weighted summation with better attenuation
    print("Method 3: Improved attenuation model...")
    
    # Better HU to attenuation conversion
    volume_windowed = np.clip(volume, -1000, 3000)
    
    # More realistic attenuation coefficients at 120 kVp
    mu_air = 0.0
    mu_water = 0.019  # mm^-1
    mu_bone = 0.048   # mm^-1
    
    # Piecewise linear attenuation model
    mu_volume = np.zeros_like(volume_windowed, dtype=np.float32)
    
    # Air/lung: HU < -500
    air_mask = volume_windowed < -500
    mu_volume[air_mask] = mu_air + (mu_water - mu_air) * (volume_windowed[air_mask] + 1000) / 500
    
    # Soft tissue: -500 <= HU < 100  
    soft_mask = (volume_windowed >= -500) & (volume_windowed < 100)
    mu_volume[soft_mask] = mu_water * (1 + volume_windowed[soft_mask] / 1000)
    
    # Bone: HU >= 100
    bone_mask = volume_windowed >= 100
    mu_volume[bone_mask] = mu_water + (mu_bone - mu_water) * np.minimum(volume_windowed[bone_mask] / 1000, 3.0)
    
    # Ensure non-negative
    mu_volume = np.maximum(mu_volume, 0)
    
    drr_improved_ap = np.sum(mu_volume, axis=1) * spacing[1]
    drr_improved_lateral = np.sum(mu_volume, axis=2) * spacing[0]
    
    print(f"📊 Projection ranges:")
    print(f"  Sum AP: {drr_sum_ap.min():.3f} to {drr_sum_ap.max():.3f}")
    print(f"  MIP AP: {drr_mip_ap.min():.0f} to {drr_mip_ap.max():.0f}")
    print(f"  Improved AP: {drr_improved_ap.min():.3f} to {drr_improved_ap.max():.3f}")
    
    return {
        'sum_ap': drr_sum_ap,
        'sum_lateral': drr_sum_lateral,
        'mip_ap': drr_mip_ap, 
        'mip_lateral': drr_mip_lateral,
        'improved_ap': drr_improved_ap,
        'improved_lateral': drr_improved_lateral,
        'mu_volume': mu_volume
    }

def apply_better_contrast_enhancement(drr_data, method_name):
    """Apply better contrast enhancement for X-ray appearance"""
    
    if 'mip' in method_name.lower():
        # For MIP, use different enhancement
        # Normalize to 0-1
        drr_norm = (drr_data - drr_data.min()) / (drr_data.max() - drr_data.min())
        # Apply power law for contrast
        enhanced = np.power(drr_norm, 0.5)  # Gamma correction
        # Invert for X-ray appearance (bright = less attenuation)
        enhanced = 1.0 - enhanced
    else:
        # For attenuation-based projections, use Beer-Lambert law
        epsilon = 1e-6
        intensity = np.exp(-drr_data)
        # Apply log for display
        enhanced = -np.log(intensity + epsilon)
        # Normalize
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())
    
    # Convert to 8-bit
    enhanced_8bit = (255 * enhanced).astype(np.uint8)
    
    return enhanced_8bit

def create_comparison_figure(projections):
    """Create comparison figure of different projection methods"""
    print(f"🖼️ Creating projection method comparison...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    
    methods = [
        ('sum_ap', 'sum_lateral', 'Simple Summation'),
        ('mip_ap', 'mip_lateral', 'Maximum Intensity Projection'), 
        ('improved_ap', 'improved_lateral', 'Improved Attenuation Model')
    ]
    
    for i, (ap_key, lat_key, title) in enumerate(methods):
        # Process AP projection
        ap_enhanced = apply_better_contrast_enhancement(projections[ap_key], ap_key)
        
        # Process Lateral projection  
        lat_enhanced = apply_better_contrast_enhancement(projections[lat_key], lat_key)
        
        # Display AP
        axes[i, 0].imshow(np.flipud(ap_enhanced), cmap='gray', aspect='auto')
        axes[i, 0].set_title(f'{title}\nAnterior-Posterior View', fontsize=12)
        axes[i, 0].axis('off')
        
        # Add orientation labels for AP
        axes[i, 0].text(0.02, 0.98, 'R', transform=axes[i, 0].transAxes, 
                       fontsize=14, color='white', weight='bold', ha='left', va='top')
        axes[i, 0].text(0.98, 0.98, 'L', transform=axes[i, 0].transAxes,
                       fontsize=14, color='white', weight='bold', ha='right', va='top')
        
        # Display Lateral
        axes[i, 1].imshow(np.flipud(lat_enhanced), cmap='gray', aspect='auto')
        axes[i, 1].set_title(f'{title}\nLateral View', fontsize=12)
        axes[i, 1].axis('off')
        
        # Add orientation labels for Lateral
        axes[i, 1].text(0.02, 0.98, 'A', transform=axes[i, 1].transAxes,
                       fontsize=14, color='white', weight='bold', ha='left', va='top')
        axes[i, 1].text(0.98, 0.98, 'P', transform=axes[i, 1].transAxes,
                       fontsize=14, color='white', weight='bold', ha='right', va='top')
    
    plt.suptitle('DRR Method Comparison - Chest CT (LUNG1-001)', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig("outputs/analysis/drr_method_comparison.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print("✅ Method comparison saved as outputs/analysis/drr_method_comparison.png")
    
    return methods

def save_best_drr(projections):
    """Save the best DRR with improved quality"""
    print(f"\n💾 === Saving Improved DRR ===")
    
    # Use the improved attenuation model
    ap_enhanced = apply_better_contrast_enhancement(projections['improved_ap'], 'improved_ap')
    lateral_enhanced = apply_better_contrast_enhancement(projections['improved_lateral'], 'improved_lateral')
    
    # Save individual views
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(np.flipud(ap_enhanced), cmap='gray', aspect='auto')
    ax.set_title('Improved DRR - Anterior-Posterior View\nChest CT (LUNG1-001, NSCLC-Radiomics)', fontsize=14, pad=20)
    ax.axis('off')
    ax.text(0.02, 0.98, 'R', transform=ax.transAxes, fontsize=16, 
           color='white', weight='bold', ha='left', va='top')
    ax.text(0.98, 0.98, 'L', transform=ax.transAxes, fontsize=16, 
           color='white', weight='bold', ha='right', va='top')
    plt.tight_layout()
    plt.savefig("outputs/iterations/drr_improved_ap.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(np.flipud(lateral_enhanced), cmap='gray', aspect='auto')
    ax.set_title('Improved DRR - Lateral View\nChest CT (LUNG1-001, NSCLC-Radiomics)', fontsize=14, pad=20)
    ax.axis('off')
    ax.text(0.02, 0.98, 'A', transform=ax.transAxes, fontsize=16,
           color='white', weight='bold', ha='left', va='top')
    ax.text(0.98, 0.98, 'P', transform=ax.transAxes, fontsize=16,
           color='white', weight='bold', ha='right', va='top')
    plt.tight_layout()
    plt.savefig("outputs/iterations/drr_improved_lateral.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # Combined view
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax1.imshow(np.flipud(ap_enhanced), cmap='gray', aspect='auto')
    ax1.set_title('Anterior-Posterior DRR', fontsize=16, pad=20)
    ax1.axis('off')
    ax1.text(0.02, 0.98, 'R', transform=ax1.transAxes, fontsize=16,
            color='white', weight='bold', ha='left', va='top')
    ax1.text(0.98, 0.98, 'L', transform=ax1.transAxes, fontsize=16,
            color='white', weight='bold', ha='right', va='top')
    
    ax2.imshow(np.flipud(lateral_enhanced), cmap='gray', aspect='auto')
    ax2.set_title('Lateral DRR', fontsize=16, pad=20)
    ax2.axis('off')
    ax2.text(0.02, 0.98, 'A', transform=ax2.transAxes, fontsize=16,
            color='white', weight='bold', ha='left', va='top')
    ax2.text(0.98, 0.98, 'P', transform=ax2.transAxes, fontsize=16,
            color='white', weight='bold', ha='right', va='top')
    
    plt.suptitle('Improved DRR - Chest CT (LUNG1-001, NSCLC-Radiomics)', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.savefig("outputs/iterations/drr_improved_combined.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print("✅ Improved DRR images saved to outputs/iterations/:")
    print("   • drr_improved_ap.png")
    print("   • drr_improved_lateral.png") 
    print("   • drr_improved_combined.png")

def main():
    """Main analysis function"""
    print("🔬 === CT Data Quality Analysis & DRR Improvement ===\n")
    
    # Load and analyze CT data
    volume, spacing = load_and_analyze_ct()
    if volume is None:
        return
    
    # Compare different projection methods
    projections = compare_projection_methods(volume, spacing)
    
    # Create comparison figure
    create_comparison_figure(projections)
    
    # Save improved DRR
    save_best_drr(projections)
    
    print(f"\n🎉 === Analysis Complete ===")
    print("Generated files:")
    print("📊 outputs/analysis/ct_slice_analysis.png - Individual CT slice quality")
    print("🔬 outputs/analysis/drr_method_comparison.png - Different DRR methods compared")
    print("✨ outputs/iterations/drr_improved_*.png - High-quality DRR outputs")
    print("\nThe improved DRR should look much more like a real chest X-ray!")

if __name__ == "__main__":
    main() 