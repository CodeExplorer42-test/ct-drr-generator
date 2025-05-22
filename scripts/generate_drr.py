#!/usr/bin/env python3
"""
DRR Generator for Downloaded TCIA Chest CT Data

This script loads the downloaded CT volume, performs forward projection 
to generate a Digitally Reconstructed Radiograph (DRR), and saves it as an image.
"""

import os
import glob
import subprocess
import sys

def install_requirements():
    """Install required packages for DRR generation"""
    packages = ['SimpleITK', 'matplotlib', 'pillow', 'numpy']
    
    for package in packages:
        try:
            if package == 'SimpleITK':
                import SimpleITK as sitk
            elif package == 'matplotlib':
                import matplotlib.pyplot as plt
            elif package == 'pillow':
                from PIL import Image
            elif package == 'numpy':
                import numpy as np
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_ct_volume(dicom_dir):
    """Load CT volume from DICOM directory"""
    try:
        import SimpleITK as sitk
        print(f"📁 Loading CT volume from: {os.path.basename(dicom_dir)}")
        
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        
        if not dicom_names:
            print("❌ No DICOM files found")
            return None, None
            
        print(f"📚 Found {len(dicom_names)} DICOM slices")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Convert to numpy array
        volume = sitk.GetArrayFromImage(image)
        
        # Get spacing and origin information
        spacing = image.GetSpacing()  # (x, y, z) spacing in mm
        origin = image.GetOrigin()    # (x, y, z) origin in mm
        direction = image.GetDirection()
        
        print(f"📐 Volume shape: {volume.shape}")
        print(f"📏 Voxel spacing: {spacing} mm")
        print(f"📍 Origin: {origin} mm")
        
        # Volume is typically (z, y, x) - we want (z, y, x) for DRR
        print(f"🎯 Volume range: {volume.min():.0f} to {volume.max():.0f} HU")
        
        metadata = {
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'shape': volume.shape
        }
        
        return volume, metadata
        
    except Exception as e:
        print(f"❌ Error loading CT volume: {e}")
        return None, None

def preprocess_volume(volume):
    """Preprocess CT volume for DRR generation"""
    import numpy as np
    print("🔧 Preprocessing volume for DRR...")
    
    # Convert HU to linear attenuation coefficients
    # Typical conversion: μ = μ_water * (1 + HU/1000)
    # For simplicity, we'll use a basic conversion
    
    # Clip extreme values
    volume_clipped = np.clip(volume, -1000, 3000)
    
    # Convert HU to approximate linear attenuation coefficients
    # Water at 120 kVp has μ ≈ 0.19 cm^-1
    mu_water = 0.019  # mm^-1 (converted from cm^-1)
    mu_volume = mu_water * (1 + volume_clipped / 1000.0)
    
    # Ensure non-negative values
    mu_volume = np.maximum(mu_volume, 0)
    
    print(f"📊 Attenuation range: {mu_volume.min():.4f} to {mu_volume.max():.4f} mm^-1")
    
    return mu_volume

def generate_drr_anterior_posterior(volume, metadata, source_distance=1040):
    """Generate DRR with Anterior-Posterior projection"""
    import numpy as np
    print("🎯 Generating AP (Anterior-Posterior) DRR...")
    
    # For AP projection, we sum along the anterior-posterior axis (typically Y-axis)
    # The volume is typically oriented as (z, y, x) where:
    # z = superior-inferior (head-foot)
    # y = anterior-posterior (front-back)  
    # x = left-right
    
    spacing = metadata['spacing']  # (x, y, z) spacing
    z_spacing = spacing[2]  # slice thickness
    
    # Sum along Y-axis (anterior-posterior direction) for AP view
    drr_ap = np.sum(volume, axis=1) * spacing[1]  # multiply by spacing for correct scaling
    
    print(f"📐 DRR AP shape: {drr_ap.shape}")
    print(f"📊 DRR AP range: {drr_ap.min():.4f} to {drr_ap.max():.4f}")
    
    return drr_ap

def generate_drr_lateral(volume, metadata):
    """Generate DRR with Lateral projection"""
    import numpy as np
    print("🎯 Generating Lateral DRR...")
    
    spacing = metadata['spacing']
    
    # Sum along X-axis (left-right direction) for lateral view
    drr_lateral = np.sum(volume, axis=2) * spacing[0]
    
    print(f"📐 DRR Lateral shape: {drr_lateral.shape}")
    print(f"📊 DRR Lateral range: {drr_lateral.min():.4f} to {drr_lateral.max():.4f}")
    
    return drr_lateral

def apply_xray_physics(drr_projection):
    """Apply X-ray physics to convert attenuation to intensity"""
    import numpy as np
    print("⚗️ Applying X-ray physics (Beer-Lambert law)...")
    
    # Beer-Lambert law: I = I0 * exp(-μt)
    # For DRR: intensity = exp(-integrated_attenuation)
    I0 = 1.0  # Incident intensity
    intensity = I0 * np.exp(-drr_projection)
    
    print(f"💡 Intensity range: {intensity.min():.4f} to {intensity.max():.4f}")
    
    return intensity

def enhance_drr_contrast(intensity):
    """Enhance DRR contrast for better visualization"""
    import numpy as np
    print("🎨 Enhancing DRR contrast...")
    
    # Apply log transform to enhance contrast (like real X-ray imaging)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-6
    log_intensity = -np.log(intensity + epsilon)
    
    # Normalize to 0-255 range
    log_min, log_max = log_intensity.min(), log_intensity.max()
    enhanced = 255 * (log_intensity - log_min) / (log_max - log_min)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    print(f"✨ Enhanced DRR range: {enhanced.min()} to {enhanced.max()}")
    
    return enhanced

def save_drr_image(drr_array, filename, title="DRR"):
    """Save DRR as PNG/JPEG with proper orientation"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server/headless environments
    import matplotlib.pyplot as plt
    print(f"💾 Saving DRR as {filename}...")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Display DRR (flip vertically for correct anatomical orientation)
    plt.imshow(np.flipud(drr_array), cmap='gray', aspect='auto')
    plt.title(f'{title} - Chest CT DRR\nPatient: LUNG1-001 (NSCLC-Radiomics)', fontsize=14, pad=20)
    plt.axis('off')
    
    # Add anatomical orientation labels
    if 'AP' in title or 'Anterior' in title:
        plt.text(0.02, 0.98, 'R', transform=plt.gca().transAxes, fontsize=16, 
                color='white', weight='bold', ha='left', va='top')
        plt.text(0.98, 0.98, 'L', transform=plt.gca().transAxes, fontsize=16, 
                color='white', weight='bold', ha='right', va='top')
        plt.text(0.98, 0.02, 'Head', transform=plt.gca().transAxes, fontsize=12, 
                color='white', weight='bold', ha='right', va='bottom')
        plt.text(0.02, 0.02, 'Feet', transform=plt.gca().transAxes, fontsize=12, 
                color='white', weight='bold', ha='left', va='bottom')
    else:
        plt.text(0.02, 0.98, 'A', transform=plt.gca().transAxes, fontsize=16, 
                color='white', weight='bold', ha='left', va='top')
        plt.text(0.98, 0.98, 'P', transform=plt.gca().transAxes, fontsize=16, 
                color='white', weight='bold', ha='right', va='top')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"✅ DRR saved as {filename}")

def main():
    """Main DRR generation function"""
    print("🏥 === DRR Generator for TCIA Chest CT ===\n")
    
    # Install required packages
    install_requirements()
    
    # Find downloaded DICOM directory
    download_dirs = glob.glob("tciaDownload/*")
    dicom_dirs = [d for d in download_dirs if os.path.isdir(d)]
    
    if not dicom_dirs:
        print("❌ No DICOM directories found. Please run the downloader first.")
        return
    
    dicom_dir = dicom_dirs[0]  # Use first directory
    print(f"🔍 Processing: {os.path.basename(dicom_dir)}\n")
    
    # Step 1: Load CT volume
    volume, metadata = load_ct_volume(dicom_dir)
    if volume is None:
        return
    
    # Step 2: Preprocess volume
    mu_volume = preprocess_volume(volume)
    
    # Step 3: Generate DRR projections
    print("\n🎯 === Generating DRR Projections ===")
    
    # Generate AP (Anterior-Posterior) DRR
    drr_ap_raw = generate_drr_anterior_posterior(mu_volume, metadata)
    drr_ap_intensity = apply_xray_physics(drr_ap_raw)
    drr_ap_enhanced = enhance_drr_contrast(drr_ap_intensity)
    
    # Generate Lateral DRR
    drr_lateral_raw = generate_drr_lateral(mu_volume, metadata)
    drr_lateral_intensity = apply_xray_physics(drr_lateral_raw)
    drr_lateral_enhanced = enhance_drr_contrast(drr_lateral_intensity)
    
    # Step 4: Save DRR images
    print("\n💾 === Saving DRR Images ===")
    
    save_drr_image(drr_ap_enhanced, "drr_anterior_posterior.png", 
                   "DRR - Anterior-Posterior View")
    save_drr_image(drr_lateral_enhanced, "drr_lateral.png", 
                   "DRR - Lateral View")
    
    # Also save as JPEG
    save_drr_image(drr_ap_enhanced, "drr_anterior_posterior.jpg", 
                   "DRR - Anterior-Posterior View")
    save_drr_image(drr_lateral_enhanced, "drr_lateral.jpg", 
                   "DRR - Lateral View")
    
    # Create a combined view
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server/headless environments
    import matplotlib.pyplot as plt
    print("\n🖼️ === Creating Combined View ===")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # AP view
    ax1.imshow(np.flipud(drr_ap_enhanced), cmap='gray', aspect='auto')
    ax1.set_title('Anterior-Posterior DRR', fontsize=16, pad=20)
    ax1.axis('off')
    ax1.text(0.02, 0.98, 'R', transform=ax1.transAxes, fontsize=16, 
            color='white', weight='bold', ha='left', va='top')
    ax1.text(0.98, 0.98, 'L', transform=ax1.transAxes, fontsize=16, 
            color='white', weight='bold', ha='right', va='top')
    
    # Lateral view
    ax2.imshow(np.flipud(drr_lateral_enhanced), cmap='gray', aspect='auto')
    ax2.set_title('Lateral DRR', fontsize=16, pad=20)
    ax2.axis('off')
    ax2.text(0.02, 0.98, 'A', transform=ax2.transAxes, fontsize=16, 
            color='white', weight='bold', ha='left', va='top')
    ax2.text(0.98, 0.98, 'P', transform=ax2.transAxes, fontsize=16, 
            color='white', weight='bold', ha='right', va='top')
    
    plt.suptitle('DRR - Chest CT (LUNG1-001, NSCLC-Radiomics)', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.savefig("drr_combined.png", dpi=300, bbox_inches='tight', facecolor='black')
    plt.savefig("drr_combined.jpg", dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print("✅ Combined DRR view saved as drr_combined.png and drr_combined.jpg")
    
    # Summary
    print(f"\n🎉 === DRR Generation Complete ===")
    print(f"📊 Generated DRRs from {volume.shape[0]} CT slices")
    print(f"📁 Output files:")
    print(f"   • drr_anterior_posterior.png/jpg - AP projection")
    print(f"   • drr_lateral.png/jpg - Lateral projection") 
    print(f"   • drr_combined.png/jpg - Both views together")
    print(f"\n🏥 Ready for clinical/research use!")

if __name__ == "__main__":
    main() 