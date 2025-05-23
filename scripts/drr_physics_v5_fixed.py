#!/usr/bin/env python3
"""
Fixed Physics-Correct DRR Generator V5
Fixes the coordinate system bugs in V4 that caused black image output.
Key fixes:
- Corrected ray-volume intersection calculations
- Fixed coordinate system alignment between SimpleITK and NumPy
- Proper source-detector-volume geometry
- Enhanced sampling and interpolation
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import os
from scipy.interpolate import RegularGridInterpolator


class FixedDRRGenerator:
    """Fixed physically accurate DRR generator with corrected coordinate systems."""
    
    def __init__(
        self,
        source_to_detector_distance: float = 1000.0,  # mm
        source_to_patient_distance: float = 600.0,    # mm
        detector_size: Tuple[float, float] = (430.0, 430.0),  # mm
        detector_resolution: Tuple[int, int] = (1024, 1024),    # Enhanced resolution
        photon_energy: float = 120.0  # keV
    ):
        self.sdd = source_to_detector_distance
        self.spd = source_to_patient_distance
        self.detector_size = detector_size
        self.detector_resolution = detector_resolution
        self.photon_energy = photon_energy
        
        # Calculate pixel spacing on detector
        self.pixel_spacing = (
            detector_size[0] / detector_resolution[0],
            detector_size[1] / detector_resolution[1]
        )
        
        print(f"Fixed DRR Generator initialized:")
        print(f"  Source-Detector Distance: {self.sdd} mm")
        print(f"  Detector size: {detector_size[0]}x{detector_size[1]} mm")
        print(f"  Detector resolution: {detector_resolution[0]}x{detector_resolution[1]} pixels")
        print(f"  Pixel spacing: {self.pixel_spacing[0]:.3f}x{self.pixel_spacing[1]:.3f} mm")
        
    def hu_to_attenuation(self, hu_values: np.ndarray) -> np.ndarray:
        """Convert HU to attenuation with enhanced tissue differentiation."""
        mu_water = 0.019  # mm^-1 at 120 keV
        
        # Enhanced tissue-specific conversion
        mu = np.zeros_like(hu_values, dtype=np.float32)
        
        # Air/lung
        air_mask = hu_values < -500
        mu[air_mask] = 0.0001
        
        # Soft tissue
        soft_mask = (hu_values >= -500) & (hu_values < 200)
        mu[soft_mask] = mu_water * (1.0 + hu_values[soft_mask] / 1000.0)
        
        # Bone - enhanced
        bone_mask = hu_values >= 200
        mu[bone_mask] = mu_water * (2.0 + hu_values[bone_mask] / 500.0)
        
        # Ensure non-negative
        mu = np.maximum(mu, 0.0001)
        
        return mu
    
    def setup_projection_geometry(self, projection_angle: float, volume_origin: np.ndarray, volume_size: np.ndarray):
        """
        Set up corrected projection geometry.
        
        Args:
            projection_angle: 0 for AP, 90 for lateral (degrees)
            volume_origin: Physical origin of volume in mm
            volume_size: Physical size of volume in mm
            
        Returns:
            source_position, detector_center, detector_u, detector_v
        """
        angle_rad = np.radians(projection_angle)
        
        # Volume center in world coordinates
        volume_center = volume_origin + volume_size / 2
        
        # Source position (rotate around volume center)
        if projection_angle == 0:  # AP view
            source_pos = volume_center + np.array([0, -self.spd, 0])
            detector_center = volume_center + np.array([0, self.sdd - self.spd, 0])
            detector_u = np.array([1, 0, 0])  # X direction
            detector_v = np.array([0, 0, 1])  # Z direction (up)
        elif projection_angle == 90:  # Lateral view
            source_pos = volume_center + np.array([-self.spd, 0, 0])
            detector_center = volume_center + np.array([self.sdd - self.spd, 0, 0])
            detector_u = np.array([0, 1, 0])  # Y direction
            detector_v = np.array([0, 0, 1])  # Z direction (up)
        else:
            # General angle
            source_pos = volume_center + np.array([
                -self.spd * np.sin(angle_rad),
                -self.spd * np.cos(angle_rad),
                0
            ])
            detector_center = volume_center + np.array([
                (self.sdd - self.spd) * np.sin(angle_rad),
                (self.sdd - self.spd) * np.cos(angle_rad),
                0
            ])
            detector_u = np.array([np.cos(angle_rad), -np.sin(angle_rad), 0])
            detector_v = np.array([0, 0, 1])
        
        return source_pos, detector_center, detector_u, detector_v
    
    def compute_ray_volume_intersection(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        volume_origin: np.ndarray,
        volume_size: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Compute intersection of ray with volume bounding box.
        
        Returns:
            t_near, t_far, intersects
        """
        # Volume bounds
        vol_min = volume_origin
        vol_max = volume_origin + volume_size
        
        # Ray-box intersection using slab method
        t_near = -np.inf
        t_far = np.inf
        
        for i in range(3):
            if abs(ray_direction[i]) < 1e-8:
                # Ray is parallel to slab
                if ray_origin[i] < vol_min[i] or ray_origin[i] > vol_max[i]:
                    return 0.0, 0.0, False
            else:
                # Compute intersection distances
                t1 = (vol_min[i] - ray_origin[i]) / ray_direction[i]
                t2 = (vol_max[i] - ray_origin[i]) / ray_direction[i]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                
                t_near = max(t_near, t1)
                t_far = min(t_far, t2)
                
                if t_near > t_far:
                    return 0.0, 0.0, False
        
        # Ensure we start from a positive t
        t_near = max(t_near, 0.0)
        
        intersects = t_near < t_far
        return t_near, t_far, intersects
    
    def cast_ray(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        mu_volume: np.ndarray,
        volume_origin: np.ndarray,
        volume_spacing: np.ndarray,
        num_samples: int = 512
    ) -> float:
        """
        Cast a ray through the volume and compute line integral.
        
        Returns:
            Line integral of attenuation coefficients
        """
        volume_size = np.array(mu_volume.shape[::-1]) * volume_spacing  # Convert to (x,y,z)
        
        # Find ray-volume intersection
        t_near, t_far, intersects = self.compute_ray_volume_intersection(
            ray_origin, ray_direction, volume_origin, volume_size
        )
        
        if not intersects or t_near >= t_far:
            return 0.0
        
        # Sample points along ray
        t_values = np.linspace(t_near, t_far, num_samples)
        sample_points = ray_origin[np.newaxis, :] + t_values[:, np.newaxis] * ray_direction[np.newaxis, :]
        
        # Convert to voxel coordinates (accounting for SimpleITK coordinate order)
        voxel_coords = (sample_points - volume_origin) / volume_spacing
        
        # Reorder for numpy indexing (z, y, x)
        voxel_coords_numpy = voxel_coords[:, [2, 1, 0]]
        
        # Create interpolator
        z_coords = np.arange(mu_volume.shape[0])
        y_coords = np.arange(mu_volume.shape[1])
        x_coords = np.arange(mu_volume.shape[2])
        
        interpolator = RegularGridInterpolator(
            (z_coords, y_coords, x_coords),
            mu_volume,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Sample attenuation values
        mu_values = interpolator(voxel_coords_numpy)
        
        # Compute line integral using trapezoidal rule
        if len(mu_values) < 2:
            return 0.0
        
        dt = (t_far - t_near) / (num_samples - 1)
        line_integral = np.trapezoid(mu_values, dx=dt)
        
        return line_integral
    
    def generate_drr(
        self,
        ct_volume: sitk.Image,
        projection_angle: float = 0.0,
        num_samples: int = 512
    ) -> np.ndarray:
        """
        Generate DRR with fixed coordinate system.
        
        Args:
            ct_volume: SimpleITK image
            projection_angle: Projection angle in degrees
            num_samples: Samples per ray
            
        Returns:
            DRR image as numpy array
        """
        print(f"Generating fixed DRR at {projection_angle}° angle...")
        
        # Get volume data and metadata
        volume_array = sitk.GetArrayFromImage(ct_volume)  # (z, y, x)
        spacing = np.array(ct_volume.GetSpacing())  # (x, y, z)
        origin = np.array(ct_volume.GetOrigin())    # (x, y, z)
        
        print(f"Volume shape (z,y,x): {volume_array.shape}")
        print(f"Spacing (x,y,z): {spacing}")
        print(f"Origin (x,y,z): {origin}")
        
        # Convert HU to attenuation
        mu_volume = self.hu_to_attenuation(volume_array)
        print(f"Attenuation range: [{mu_volume.min():.5f}, {mu_volume.max():.5f}] mm^-1")
        
        # Volume size in physical coordinates
        volume_size = np.array(volume_array.shape[::-1]) * spacing  # (x,y,z)
        
        # Set up projection geometry
        source_pos, detector_center, detector_u, detector_v = self.setup_projection_geometry(
            projection_angle, origin, volume_size
        )
        
        print(f"Source position: {source_pos}")
        print(f"Detector center: {detector_center}")
        
        # Initialize DRR
        drr = np.zeros(self.detector_resolution)
        
        # Generate detector pixel positions
        u_coords = np.linspace(-self.detector_size[0]/2, self.detector_size[0]/2, self.detector_resolution[0])
        v_coords = np.linspace(-self.detector_size[1]/2, self.detector_size[1]/2, self.detector_resolution[1])
        
        total_rays = self.detector_resolution[0] * self.detector_resolution[1]
        ray_count = 0
        
        print("Casting rays through volume...")
        
        for i, u in enumerate(u_coords):
            for j, v in enumerate(v_coords):
                # Detector pixel position
                pixel_pos = detector_center + u * detector_u + v * detector_v
                
                # Ray direction
                ray_dir = pixel_pos - source_pos
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
                
                # Cast ray
                line_integral = self.cast_ray(
                    source_pos, ray_dir, mu_volume, origin, spacing, num_samples
                )
                
                # Apply Beer-Lambert law
                drr[j, i] = np.exp(-line_integral)
                
                ray_count += 1
                if ray_count % 50000 == 0:
                    print(f"Progress: {ray_count}/{total_rays} rays ({100*ray_count/total_rays:.1f}%)")
        
        # Convert to X-ray appearance
        drr = 1.0 - drr  # Invert for radiographic appearance
        drr = np.power(drr, 0.6)  # Gamma correction
        
        print(f"DRR generation complete. Range: [{drr.min():.3f}, {drr.max():.3f}]")
        
        return drr


def main():
    """Generate fixed DRRs."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/physics_v5_fixed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CT series
    series_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not series_dirs:
        print("No CT series found!")
        return
    
    # Use first series for testing
    series_dir = series_dirs[0]
    print(f"Processing series: {series_dir.name}")
    
    # Load DICOM series
    print("Loading DICOM series...")
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(str(series_dir))
    
    if not dicom_files:
        print("No DICOM files found!")
        return
    
    reader.SetFileNames(dicom_files)
    ct_volume = reader.Execute()
    
    # Print volume information
    print(f"Volume size: {ct_volume.GetSize()}")
    print(f"Volume spacing: {ct_volume.GetSpacing()} mm")
    print(f"Volume origin: {ct_volume.GetOrigin()} mm")
    
    # Create fixed DRR generator
    generator = FixedDRRGenerator(
        source_to_detector_distance=1000.0,
        source_to_patient_distance=600.0,
        detector_size=(430.0, 430.0),
        detector_resolution=(1024, 1024),  # Enhanced resolution
    )
    
    # Generate AP view
    print("\nGenerating fixed AP view...")
    drr_ap = generator.generate_drr(ct_volume, projection_angle=0.0, num_samples=256)
    
    # Save AP view
    plt.figure(figsize=(12, 12))
    plt.imshow(drr_ap, cmap='gray')
    plt.title('Fixed DRR - Anterior-Posterior View (V5)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_physics_v5_fixed_ap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate lateral view
    print("\nGenerating fixed lateral view...")
    drr_lateral = generator.generate_drr(ct_volume, projection_angle=90.0, num_samples=256)
    
    # Save lateral view
    plt.figure(figsize=(12, 12))
    plt.imshow(drr_lateral, cmap='gray')
    plt.title('Fixed DRR - Lateral View (V5)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_physics_v5_fixed_lateral.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    axes[0].imshow(drr_ap, cmap='gray')
    axes[0].set_title('Fixed AP View', fontsize=18)
    axes[0].axis('off')
    
    axes[1].imshow(drr_lateral, cmap='gray')
    axes[1].set_title('Fixed Lateral View', fontsize=18)
    axes[1].axis('off')
    
    plt.suptitle('Fixed Physics-Correct DRR Generation (V5)', fontsize=22)
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_physics_v5_fixed_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFixed DRRs saved to {output_dir}")
    print("Key fixes applied:")
    print("  ✓ Corrected ray-volume intersection")
    print("  ✓ Fixed coordinate system alignment")
    print("  ✓ Proper source-detector geometry")
    print("  ✓ Enhanced resolution and sampling")


if __name__ == "__main__":
    main()