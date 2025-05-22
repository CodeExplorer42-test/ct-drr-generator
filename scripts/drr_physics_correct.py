#!/usr/bin/env python3
"""
Physically correct DRR generation using proper ray-casting and Beer-Lambert law.
This implementation uses cone-beam geometry and proper line integral calculation.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import os
from scipy.interpolate import RegularGridInterpolator


class DRRGenerator:
    """Generates physically accurate DRRs using ray-casting and proper X-ray physics."""
    
    def __init__(
        self,
        source_to_detector_distance: float = 1000.0,  # mm
        source_to_patient_distance: float = 600.0,    # mm
        detector_size: Tuple[float, float] = (430.0, 430.0),  # mm
        detector_resolution: Tuple[int, int] = (512, 512),    # pixels
        photon_energy: float = 120.0  # keV (typical chest X-ray)
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
        
    def hu_to_attenuation(self, hu_values: np.ndarray) -> np.ndarray:
        """
        Convert Hounsfield Units to linear attenuation coefficients.
        Uses proper tissue-specific conversion for 120 keV X-rays.
        """
        # Water attenuation at 120 keV (approximately)
        mu_water = 0.019  # mm^-1
        
        # Convert HU to linear attenuation coefficient
        # HU = 1000 * (mu - mu_water) / mu_water
        # mu = mu_water * (1 + HU/1000)
        mu = mu_water * (1.0 + hu_values / 1000.0)
        
        # Clamp negative values (air) to near zero
        mu = np.maximum(mu, 0.0001)
        
        return mu
    
    def create_ray_directions(self, projection_angle: float) -> np.ndarray:
        """
        Create ray directions for cone-beam geometry.
        
        Args:
            projection_angle: 0 for AP view, 90 for lateral view (degrees)
        
        Returns:
            Array of ray directions (detector_pixels_y, detector_pixels_x, 3)
        """
        angle_rad = np.radians(projection_angle)
        
        # Create detector grid
        y_pixels, x_pixels = self.detector_resolution
        
        # Detector coordinates (centered at origin)
        x_coords = np.linspace(
            -self.detector_size[0]/2,
            self.detector_size[0]/2,
            x_pixels
        )
        y_coords = np.linspace(
            -self.detector_size[1]/2,
            self.detector_size[1]/2,
            y_pixels
        )
        
        # Create meshgrid
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
        
        # Source position (rotate around patient)
        source_x = -self.sdd * np.sin(angle_rad)
        source_y = 0
        source_z = -self.sdd * np.cos(angle_rad)
        source_pos = np.array([source_x, source_y, source_z])
        
        # Detector positions (opposite to source)
        detector_x = X * np.cos(angle_rad)
        detector_y = Y
        detector_z = X * np.sin(angle_rad)
        
        # Ray directions (from source to each detector pixel)
        ray_dirs = np.zeros((y_pixels, x_pixels, 3))
        
        for i in range(y_pixels):
            for j in range(x_pixels):
                detector_pos = np.array([
                    detector_x[i, j],
                    detector_y[i, j],
                    detector_z[i, j]
                ])
                ray_dir = detector_pos - source_pos
                ray_dirs[i, j] = ray_dir / np.linalg.norm(ray_dir)
        
        return ray_dirs, source_pos
    
    def ray_cast_volume(
        self,
        volume: np.ndarray,
        spacing: np.ndarray,
        origin: np.ndarray,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        num_samples: int = 512
    ) -> float:
        """
        Cast a single ray through the volume and compute line integral.
        
        Returns:
            Line integral of attenuation coefficients along the ray
        """
        # Volume dimensions
        vol_shape = volume.shape
        vol_physical_size = vol_shape * spacing
        
        # Find ray-volume intersection points
        # Using simple AABB intersection
        t_min = 0.0
        t_max = 1000.0  # Large value
        
        for i in range(3):
            if abs(ray_direction[i]) > 1e-6:
                t1 = (origin[i] - ray_origin[i]) / ray_direction[i]
                t2 = (origin[i] + vol_physical_size[i] - ray_origin[i]) / ray_direction[i]
                
                t_enter = min(t1, t2)
                t_exit = max(t1, t2)
                
                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)
        
        if t_min >= t_max:
            return 0.0  # Ray doesn't intersect volume
        
        # Sample points along ray
        t_values = np.linspace(t_min, t_max, num_samples)
        sample_points = ray_origin[np.newaxis, :] + t_values[:, np.newaxis] * ray_direction[np.newaxis, :]
        
        # Convert physical coordinates to voxel indices
        voxel_coords = (sample_points - origin) / spacing
        
        # Create interpolator for the volume
        x_coords = np.arange(vol_shape[0])
        y_coords = np.arange(vol_shape[1])
        z_coords = np.arange(vol_shape[2])
        
        interpolator = RegularGridInterpolator(
            (x_coords, y_coords, z_coords),
            volume,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Sample attenuation values
        attenuation_values = interpolator(voxel_coords)
        
        # Compute line integral using trapezoidal rule
        dt = (t_max - t_min) / (num_samples - 1)
        line_integral = np.trapezoid(attenuation_values, dx=dt)
        
        return line_integral
    
    def generate_drr(
        self,
        ct_volume: sitk.Image,
        projection_angle: float = 0.0,
        num_samples: int = 512
    ) -> np.ndarray:
        """
        Generate a DRR from CT volume using proper ray-casting.
        
        Args:
            ct_volume: SimpleITK image containing CT data
            projection_angle: 0 for AP, 90 for lateral (degrees)
            num_samples: Number of samples along each ray
            
        Returns:
            DRR image as numpy array
        """
        # Get volume data and metadata
        volume_array = sitk.GetArrayFromImage(ct_volume)
        spacing = np.array(ct_volume.GetSpacing())[::-1]  # Convert to numpy order (z,y,x)
        origin = np.array(ct_volume.GetOrigin())[::-1]
        
        # Convert HU to attenuation coefficients
        print("Converting HU to attenuation coefficients...")
        mu_volume = self.hu_to_attenuation(volume_array)
        
        # Create ray directions
        print(f"Creating ray geometry for {projection_angle}° projection...")
        ray_dirs, source_pos = self.create_ray_directions(projection_angle)
        
        # Initialize DRR image
        drr = np.zeros(self.detector_resolution)
        
        # Cast rays through volume
        print("Casting rays through volume...")
        total_rays = self.detector_resolution[0] * self.detector_resolution[1]
        ray_count = 0
        
        for i in range(self.detector_resolution[0]):
            for j in range(self.detector_resolution[1]):
                # Get ray direction
                ray_dir = ray_dirs[i, j]
                
                # Compute line integral
                line_integral = self.ray_cast_volume(
                    mu_volume,
                    spacing,
                    origin,
                    source_pos,
                    ray_dir,
                    num_samples
                )
                
                # Apply Beer-Lambert law
                # I = I0 * exp(-integral)
                drr[i, j] = np.exp(-line_integral)
                
                ray_count += 1
                if ray_count % 10000 == 0:
                    print(f"Progress: {ray_count}/{total_rays} rays ({100*ray_count/total_rays:.1f}%)")
        
        # Convert transmission to intensity (invert for X-ray appearance)
        drr = 1.0 - drr
        
        # Apply gamma correction for better visualization
        drr = np.power(drr, 0.5)
        
        return drr


def main():
    """Generate DRRs with correct physics implementation."""
    # Set up paths
    data_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/data/tciaDownload")
    output_dir = Path("/Users/sankaranarayanan/Downloads/ctdownloader/outputs/physics_correct")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CT series
    series_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not series_dirs:
        print("No CT series found in data directory!")
        return
    
    # Use first series
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
    
    # Create DRR generator
    generator = DRRGenerator(
        source_to_detector_distance=1000.0,
        source_to_patient_distance=600.0,
        detector_size=(430.0, 430.0),
        detector_resolution=(512, 512)
    )
    
    # Generate AP view
    print("\nGenerating AP (frontal) view...")
    drr_ap = generator.generate_drr(ct_volume, projection_angle=0.0, num_samples=256)
    
    # Save AP view
    plt.figure(figsize=(10, 10))
    plt.imshow(drr_ap, cmap='gray')
    plt.title('DRR - Anterior-Posterior View (Physics Correct)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_physics_correct_ap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate lateral view
    print("\nGenerating lateral view...")
    drr_lateral = generator.generate_drr(ct_volume, projection_angle=90.0, num_samples=256)
    
    # Save lateral view
    plt.figure(figsize=(10, 10))
    plt.imshow(drr_lateral, cmap='gray')
    plt.title('DRR - Lateral View (Physics Correct)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_physics_correct_lateral.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(drr_ap, cmap='gray')
    axes[0].set_title('Anterior-Posterior View', fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(drr_lateral, cmap='gray')
    axes[1].set_title('Lateral View', fontsize=16)
    axes[1].axis('off')
    
    plt.suptitle('Physically Correct DRR Generation', fontsize=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'drr_physics_correct_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDRRs saved to {output_dir}")


if __name__ == "__main__":
    main()