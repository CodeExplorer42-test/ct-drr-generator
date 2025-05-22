#!/usr/bin/env python3
"""
Comprehensive CT Volume Analysis Script
Analyzes CT slices for volume characteristics, projection data, and DRR generation parameters
"""

import os
import sys
import json
import numpy as np
import pydicom
import SimpleITK as sitk
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CTVolumeAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dicom_files = []
        self.volume_data = None
        self.metadata = {}
        self.analysis_results = {}
        
    def load_dicom_series(self):
        """Load all DICOM files from the directory"""
        logger.info(f"Loading DICOM files from {self.data_dir}")
        
        # Get all DICOM files
        self.dicom_files = sorted([
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.dcm')
        ])
        
        logger.info(f"Found {len(self.dicom_files)} DICOM files")
        
        # Read the series using SimpleITK
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(self.dicom_files)
        self.volume_data = reader.Execute()
        
        # Get basic volume information
        self.metadata['num_slices'] = len(self.dicom_files)
        self.metadata['volume_size'] = self.volume_data.GetSize()
        self.metadata['volume_spacing'] = self.volume_data.GetSpacing()
        self.metadata['volume_origin'] = self.volume_data.GetOrigin()
        self.metadata['volume_direction'] = self.volume_data.GetDirection()
        
        logger.info(f"Volume size: {self.metadata['volume_size']}")
        logger.info(f"Volume spacing: {self.metadata['volume_spacing']}")
        
    def analyze_dicom_metadata(self):
        """Analyze DICOM metadata from all slices"""
        logger.info("Analyzing DICOM metadata...")
        
        # Read first, middle, and last slices for metadata analysis
        sample_indices = [0, len(self.dicom_files)//2, len(self.dicom_files)-1]
        
        self.metadata['dicom_info'] = {}
        
        for idx in sample_indices:
            ds = pydicom.dcmread(self.dicom_files[idx])
            
            # Extract key metadata
            slice_info = {
                'filename': os.path.basename(self.dicom_files[idx]),
                'patient_id': str(ds.PatientID) if hasattr(ds, 'PatientID') else None,
                'study_date': str(ds.StudyDate) if hasattr(ds, 'StudyDate') else None,
                'modality': str(ds.Modality) if hasattr(ds, 'Modality') else None,
                'manufacturer': str(ds.Manufacturer) if hasattr(ds, 'Manufacturer') else None,
                'slice_thickness': float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else None,
                'pixel_spacing': list(map(float, ds.PixelSpacing)) if hasattr(ds, 'PixelSpacing') else None,
                'image_position': list(map(float, ds.ImagePositionPatient)) if hasattr(ds, 'ImagePositionPatient') else None,
                'image_orientation': list(map(float, ds.ImageOrientationPatient)) if hasattr(ds, 'ImageOrientationPatient') else None,
                'kvp': float(ds.KVP) if hasattr(ds, 'KVP') else None,
                'exposure': float(ds.Exposure) if hasattr(ds, 'Exposure') else None,
                'window_center': float(ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter) if hasattr(ds, 'WindowCenter') else None,
                'window_width': float(ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth) if hasattr(ds, 'WindowWidth') else None,
                'rescale_intercept': float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else None,
                'rescale_slope': float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else None,
            }
            
            self.metadata['dicom_info'][f'slice_{idx}'] = slice_info
            
        # Analyze consistency across slices
        self._check_slice_consistency()
        
    def _check_slice_consistency(self):
        """Check consistency of spacing and positioning across slices"""
        logger.info("Checking slice consistency...")
        
        positions = []
        spacings = []
        
        for i in range(0, len(self.dicom_files), max(1, len(self.dicom_files)//10)):
            ds = pydicom.dcmread(self.dicom_files[i])
            if hasattr(ds, 'ImagePositionPatient'):
                positions.append(list(map(float, ds.ImagePositionPatient)))
            if hasattr(ds, 'PixelSpacing'):
                spacings.append(list(map(float, ds.PixelSpacing)))
                
        positions = np.array(positions)
        spacings = np.array(spacings)
        
        # Calculate slice spacing consistency
        if len(positions) > 1:
            slice_distances = np.diff(positions[:, 2])  # Z-axis differences
            self.metadata['slice_spacing_mean'] = float(np.mean(slice_distances))
            self.metadata['slice_spacing_std'] = float(np.std(slice_distances))
            self.metadata['slice_spacing_consistent'] = bool(np.std(slice_distances) < 0.1)
            
        self.metadata['pixel_spacing_consistent'] = bool(np.std(spacings) < 0.01) if len(spacings) > 1 else True
        
    def analyze_volume_characteristics(self):
        """Analyze volume characteristics for DRR generation"""
        logger.info("Analyzing volume characteristics...")
        
        # Convert to numpy array
        volume_array = sitk.GetArrayFromImage(self.volume_data)
        
        self.analysis_results['volume_shape'] = volume_array.shape
        self.analysis_results['voxel_count'] = int(np.prod(volume_array.shape))
        self.analysis_results['memory_size_mb'] = float(volume_array.nbytes / (1024 * 1024))
        
        # Intensity statistics
        self.analysis_results['intensity_stats'] = {
            'min': float(np.min(volume_array)),
            'max': float(np.max(volume_array)),
            'mean': float(np.mean(volume_array)),
            'std': float(np.std(volume_array)),
            'percentiles': {
                '1%': float(np.percentile(volume_array, 1)),
                '5%': float(np.percentile(volume_array, 5)),
                '25%': float(np.percentile(volume_array, 25)),
                '50%': float(np.percentile(volume_array, 50)),
                '75%': float(np.percentile(volume_array, 75)),
                '95%': float(np.percentile(volume_array, 95)),
                '99%': float(np.percentile(volume_array, 99))
            }
        }
        
        # Analyze non-zero voxels (tissue)
        non_zero_mask = volume_array > -900  # Threshold for non-air voxels
        self.analysis_results['tissue_voxels'] = {
            'count': int(np.sum(non_zero_mask)),
            'percentage': float(np.sum(non_zero_mask) / volume_array.size * 100),
            'mean_intensity': float(np.mean(volume_array[non_zero_mask]))
        }
        
        # Calculate volume bounding box
        self._calculate_bounding_box(volume_array)
        
        # Analyze for optimal DRR parameters
        self._analyze_drr_parameters(volume_array)
        
    def _calculate_bounding_box(self, volume_array):
        """Calculate bounding box of the actual data"""
        logger.info("Calculating volume bounding box...")
        
        # Find non-background voxels
        threshold = np.percentile(volume_array, 5)
        mask = volume_array > threshold
        
        # Find bounding box
        coords = np.argwhere(mask)
        if len(coords) > 0:
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            
            self.analysis_results['bounding_box'] = {
                'min_coords': list(map(int, min_coords)),
                'max_coords': list(map(int, max_coords)),
                'size': list(map(int, max_coords - min_coords + 1))
            }
        else:
            self.analysis_results['bounding_box'] = None
            
    def _analyze_drr_parameters(self, volume_array):
        """Analyze optimal parameters for DRR generation"""
        logger.info("Analyzing optimal DRR parameters...")
        
        # Calculate physical dimensions
        spacing = np.array(self.metadata['volume_spacing'])
        size = np.array(self.metadata['volume_size'])
        physical_size = spacing * size
        
        self.analysis_results['physical_dimensions'] = {
            'size_mm': list(map(float, physical_size)),
            'size_cm': list(map(float, physical_size / 10))
        }
        
        # Suggest DRR parameters
        self.analysis_results['suggested_drr_params'] = {
            'source_to_detector_distance': 1000.0,  # Standard SID in mm
            'source_to_patient_distance': 600.0,    # Typical SPD in mm
            'detector_size': [430, 430],             # Standard detector size in mm
            'detector_spacing': [0.5, 0.5],          # Detector pixel spacing in mm
            'projection_angles': {
                'ap': [0, 0, 0],                     # Anterior-Posterior
                'lateral': [0, 90, 0],               # Lateral
                'oblique_45': [0, 45, 0]             # Oblique 45 degrees
            },
            'ray_casting_threshold': float(np.percentile(volume_array, 10)),
            'window_level': float(np.mean([np.percentile(volume_array, 30), 
                                         np.percentile(volume_array, 70)])),
            'window_width': float(np.percentile(volume_array, 95) - 
                                np.percentile(volume_array, 5))
        }
        
    def generate_sample_projections(self):
        """Generate sample projections for visualization"""
        logger.info("Generating sample projections...")
        
        volume_array = sitk.GetArrayFromImage(self.volume_data)
        
        # Create simple MIP (Maximum Intensity Projection) in three views
        projections = {
            'axial_mip': np.max(volume_array, axis=0),
            'sagittal_mip': np.max(volume_array, axis=2),
            'coronal_mip': np.max(volume_array, axis=1),
            'axial_avg': np.mean(volume_array, axis=0),
            'sagittal_avg': np.mean(volume_array, axis=2),
            'coronal_avg': np.mean(volume_array, axis=1)
        }
        
        self.analysis_results['projection_stats'] = {}
        for name, proj in projections.items():
            self.analysis_results['projection_stats'][name] = {
                'shape': proj.shape,
                'min': float(np.min(proj)),
                'max': float(np.max(proj)),
                'mean': float(np.mean(proj))
            }
            
        return projections
        
    def save_analysis_report(self, output_dir):
        """Save comprehensive analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Combine all analysis data
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_directory': self.data_dir,
            'metadata': self.metadata,
            'volume_analysis': self.analysis_results
        }
        
        # Save as JSON
        report_path = os.path.join(output_dir, 'ct_volume_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Analysis report saved to {report_path}")
        
        # Save projections as images
        projections = self.generate_sample_projections()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, proj) in enumerate(projections.items()):
            ax = axes[idx]
            im = ax.imshow(proj, cmap='gray')
            ax.set_title(name)
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            
        plt.tight_layout()
        projection_path = os.path.join(output_dir, 'sample_projections.png')
        plt.savefig(projection_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sample projections saved to {projection_path}")
        
        # Generate text summary
        self._generate_text_summary(output_dir)
        
    def _generate_text_summary(self, output_dir):
        """Generate human-readable text summary"""
        summary_path = os.path.join(output_dir, 'ct_volume_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("CT VOLUME ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.data_dir}\n\n")
            
            f.write("VOLUME INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of slices: {self.metadata['num_slices']}\n")
            f.write(f"Volume dimensions: {self.metadata['volume_size']}\n")
            f.write(f"Voxel spacing (mm): {self.metadata['volume_spacing']}\n")
            f.write(f"Physical size (cm): {self.analysis_results['physical_dimensions']['size_cm']}\n")
            f.write(f"Memory size: {self.analysis_results['memory_size_mb']:.2f} MB\n\n")
            
            f.write("INTENSITY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            stats = self.analysis_results['intensity_stats']
            f.write(f"Min: {stats['min']:.2f}\n")
            f.write(f"Max: {stats['max']:.2f}\n")
            f.write(f"Mean: {stats['mean']:.2f}\n")
            f.write(f"Std: {stats['std']:.2f}\n\n")
            
            f.write("SUGGESTED DRR PARAMETERS:\n")
            f.write("-" * 30 + "\n")
            drr_params = self.analysis_results['suggested_drr_params']
            f.write(f"Source-to-Detector Distance: {drr_params['source_to_detector_distance']} mm\n")
            f.write(f"Source-to-Patient Distance: {drr_params['source_to_patient_distance']} mm\n")
            f.write(f"Detector Size: {drr_params['detector_size']} mm\n")
            f.write(f"Window Level: {drr_params['window_level']:.2f}\n")
            f.write(f"Window Width: {drr_params['window_width']:.2f}\n")
            
        logger.info(f"Text summary saved to {summary_path}")

def main():
    # Set up paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "tciaDownload" / "1.3.6.1.4.1.32722.99.99.298991776521342375010861296712563382046"
    output_dir = project_root / "outputs" / "analysis"
    
    # Create analyzer
    analyzer = CTVolumeAnalyzer(str(data_dir))
    
    # Run analysis
    try:
        analyzer.load_dicom_series()
        analyzer.analyze_dicom_metadata()
        analyzer.analyze_volume_characteristics()
        analyzer.save_analysis_report(str(output_dir))
        
        logger.info("CT volume analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main() 