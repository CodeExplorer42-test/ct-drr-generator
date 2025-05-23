# Stereo DRR Implementation Log

## Date: 2025-01-23

## Overview
Implemented stereo DRR generation with 3-degree separation for 3D reconstruction from chest CT data.

## Datasets Used

### 1. NSCLC-Radiomics (Original)
- **Series UID**: 1.3.6.1.4.1.32722.99.99.298991776521342375010861296712563382046
- **Patient ID**: LUNG1-001
- **Slices**: 134
- **Spacing**: (0.98, 0.98, 3.0) mm
- **Scanner**: SIEMENS Biograph 40

### 2. COVID-19-NY-SBU (New)
- **Series UID**: 1.3.6.1.4.1.14519.5.2.1.99.1071.29029751181371965166204843962164
- **Patient ID**: A670621
- **Slices**: 401
- **Spacing**: (0.65, 0.65, 1.0) mm
- **Scanner**: TOSHIBA Aquilion ONE

## Implementation Details

### Stereo Parameters
- **Angular Separation**: ±3 degrees
- **Rotation Axis**: 
  - Y-axis for AP views (simulates horizontal camera movement)
  - Z-axis for Lateral views
- **Based on**: drr_refined.py (V8) - Clinical quality implementation

### Key Features
1. **Volume Rotation**: Using SimpleITK's Euler3DTransform
2. **Clinical Quality**: Maintains tissue-specific attenuation from V8
3. **Output Formats**:
   - Individual left/right eye images
   - Side-by-side comparison
   - Red-cyan anaglyph for 3D viewing

### Technical Approach
- Parallel projection (not perspective) to maintain clinical accuracy
- Rotation applied to volume before projection
- Maintains proper anatomical orientation and radiographic conventions

## Usage for 3D Reconstruction
The generated stereo pairs can be used for:
1. Depth map estimation using stereo matching algorithms
2. 3D surface reconstruction from X-ray pairs
3. Training deep learning models for 2D-to-3D conversion
4. Validation of 3D reconstruction algorithms

## Next Steps
1. Run the stereo DRR generator
2. Validate the 3-degree separation is appropriate
3. Test with 3D reconstruction algorithms
4. Consider implementing variable angle separation
5. Add perspective projection option for more realistic stereo