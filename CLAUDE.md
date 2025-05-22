# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For any scripts to work, activate venv and run in venv.

## Project Overview

This is a medical imaging pipeline for generating Digitally Reconstructed Radiographs (DRRs) from chest CT data downloaded from The Cancer Imaging Archive (TCIA). The project demonstrates the complete workflow from data acquisition to clinical-quality DRR generation using proper X-ray physics simulation.

**Important**: This repository contains multiple implementations showing the evolution from broken physics to production-ready code. All versions are preserved for educational and experimental purposes.

## Implementation Versions

### V1: `generate_drr.py` (Initial Attempt - Flawed)
- **Status**: ❌ Broken physics but educational
- **Issues**: Wrong projection method, arbitrary path capping, incorrect Beer-Lambert
- **Output**: `outputs/iterations/`
- **Learning**: Shows common DRR generation mistakes

### V2: `corrected_drr_generator.py` (Improved Attenuation)
- **Status**: ✅ Good quality with enhanced bone
- **Features**: Fixed overflow, tissue-specific attenuation, MIP projections
- **Output**: `outputs/final/`
- **Use case**: When bone visualization is priority

### V3: `drr_fixed.py` (Parallel Projection)
- **Status**: ✅ Fast and reliable
- **Features**: Correct coordinate handling, diagnostic outputs
- **Output**: `outputs/fixed/`
- **Use case**: Quick testing and debugging

### V4: `drr_physics_correct.py` (Ray-Casting)
- **Status**: ⚠️ Experimental (produces black images)
- **Features**: Full cone-beam geometry, trilinear interpolation
- **Output**: `outputs/physics_correct/`
- **Issues**: Coordinate system bugs need fixing

### V5: `drr_final.py` (Production Ready)
- **Status**: ✅ Best quality, recommended
- **Features**: Proper anatomical projections, clinical contrast
- **Output**: `outputs/final_correct/`
- **Use case**: Final DRR generation for any purpose

### V6: `drr_clinical.py` (Failed Clinical Attempt)
- **Status**: ❌ Failed - Not clinically realistic
- **Features**: Attempted clinical windowing, sigmoid curves, edge enhancement
- **Output**: `outputs/clinical/`
- **Issues**: Wrong background color, poor tissue differentiation, artificial appearance
- **Learning**: Need to match real clinical X-ray characteristics

### V7: `drr_true_clinical.py` (Failed - Oversaturated)
- **Status**: ❌ Failed - Oversaturated and incorrect aspect ratio
- **Features**: Black background, tissue-specific attenuation
- **Output**: `outputs/true_clinical/`
- **Issues**: Squished aspect ratio, washed out bones, oversaturated contrast
- **Learning**: Need proper aspect ratio handling and calibrated attenuation

### V8: `drr_refined.py` (Clinical Success) ✨ RECOMMENDED
- **Status**: ✅ Clinical quality achieved
- **Features**: 
  - Correct aspect ratio (3:1 for anisotropic voxels)
  - Calibrated attenuation coefficients
  - Clinical windowing (center=-938, width=1079)
  - Preserved tissue detail without washout
- **Output**: `outputs/refined/`
- **Use case**: Clinical-quality chest X-rays for medical interpretation

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv tcia_env
source tcia_env/bin/activate

# Install dependencies (including scipy for ray-casting)
pip install -r requirements.txt
```

### Standard Workflow
```bash
# 1. Download DRR-ready chest CT data
python scripts/drr_ready_downloader.py

# 2. Analyze CT volume (optional but recommended)
python scripts/analyze_ct_volume.py

# 3. Generate production DRRs
python scripts/drr_final.py

# For experiments, see docs/EXPERIMENTS.md
```

### Testing Different Implementations
```bash
# Run all versions to compare outputs
python scripts/generate_drr.py          # V1 - See what NOT to do
python scripts/corrected_drr_generator.py # V2 - Enhanced bone
python scripts/drr_fixed.py              # V3 - Fast parallel
python scripts/drr_physics_correct.py    # V4 - Broken ray-casting
python scripts/drr_final.py              # V5 - Production quality
python scripts/drr_clinical.py           # V6 - Failed clinical attempt
```

## Architecture & Data Flow

### Processing Pipeline
1. **Data Acquisition** (`tcia_downloader.py` or `drr_ready_downloader.py`)
   - Downloads DICOM CT data from TCIA public datasets
   - Stores in `data/tciaDownload/`
   - Creates metadata CSV files

2. **Metadata Extraction** (`extract_drr_metadata.py`)
   - Validates geometric completeness of DICOM data
   - Extracts spacing, orientation, and position parameters

3. **Volume Analysis** (`analyze_ct_volume.py`)
   - Performs comprehensive CT volume assessment
   - Generates parameter recommendations for DRR

4. **Quality Analysis** (`analyze_ct_quality.py`)
   - Compares different projection methods (sum, MIP, attenuation-based)
   - Creates comparison visualizations in `outputs/analysis/`

5. **DRR Generation** (Multiple implementations V1-V5)
   - Each version demonstrates different approaches
   - V5 (`drr_final.py`) is recommended for production use

### Key Technical Components

**Medical Imaging Stack**:
- SimpleITK for DICOM volume loading and processing
- pydicom (via tcia_utils) for metadata extraction
- numpy for numerical computations
- matplotlib for high-quality image output
- scipy for interpolation (ray-casting implementation)

**DRR Physics Implementation (V5 - Production)**:
- Parallel projection (fast and sufficient for medical use)
- Tissue-specific attenuation coefficients:
  - Air/lung: 0.0001 mm⁻¹
  - Soft tissue: 0.019-0.020 mm⁻¹
  - Bone: up to 0.048 mm⁻¹
- Logarithmic film response simulation
- Gamma correction for clinical appearance

### Data Organization
- Input DICOM files: `data/tciaDownload/[SeriesUID]/`
- Metadata CSVs: `data/downloadSeries_metadata_*.csv`
- Implementation outputs:
  - V1: `outputs/iterations/`
  - V2: `outputs/final/`
  - V3: `outputs/fixed/`
  - V4: `outputs/physics_correct/`
  - V5: `outputs/final_correct/`
- Analysis outputs: `outputs/analysis/`
- Experiment space: `outputs/experiments/`

## Important Considerations

### Medical Data Handling
- DICOM files contain patient data - handle with appropriate care
- The downloaded data is from public TCIA datasets (NSCLC-Radiomics collection)
- Series UID: 1.3.6.1.4.1.32722.99.99.298991776521342375010861296712563382046

### DRR Generation Parameters
- Default projections: Anterior-Posterior (AP) and Lateral views
- Output resolution: 300 DPI for publication quality
- Volume typically contains 100-150 slices at 512×512 resolution
- Voxel spacing varies but is typically around 1×1×3 mm

### Numerical Stability
- All implementations after V1 include overflow protection
- Path length calculations properly handle chest dimensions (200-300mm)
- Attenuation values validated to be within realistic ranges

## Evolution Summary (What We Learned)

### V1 → V2: Fixed Numerical Issues
- **Problem**: Overflow in exponential calculations
- **Solution**: Proper scaling and clamping
- **Result**: Stable computation but wrong projection geometry

### V2 → V3: Fixed Coordinate Systems
- **Problem**: SimpleITK (x,y,z) vs NumPy (z,y,x) confusion
- **Solution**: Consistent coordinate handling
- **Result**: Recognizable anatomy but needed better projection

### V3 → V4: Attempted Full Physics
- **Problem**: Parallel projection too simple?
- **Attempt**: Full cone-beam ray-casting
- **Result**: Too complex, debugging needed, very slow

### V4 → V5: Pragmatic Solution
- **Realization**: Parallel projection is sufficient
- **Focus**: Correct anatomical axes and display
- **Result**: Clinical-quality DRRs, fast execution

## Known Issues to Fix

### V4 Ray-Casting Implementation
1. Ray-volume intersection produces empty/black images
2. Coordinate transformation errors between source-detector-volume
3. Performance too slow for practical use (needs GPU acceleration)

### General Improvements Needed
1. Add DICOM output for clinical systems
2. Implement variable angle projections
3. Add dual-energy subtraction option
4. Create GUI for parameter adjustment

## Critical Implementation Details

### Correct Projection Axes (V5)
- **AP View**: Sum along Y-axis (anterior to posterior)
- **Lateral View**: Sum along X-axis (right to left)
- **Important**: Flip Z-axis for radiographic convention

### Attenuation Model (All versions after V1)
```python
# Correct HU to attenuation conversion
mu_water = 0.019  # mm^-1 at ~70 keV
mu = mu_water * (1.0 + HU / 1000.0)

# Tissue-specific (V5)
if HU < -500:
    mu = 0.0001  # Air/lung
elif HU < 200:
    mu = mu_water * (1.0 + HU / 1000.0)  # Soft tissue
else:
    mu = mu_water * (1.5 + HU / 1000.0)  # Bone
```

### Display Transform (V5)
```python
# 1. Apply Beer-Lambert law
transmission = np.exp(-projection_scaled)

# 2. Logarithmic film response
drr = -np.log10(transmission + epsilon) / 3.0

# 3. Gamma correction
drr = np.power(drr, 0.5)

# 4. Invert for radiographic appearance
drr = 1.0 - drr
```

## Volume Specifications (Current Data)
- Volume size: 512×512×134 voxels
- Voxel spacing: (0.98, 0.98, 3.0) mm
- Physical dimensions: 50×50×40.2 cm
- HU range: -1024 to 3034
- Suggested window: center=-938, width=1079

## Critical Requirements for True Clinical Chest X-rays

Based on V6 failure, real clinical chest X-rays MUST have:

1. **Black Background**: Air/empty space = completely black (no X-ray attenuation)
2. **White Bones**: Ribs, spine, clavicles = bright white with sharp edges
3. **Gray Scale Soft Tissues**: 
   - Heart silhouette: Medium gray, clearly defined borders
   - Diaphragm: Visible dome shape
   - Blood vessels: Faint gray branching patterns in lungs
4. **Dark Lung Fields**: Nearly black but with visible:
   - Vascular markings (faint gray lines)
   - Bronchial structures
   - Any pathology (infiltrates, masses) as white/gray areas
5. **Clinical Dynamic Range**: Full black-to-white spectrum utilized
6. **Diagnostic Quality**: Can identify:
   - Pneumonia (white patches)
   - Pleural effusions (white fluid levels)
   - Masses or nodules
   - Cardiomegaly
   - Pneumothorax

Current V5 (`drr_final.py`) produces decent anatomy but lacks the clinical appearance needed for infection detection. V6 attempted to fix this but created an overprocessed, artificial-looking result.

## Future Experiments

See `docs/EXPERIMENTS.md` for detailed guides on:
1. Parameter tuning experiments
2. Performance optimization
3. New projection geometries
4. Clinical applications
5. GPU acceleration

## Quick Decision Guide

- **Need DRRs now?** → Use `drr_final.py` (V5)
- **Need bone emphasis?** → Use `corrected_drr_generator.py` (V2)
- **Debugging coordinate issues?** → Use `drr_fixed.py` (V3)
- **Learning DRR physics?** → Study all versions V1→V5
- **Want to fix ray-casting?** → Start with `drr_physics_correct.py` (V4)

Remember: All implementations are preserved for learning. V5 is production-ready.