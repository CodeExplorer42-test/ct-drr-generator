# Script Usage Guide

This guide explains when and how to use each script in the pipeline.

## Data Acquisition Scripts

### `tcia_downloader.py`
**Purpose**: General TCIA data exploration and download  
**Use when**: You want to browse available collections or download specific series  
**Key features**:
- Interactive collection browsing
- Series metadata extraction
- Flexible download options

### `drr_ready_downloader.py`
**Purpose**: Automated download of DRR-suitable chest CT data  
**Use when**: You need chest CT data specifically for DRR generation  
**Key features**:
- Pre-filtered for chest CT scans
- Automatic series selection
- Guaranteed geometric completeness

## Analysis Scripts

### `extract_drr_metadata.py`
**Purpose**: Extract and validate DICOM geometric parameters  
**Use when**: You need to verify CT data is suitable for DRR  
**Output**: Console report of spacing, orientation, and completeness

### `analyze_ct_volume.py`
**Purpose**: Comprehensive CT volume analysis  
**Use when**: You want detailed statistics and DRR parameters  
**Output**: 
- Volume statistics (HU ranges, dimensions)
- Recommended DRR parameters
- JSON report in `outputs/analysis/`

### `analyze_ct_quality.py`
**Purpose**: Compare different projection methods  
**Use when**: Understanding projection techniques  
**Output**: Side-by-side comparison images in `outputs/analysis/`

## DRR Generation Scripts (Chronological Order)

### 1. `generate_drr.py` (Version 1)
**Status**: ❌ Flawed but educational  
**Issues**: Incorrect physics, numerical problems  
**Use for**: Understanding common mistakes

### 2. `corrected_drr_generator.py` (Version 2)
**Status**: ✅ Good quality, enhanced bone  
**Features**: MIP projections, numerical fixes  
**Use for**: Bone-enhanced visualizations

### 3. `drr_fixed.py` (Version 3)
**Status**: ✅ Fast and reliable  
**Features**: Parallel projection, diagnostics  
**Use for**: Quick testing and debugging

### 4. `drr_physics_correct.py` (Version 4)
**Status**: ⚠️ Experimental (currently broken)  
**Features**: Full ray-casting implementation  
**Use for**: Advanced physics experiments

### 5. `drr_final.py` (Version 5)
**Status**: ✅ Production ready  
**Features**: Clinical quality, proper anatomy  
**Use for**: Final DRR generation

## Recommended Workflows

### Standard DRR Generation
```bash
# 1. Download data
python scripts/drr_ready_downloader.py

# 2. Analyze volume
python scripts/analyze_ct_volume.py

# 3. Generate clinical DRRs
python scripts/drr_final.py
```

### Experimental Comparison
```bash
# Run all versions for comparison
python scripts/generate_drr.py
python scripts/corrected_drr_generator.py
python scripts/drr_fixed.py
python scripts/drr_final.py

# Compare outputs in respective folders
```

### Quality Analysis
```bash
# Analyze CT quality first
python scripts/analyze_ct_quality.py

# Then generate DRRs with chosen method
python scripts/drr_final.py
```

## Output Organization

Each script saves to specific folders:
- `generate_drr.py` → `outputs/iterations/`
- `corrected_drr_generator.py` → `outputs/final/`
- `drr_fixed.py` → `outputs/fixed/`
- `drr_physics_correct.py` → `outputs/physics_correct/`
- `drr_final.py` → `outputs/final_correct/`
- Analysis scripts → `outputs/analysis/`

## Performance Comparison

| Script | Speed | Quality | Physics Accuracy |
|--------|-------|---------|------------------|
| generate_drr.py | Fast | Poor | Wrong |
| corrected_drr_generator.py | Fast | Good | Approximate |
| drr_fixed.py | Very Fast | Good | Simplified |
| drr_physics_correct.py | Very Slow | N/A | Accurate (broken) |
| drr_final.py | Fast | Excellent | Good enough |

## Customization Tips

1. **Adjusting Contrast**: Modify gamma values in display code
2. **Changing Projections**: Edit projection angles in main()
3. **Tissue Emphasis**: Adjust attenuation coefficients
4. **Resolution**: Change detector_resolution parameters
5. **Performance**: Reduce num_samples for faster execution