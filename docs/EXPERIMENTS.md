# DRR Experiments Guide

This guide provides instructions for conducting various experiments with our DRR implementations.

## Available Implementations

### 1. Basic Implementation (`generate_drr.py`)
Use this to understand what NOT to do in DRR generation.
```bash
python scripts/generate_drr.py
```

### 2. Attenuation-Corrected (`corrected_drr_generator.py`)
Shows improved tissue differentiation but incorrect projection geometry.
```bash
python scripts/corrected_drr_generator.py
```

### 3. Parallel Projection (`drr_fixed.py`)
Fast, simple parallel beam projection with diagnostics.
```bash
python scripts/drr_fixed.py
```

### 4. Ray-Casting Implementation (`drr_physics_correct.py`)
Full cone-beam ray casting (currently produces black images - good for debugging).
```bash
python scripts/drr_physics_correct.py
```

### 5. Production Version (`drr_final.py`)
Clinical-quality DRRs with proper anatomy.
```bash
python scripts/drr_final.py
```

### 6. Failed Clinical Attempt (`drr_clinical.py`)
Attempted to create clinical chest X-ray appearance but failed.
```bash
python scripts/drr_clinical.py
```
**Issues**: Wrong background (not black), poor tissue contrast, artificial appearance

## Experiment Ideas

### 1. Parameter Tuning
Modify attenuation coefficients in `drr_final.py`:
- `mu_water`: Try values from 0.015 to 0.025
- Bone scaling factor: Adjust from 1.5 to 2.5
- Window/level adjustments for different tissue emphasis

### 2. Projection Geometry
In `drr_physics_correct.py`, experiment with:
- Source-to-detector distance (800-1200mm)
- Detector resolution (256x256 to 1024x1024)
- Number of ray samples (128 to 1024)

### 3. Visualization Methods
Try different approaches in any implementation:
- Linear vs logarithmic scaling
- Different gamma values (0.3 to 1.0)
- Various colormaps (bone, hot, viridis)

### 4. Performance Optimization
- Compare execution times between implementations
- Profile ray-casting vs parallel projection
- Test GPU acceleration with CuPy

### 5. Clinical Applications
- Generate DRRs at different angles (oblique views)
- Create maximum intensity projections (MIP)
- Implement dual-energy subtraction

## Data Analysis Scripts

### CT Volume Analysis
```bash
python scripts/analyze_ct_volume.py
```
Provides detailed volume statistics and parameter recommendations.

### Quality Comparison
```bash
python scripts/analyze_ct_quality.py
```
Compares different projection methods side-by-side.

## Adding New Experiments

1. Copy `drr_final.py` as a template
2. Modify the physics parameters or projection method
3. Save outputs to a new subfolder in `outputs/experiments/`
4. Document your changes in this file

## Troubleshooting

- **Black images**: Check coordinate systems and ray-volume intersections
- **Too bright/dark**: Adjust attenuation coefficients or scaling
- **No anatomy visible**: Verify projection axis and volume orientation
- **Slow performance**: Reduce ray samples or detector resolution
- **Not clinical looking**: See requirements in CLAUDE.md for true clinical X-ray appearance

## Failed Experiments Log

### V6: Clinical Appearance (`drr_clinical.py`)
**Goal**: Create DRRs that look exactly like clinical chest X-rays for infection detection
**Result**: Failed - produced overprocessed, artificial-looking images

**What was tried**:
- Applied clinical windowing (-400 center, 1500 width)
- Enhanced tissue-specific attenuation coefficients
- Used sigmoid contrast curves
- Added edge enhancement
- Applied heavy gamma correction

**Why it failed**:
1. Background wasn't black (clinical X-rays have pure black where there's air)
2. Tissue differentiation was poor (couldn't distinguish organs clearly)
3. Over-processing made it look artificial
4. Missing the subtle gradations seen in real X-rays

**Lessons learned**: 
- Need to start from first principles of X-ray imaging
- Background MUST be black (air = no absorption)
- Bones MUST be bright white
- Soft tissues need subtle gray gradations
- Less processing might be better than more