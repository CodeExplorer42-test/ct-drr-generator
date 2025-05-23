# Stereo DRR Failure Analysis

## Date: 2025-01-23

## ❌ FAILURE: Initial stereo DRR implementation produced black/empty images

## Problems Identified

1. **Volume Rotation Issues**
   - SimpleITK resampling during rotation corrupted the data
   - Lost proper HU values during transformation
   - Default pixel value (-1000) dominated the rotated volume

2. **Incorrect Projection Approach**
   - Should not rotate the entire volume
   - Need to implement oblique ray casting or sheared projections
   - Current parallel projection needs modification for stereo

3. **Spacing Calculation**
   - Lost proper voxel spacing after rotation
   - Attenuation calculations became incorrect

## Why It Failed

The approach of rotating the entire CT volume before projection is fundamentally flawed because:
- Resampling introduces artifacts and data loss
- Computational overhead is massive
- Not how real stereo X-ray systems work

## Correct Approach

Real stereo radiography uses:
1. **Fixed patient/volume**
2. **Two X-ray sources** at slightly different positions
3. **Oblique projections** through the same volume

## Next Steps

Need to implement proper stereo by:
1. Keep volume fixed
2. Implement sheared/oblique projection
3. Or simulate two source positions with appropriate ray directions