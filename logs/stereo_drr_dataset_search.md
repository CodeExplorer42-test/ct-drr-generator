# Stereo DRR Dataset Search Log

## Date: 2025-01-23

## Objective
Find a suitable chest CT dataset from TCIA for generating stereo DRR pairs with 3-degree separation for 3D reconstruction.

## Requirements
1. Complete chest CT volume with DICOM data
2. Full geometric information (spacing, orientation, position)
3. Good quality for DRR generation
4. Different from the current NSCLC-Radiomics dataset

## Search Results

### 1. Low Dose CT Image and Projection Data (LDCT-and-Projection-data)
- **Status**: Requires restricted license agreement
- **Pros**: Contains actual projection data and geometry (DICOM-CT-PD format)
- **Cons**: Access restricted, may be overkill for DRR generation
- **URL**: https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/

### 2. CT Images in COVID-19
- **Status**: Open access
- **Format**: NIfTI (converted from DICOM)
- **Size**: 632 patients + 121 serial CTs
- **Pros**: Large dataset, unenhanced chest CTs
- **Cons**: NIfTI format (not original DICOM)
- **DOI**: https://doi.org/10.7937/TCIA.2020.GQRY-NC81

### 3. COVID19-CT-dataset
- **Status**: Open access
- **Format**: DICOM (512×512 pixels, 16-bit)
- **Size**: 1000+ CT images
- **Scanner**: NeuViz 16-slice, Helical mode
- **Pros**: Original DICOM, no contrast, good specifications
- **Cons**: COVID-specific, may have pathology

### 4. NSCLC-Radiogenomics
- **Status**: Open access
- **Format**: DICOM CT and PET/CT
- **Size**: 211 subjects
- **Pros**: Segmentation maps, tumor annotations
- **Cons**: Cancer patients, may affect DRR quality

### 5. LUNG-PET-CT-DX
- **Status**: Open access
- **Format**: DICOM CT and PET-CT
- **Pros**: XML annotations, expert-reviewed
- **Cons**: Cancer patients, mixed with PET data

### 6. QIN-LUNG-CT
- **Status**: Open access
- **Format**: DICOM CT
- **Source**: H. Lee Moffitt Cancer Center
- **Pros**: Pre-surgery diagnostic CTs
- **Cons**: NSCLC patients

## Recommendation
Based on the search, I recommend exploring these options in order:
1. **COVID19-CT-dataset** - Best match (DICOM, 16-slice helical, no contrast)
2. **QIN-LUNG-CT** - Good alternative (diagnostic quality CTs)
3. **NSCLC-Radiogenomics** - If we need segmentation data

## Next Steps
1. Download metadata from COVID19-CT-dataset
2. Select a suitable patient with complete geometric data
3. Verify DICOM tags for spacing and orientation
4. Implement stereo DRR generation