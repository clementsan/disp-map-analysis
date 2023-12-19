This project contains various python scripts used for disparity map analysis


## 1. Data pre-processing stage

Note: Data preprocessing stage to generate input files for AI analysis via ir-tp-net

### 1.1. Extracting images of interest from multi-layer 3D TIFF files

Example:
``` 
python3 ./Preprocessing_2DPhaseCorrelation.py --crop_border --input $i --targetdisp $TargetDispImage \
		--groundtruth $GroundTruthImage --confidence $ConfidenceImage --disp_lma $DispLMAImage --corr $CorrImage --verbose
```

Notes:
 - Input file is a special multi-layer TIFF file
 - Output CorrImage is a 3D TIFF file with 120 layers
 - Other output files are 2D images

### 1.2. Generating 3D TIFF files as direct input files to neural network ir-tp-net

Example:
``` 
python3 ./Preprocessing_CombinedImages.py --corr $CorrImage --targetdisp $TargetDispImage \
		--groundtruth $GroundTruthImage --confidence $ConfidenceImage --disp_lma $DispLMAImage --output $CombinedImage
``` 
Notes: 
 - Output file is a 3D image with 124 layers
  - scaled to the size of the correlation image

## 2. AI analysis stage

Please see "ir-tp-net" project for neural network training and testing to generate predicted disparity map


## 3. Data post-processing stage

Note: Data postprocessing stage to analyze outputs from AI analysis

###  3.1. Density analysis

Examples:
``` 
python3 ./Compute_Density.py --pred $i --groundtruth $GroundTruthImage --adjtilesdim 1 --output $Density_CSVFile --inclusionmask $InclusionMask_File --exclusionmask $ExclusionMask_File --threshold 2.0 --verbose
``` 

Note: input file is the predicted disparity map

###  3.2. RMSE analysis

Example:
``` 
python3 ./Compute_RMSE_WithThreshold.py --pred $i --groundtruth $GroundTruthImage --adjtilesdim $AdjTilesDim --threshold $RMSE_Threshold --output $RMSE_CSVFile
			
python3 ./Compute_RMSE_WithFiltering.py --pred $i --groundtruth $GroundTruthImage --confidence $ConfidenceImage --disp_lma $DispLMAImage --adjtilesdim 1 --threshold $Threshold --output $RMSE_CSVFile --output_mask $RMSEFiltering_MaskFile
``` 

Note: input file is the predicted disparity map


## 4. Data Quality control

Quality control stage generating multi-layer 3D tiff file

``` 
python3 ./Compute_Inference_QCImage.py --pred $i --groundtruth $GroundTruthImage --targetdisp $TargetDispImage --mask $DataFilteringMask --threshold 2.0 --output $InferenceQC_File --verbose
``` 

Notes: 
 - input file is the predicted disparity map
 - output file includes predicted disparity map, ground truth map, target disparity map and mask map
