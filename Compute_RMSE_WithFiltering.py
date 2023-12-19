#!/usr/bin/env python3


import imageio 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import sys
import argparse
import time 
import os 

def arg_parser():
	parser = argparse.ArgumentParser(description='Data analysis - RMSE computation')
	required = parser.add_argument_group('Required')
	required.add_argument('--pred', type=str, required=True,
		help='Prediction TIFF file (single layer)')
	required.add_argument('--groundtruth', type=str, required=True,
		help='Ground Truth TIFF file (single layer)')
	required.add_argument('--confidence', type=str, required=True,
		help='Confidence TIFF file (single layer)')
	required.add_argument('--disp_lma', type=str, required=True,
		help='LMA disparity TIFF file (single layer)')
	required.add_argument('--adjtilesdim', type=int, required=True,
		help='Adjacent tiles dimensions (e.g. 1, 3 or 5) to exclude NaN border')
	required.add_argument('--output', type=str, required=True,
		help='CSV file')
	options = parser.add_argument_group('Options')
	options.add_argument('--threshold', type=float, default = 0.15,
						  help='threshold on confidence map')
	options.add_argument('--output_mask', type=str,
		help='output mask image (TIFF file)')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser

# Remove NaN border
def cropping(img, border):
	if border == 0:
		return img
	else:
		img_cropped = img[border:-border,border:-border]
		return img_cropped

#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	
	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	imgPred_name = args.pred
	imgPred_basename = os.path.basename(imgPred_name)
	AdjacentTilesDim = args.adjtilesdim
	imgGT_name = args.groundtruth
	imgDispLMA_name = args.disp_lma
	imgConfidence_name = args.confidence
	output_name = args.output
	outputmask_name = args.output_mask
	threshold = args.threshold

	# Read images
	#print("Reading Pred file...")
	imgPred = imageio.imread(imgPred_name)

	#print("Reading GroundTruth file...")
	imgGT = imageio.imread(imgGT_name)

	#print("Reading Confidence file...")
	imgConfidence = imageio.imread(imgConfidence_name)

	#print("Reading DispLMA file...")
	imgDispLMA = imageio.imread(imgDispLMA_name)

	# Remove NaN border when needed	
	if (AdjacentTilesDim == 3):
		Border = 1	
	elif (AdjacentTilesDim == 5):
		Border = 2
	else:
		Border = 0
	imgPred_Crop = cropping(imgPred, Border)
	imgGT_Crop = cropping(imgGT, Border)
	imgConfidence_Crop = cropping(imgConfidence, Border)
	imgDispLMA_Crop = cropping(imgDispLMA, Border)

	# Quality control - NaN
	TestNaN_imgPred_Crop = np.any(np.isnan(imgPred_Crop))
	TestNaN_imgGT_Crop = np.any(np.isnan(imgGT_Crop))

	# #  List indices with Nan values
	# ListNaN_imgPred_Crop = np.argwhere(np.isnan(imgPred_Crop))
	# print('ListNaN_imgPred_Crop: ', ListNaN_imgPred_Crop)

	# Verbose mode
	if args.verbose:
		# print('imgPred type: ', imgPred.dtype)
		print('imgPred shape: ', imgPred.shape)		
		# print('imgGT type: ', imgGT.dtype)
		print('imgGT shape: ', imgGT.shape)
		print('imgPred_Crop shape: ', imgPred_Crop.shape)
		print('imgGT_Crop shape: ', imgGT_Crop.shape)
		print('TestNaN_imgPred_Crop: ', TestNaN_imgPred_Crop)
		print('TestNaN_imgGT_Crop: ', TestNaN_imgGT_Crop)


	# Define sample_weight using Confidence and DispLMA maps
	imgDispLMAMask_Crop = np.uint8(np.reshape(~np.isnan(imgDispLMA_Crop), imgDispLMA_Crop.shape))
	imgConfidenceMask_Crop = np.uint8(np.where(imgConfidence_Crop >= threshold, 1, 0))
	imgSampleWeight_Crop = np.uint8(np.logical_and(imgDispLMAMask_Crop, imgConfidenceMask_Crop))

	if outputmask_name is not None:
		imageio.imwrite(outputmask_name, imgSampleWeight_Crop * 255)

	# Compute RMSE
	rmse = mean_squared_error(imgGT_Crop, imgPred_Crop, sample_weight=imgSampleWeight_Crop, squared=False)
	print('rmse: ',rmse)

	QC_data = np.array([[imgPred_basename,rmse]])
	Columns = ['FileName','RMSE']
	df = pd.DataFrame(QC_data,columns=Columns)
	#print(df.head())

	print('Saving CSV file...')
	df.to_csv(output_name, index=False)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	#print("Computing Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

