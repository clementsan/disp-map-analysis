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
	parser = argparse.ArgumentParser(description='Data analysis - density computation')
	required = parser.add_argument_group('Required')
	required.add_argument('--pred', type=str, required=True,
		help='Prediction TIFF file (single layer)')
	required.add_argument('--groundtruth', type=str, required=True,
		help='Ground Truth TIFF file (single layer)')
	required.add_argument('--adjtilesdim', type=int, required=True,
		help='Adjacent tiles dimensions (e.g. 1, 3 or 5) to exclude NaN border')
	required.add_argument('--output', type=str, required=True,
		help='CSV file')
	options = parser.add_argument_group('Options')
	options.add_argument('--inclusionmask', type=str, 
						  help='save inclusion mask as output (diff < 2 pixels)')
	options.add_argument('--exclusionmask', type=str, 
						  help='save exclusion mask as output (diff >= 2 pixels)')
	options.add_argument('--threshold', type=float, default = 2.0,
		help='threshold for inclusion / exclusion mask')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser

#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	# print(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	imgPred_name = args.pred
	imgPred_basename = os.path.basename(imgPred_name)
	AdjacentTilesDim = args.adjtilesdim
	imgGT_name = args.groundtruth
	output_name = args.output
	inclusion_mask_name = args.inclusionmask
	exclusion_mask_name = args.exclusionmask
	threshold = args.threshold
	

	# Read Pred and GroundTruth images
	#print("Reading Pred file...")
	imgPred = imageio.imread(imgPred_name)


	#print("Reading GroundTruth file...")
	imgGT = imageio.imread(imgGT_name)

	# Remove NaN border when needed	
	imgPred_Crop = imgPred
	imgGT_Crop = imgGT
	if (AdjacentTilesDim == 3):
		Border = 1
		imgPred_Crop = imgPred[Border:-Border,Border:-Border]
		imgGT_Crop = imgGT[Border:-Border,Border:-Border]
	elif (AdjacentTilesDim == 5):
		Border = 2
		imgPred_Crop = imgPred[Border:-Border,Border:-Border]
		imgGT_Crop = imgGT[Border:-Border,Border:-Border]

	# Quality control - NaN
	TestNaN_imgPred_Crop = np.any(np.isnan(imgPred_Crop))
	TestNaN_imgGT_Crop = np.any(np.isnan(imgGT_Crop))

	# #  List indices with Nan values
	# ListNaN_imgPred_Crop = np.argwhere(np.isnan(imgPred_Crop))
	# print('ListNaN_imgPred_Crop: ', ListNaN_imgPred_Crop)

	# Verbose mode
	if args.verbose:
		print('imgPred type: ', imgPred.dtype)
		print('imgPred shape: ', imgPred.shape)		
		print('imgGT type: ', imgGT.dtype)
		print('imgGT shape: ', imgGT.shape)
		print('imgPred_Crop shape: ', imgPred_Crop.shape)
		print('imgGT_Crop shape: ', imgGT_Crop.shape)
		print('TestNaN_imgPred_Crop: ', TestNaN_imgPred_Crop)
		print('TestNaN_imgGT_Crop: ', TestNaN_imgGT_Crop)

	# Compute Image difference
	np_diff = np.abs(imgPred_Crop - imgGT_Crop)
	print('np_diff type: ', np_diff.dtype)
	print('np_diff shape: ', np_diff.shape)		
		
	# Generate output mask
	np_exclusionmask = np.uint8(np.where(np_diff >= threshold, 1, 0))
	np_inclusionmask = np.uint8(np.where(np_diff < threshold, 1, 0))
	print('np_inclusionmask type: ', np_inclusionmask.dtype)
	print('np_inclusionmask shape: ', np_inclusionmask.shape)		
	
	# Compute density
	density = np.sum(np_inclusionmask) / (np_inclusionmask.shape[0] * np_inclusionmask.shape[1])
	print('density: ',density)

	QC_data = np.array([[imgPred_basename,density]])
	Columns = ['FileName','Density']
	df = pd.DataFrame(QC_data,columns=Columns)
	print(df.head())
	df.to_csv(output_name, index=False)

	if inclusion_mask_name is not None:
		print('\t\t Writing output inclusion mask - imageio...')
		imageio.imwrite(inclusion_mask_name, np_inclusionmask)

	if inclusion_mask_name is not None:
		print('\t\t Writing output exclusion mask - imageio...')
		imageio.imwrite(exclusion_mask_name, np_exclusionmask)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	#print("Computing Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

