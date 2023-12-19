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
	required.add_argument('--adjtilesdim', type=int, required=True,
		help='Adjacent tiles dimensions (e.g. 1, 3 or 5) to exclude NaN border')
	required.add_argument('--output', type=str, required=True,
		help='CSV file')
	options = parser.add_argument_group('Options')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser

#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	imgPred_name = args.pred
	imgPred_basename = os.path.basename(imgPred_name)
	AdjacentTilesDim = args.adjtilesdim
	imgGT_name = args.groundtruth
	output_name = args.output

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
		# print('imgPred type: ', imgPred.dtype)
		print('imgPred shape: ', imgPred.shape)		
		# print('imgGT type: ', imgGT.dtype)
		print('imgGT shape: ', imgGT.shape)
		print('imgPred_Crop shape: ', imgPred_Crop.shape)
		print('imgGT_Crop shape: ', imgGT_Crop.shape)
		print('TestNaN_imgPred_Crop: ', TestNaN_imgPred_Crop)
		print('TestNaN_imgGT_Crop: ', TestNaN_imgGT_Crop)

	# Compute RMSE
	rmse = mean_squared_error(imgGT_Crop, imgPred_Crop, squared=False)
	print('rmse: ',rmse)

	QC_data = np.array([[imgPred_basename,rmse]])
	Columns = ['FileName','RMSE']
	df = pd.DataFrame(QC_data,columns=Columns)
	print(df.head())
	df.to_csv(output_name, index=False)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	#print("Computing Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

