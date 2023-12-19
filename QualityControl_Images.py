#!/usr/bin/env python3


import imageio 
import numpy as np
import pandas as pd

import sys
import argparse
import time 
import os 

def arg_parser():
	parser = argparse.ArgumentParser(description='Quality control')
	required = parser.add_argument_group('Required')
	required.add_argument('--corr', type=str, required=True,
						  help='2D Corr TIFF file (multi-layer)')
	required.add_argument('--targetdisp', type=str, required=True,
						  help='Target Disparity TIFF file (single layer)')
	required.add_argument('--groundtruth', type=str, required=True,
						  help='Ground Truth TIFF file (single layer)')
	required.add_argument('--output', type=str, required=True,
						  help='CSV file')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	img2DCorr_name = args.corr
	img2DCorr_basename = os.path.basename(img2DCorr_name)
	imgTargetDisp_name = args.targetdisp
	imgGT_name = args.groundtruth
	output_name = args.output

	# Read 2dcorr image - all layers
	#print("Reading 2dcorr file...")
	img2DCorr = imageio.mimread(img2DCorr_name,memtest=False)
	img2DCorr = np.array(img2DCorr)
	# print('\nimg2DCorr type: ', img2DCorr.dtype)
	#print('img2DCorr shape: ', img2DCorr.shape)

	# Read TargetDisp and GroundTruth images
	#print("Reading TargetDisp file...")
	imgTargetDisp = imageio.imread(imgTargetDisp_name)
	# print('imgTargetDisp type: ', imgTargetDisp.dtype)
	#print('imgTargetDisp shape: ', imgTargetDisp.shape)

	#print("Reading GroundTruth file...")
	imgGT = imageio.imread(imgGT_name)
	# print('imgGT type: ', imgGT.dtype)
	#print('imgGT shape: ', imgGT.shape)

	# Compute Image info: mean,min,max values
	img2DCorr_mean = np.mean(img2DCorr)
	img2DCorr_min = np.min(img2DCorr)
	img2DCorr_max = np.max(img2DCorr)
	
	imgTargetDisp_mean = np.mean(imgTargetDisp)
	imgTargetDisp_min = np.min(imgTargetDisp)
	imgTargetDisp_max = np.max(imgTargetDisp)
	
	imgGT_mean = np.mean(imgGT)
	imgGT_min = np.min(imgGT)
	imgGT_max = np.max(imgGT)
	# print('\n img2DCorr pixel info:')
	# print('\t img2DCorr_mean: ', img2DCorr_mean)
	# print('\t img2DCorr_min: ', img2DCorr_min)
	# print('\t img2DCorr_max: ', img2DCorr_max)

	# Compute image difference between TargetDisp and GroundTruth images
	Diff = np.abs(imgTargetDisp - imgGT)
	Diff_mean = np.mean(Diff)
	Diff_min = np.min(Diff)
	Diff_max = np.max(Diff)
	# print('Image difference:')
	# print('\t Diff_mean:',Diff_mean )
	# print('\t Diff_min:', Diff_min)
	# print('\t Diff_max:', Diff_max)
	

	# Save information to CSV file
	QC_data = np.array([[img2DCorr_basename,img2DCorr.shape[0],img2DCorr.shape[1], img2DCorr.shape[2], \
		imgTargetDisp.shape[0], imgTargetDisp.shape[1], imgGT.shape[0], imgGT.shape[1], \
		img2DCorr_mean, img2DCorr_min, img2DCorr_max, \
		imgTargetDisp_mean, imgTargetDisp_min, imgTargetDisp_max, \
		imgGT_mean, imgGT_min, imgGT_max, \
		Diff_mean, Diff_min, Diff_max]
	])
	Columns = ['FileName','img2DCorr_Shape0','img2DCorr_Shape1','img2DCorr_Shape2',\
	'imgTargetDisp_Shape0','imgTargetDisp_Shape1',\
	'imgGT_Shape0','imgGT_Shape1',
	'img2DCorr_mean','img2DCorr_min','img2DCorr_max',\
	'imgTargetDisp_mean','imgTargetDisp_min','imgTargetDisp_max',\
	'imgGT_mean','imgGT_min','imgGT_max',\
	'Diff_mean','Diff_min','Diff_max',\
	]
	df = pd.DataFrame(QC_data,columns=Columns)
	print(df.head())
	df.to_csv(output_name, index=False)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	#print("Computing Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

