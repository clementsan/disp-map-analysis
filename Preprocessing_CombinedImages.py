#!/usr/bin/env python3


import imageio 
import numpy as np

import sys
import argparse
import time 

def arg_parser():
	parser = argparse.ArgumentParser(description='Combine images to generate 3D volume')
	required = parser.add_argument_group('Required')
	required.add_argument('--corr', type=str, required=True,
						  help='2D Corr TIFF file (multi-layer)')
	required.add_argument('--targetdisp', type=str, required=True,
						  help='Target Disparity TIFF file (single layer)')
	required.add_argument('--groundtruth', type=str, required=True,
						  help='Ground Truth TIFF file (single layer)')
	required.add_argument('--confidence', type=str, required=True,
						  help='Confidence TIFF file (single layer)')
	required.add_argument('--disp_lma', type=str, required=True,
						  help='LMA Disparity TIFF file (single layer)')
	required.add_argument('--output', type=str, required=True,
						  help='Combined TIFF file (multi-layer)')
	options = parser.add_argument_group('Options')
	options.add_argument('--repeat', type=int, default=15,
						  help='Repeat tile number (default 15)')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	img2DCorr_name = args.corr
	imgTargetDisp_name = args.targetdisp
	imgGT_name = args.groundtruth
	imgConfidence_name = args.confidence
	imgDispLMA_name = args.disp_lma
	output_name = args.output
	repeatNb = args.repeat
	
	# Read 2dcorr image - all layers
	print("Reading 2dcorr file...")
	img2DCorr = imageio.mimread(img2DCorr_name,memtest=False)
	img2DCorr = np.array(img2DCorr)
	# print('\nimg2DCorr type: ', img2DCorr.dtype)
	# print('img2DCorr shape: ', img2DCorr.shape)

	# Read TargetDisp and GroundTruth images
	print("Reading TargetDisp file...")
	imgTargetDisp = imageio.imread(imgTargetDisp_name)
	# print('imgTargetDisp type: ', imgTargetDisp.dtype)
	# print('imgTargetDisp shape: ', imgTargetDisp.shape)

	print("Reading GroundTruth file...")
	imgGT = imageio.imread(imgGT_name)
	# print('imgGT type: ', imgGT.dtype)
	# print('imgGT shape: ', imgGT.shape)

	print("Reading Confidence file...")
	imgConfidence = imageio.imread(imgConfidence_name)
	# print('imgConfidence type: ', imgConfidence.dtype)
	# print('imgConfidence shape: ', imgConfidence.shape)

	print("Reading DispLMA file...")
	imgDispLMA = imageio.imread(imgDispLMA_name)
	# print('imgDispLMA type: ', imgDispLMA.dtype)
	# print('imgDispLMA shape: ', imgDispLMA.shape)

	# - - - - - - - - - - -
	print("Generating combined image...")
	# Resample TargetDisp and imgTG (repeating values, to match img2DCor size)
	imgTargetDisp_repeat0 = np.repeat(imgTargetDisp, repeatNb, axis=0)
	imgTargetDisp_repeat = np.repeat(imgTargetDisp_repeat0, repeatNb, axis=1)
	imgTargetDisp_repeat = np.expand_dims(imgTargetDisp_repeat, axis=0)

	imgGT_repeat0 = np.repeat(imgGT, repeatNb, axis=0)
	imgGT_repeat = np.repeat(imgGT_repeat0, repeatNb, axis=1)
	imgGT_repeat = np.expand_dims(imgGT_repeat, axis=0)
	# print('imgTargetDisp_repeat shape: ', imgTargetDisp_repeat.shape)
	# print('imgGT_repeat shape: ', imgGT_repeat.shape)

	imgConfidence_repeat0 = np.repeat(imgConfidence, repeatNb, axis=0)
	imgConfidence_repeat = np.repeat(imgConfidence_repeat0, repeatNb, axis=1)
	imgConfidence_repeat = np.expand_dims(imgConfidence_repeat, axis=0)
	print('\t imgConfidence_repeat shape: ', imgConfidence_repeat.shape)

	imgDispLMA_repeat0 = np.repeat(imgDispLMA, repeatNb, axis=0)
	imgDispLMA_repeat = np.repeat(imgDispLMA_repeat0, repeatNb, axis=1)
	imgDispLMA_repeat = np.expand_dims(imgDispLMA_repeat, axis=0)
	print('\t imgDispLMA_repeat shape: ', imgDispLMA_repeat.shape)

	# Stack layers to img2dCorr, to generate one 3D volume
	imgAll = np.concatenate((img2DCorr,imgTargetDisp_repeat), axis = 0)
	imgAll = np.concatenate((imgAll,imgGT_repeat), axis = 0)
	imgAll = np.concatenate((imgAll,imgConfidence_repeat), axis = 0)
	imgAll = np.concatenate((imgAll,imgDispLMA_repeat), axis = 0)
	
	imageio.mimwrite(output_name,imgAll)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	print("Reading Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

