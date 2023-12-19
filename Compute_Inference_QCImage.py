#!/usr/bin/env python3


import imageio 
import numpy as np
import tifffile 

import sys
import argparse
import time 

def arg_parser():
	parser = argparse.ArgumentParser(description='Quality control - generate QC 3D volume for inference with multiple layers')
	required = parser.add_argument_group('Required')
	required.add_argument('--targetdisp', type=str, required=True,
						  help='TargetDisparity TIFF file (multi-layer)')
	required.add_argument('--groundtruth', type=str, required=True,
						  help='2D TIFF file (single layer)')
	required.add_argument('--pred', type=str, required=True,
						  help='2D TIFF file (single layer)')
	required.add_argument('--output', type=str, required=True,
						  help='output TIFF file (multi-layer)')
	options = parser.add_argument_group('Options')
	options.add_argument('--threshold', type=float, default = 2.0,
		help='threshold to detect outliers between prediction and ground truth')
	options.add_argument('--mask', type=str,
						  help='Data filtering mask (single layer)')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	imgTargetDisp_name = args.targetdisp
	imgGroundTruth_name = args.groundtruth
	imgPred_name = args.pred
	output_name = args.output
	threshold = args.threshold
	imgMask_name = args.mask

	# Read TargetDisp image file
	print("\nReading TargetDisp file...")
	imgTargetDisp = imageio.imread(imgTargetDisp_name)
	imgTargetDisp = np.array(imgTargetDisp)
	imgTargetDisp_3D = np.expand_dims(imgTargetDisp, axis=0)
	print('\t imgTargetDisp shape: ', imgTargetDisp.shape)
	print('\t imgTargetDisp_3D shape: ', imgTargetDisp_3D.shape)

	# Read GroundTruth image file
	print("\nReading GroundTruth file...")
	imgGroundTruth = imageio.imread(imgGroundTruth_name)
	imgGroundTruth = np.array(imgGroundTruth)
	imgGroundTruth_3D = np.expand_dims(imgGroundTruth, axis=0)
	print('\t imgGroundTruth shape: ', imgGroundTruth.shape)
	print('\t imgGroundTruth_3D shape: ', imgGroundTruth_3D.shape)

	# Read Pred image file
	print("\nReading Pred file...")
	imgPred = imageio.imread(imgPred_name)
	imgPred = np.array(imgPred)
	imgPred_3D = np.expand_dims(imgPred, axis=0)
	print('\t imgPred shape: ', imgPred.shape)
	print('\t imgPred_3D shape: ', imgPred_3D.shape)

	# Optional: read data filtering mask
	if imgMask_name:
		print("\nReading data filtering mask...")
		imgMask = imageio.imread(imgMask_name)
		imgMask = np.array(imgMask)
		imgMask_3D = np.expand_dims(imgMask, axis=0)
		print('\t imgMask shape: ', imgMask.shape)
		print('\t imgMask_3D shape: ', imgMask_3D.shape)	

	# Computation additional layers
	#  - Change layer: absolute difference between prediction and ground truth
	#  - Outlier mask: change >= threshold (2 pixels)

	# Compute absolute image difference
	imgAbsDiff = np.absolute(imgPred - imgGroundTruth)
	imgAbsDiff_3D = np.expand_dims(imgAbsDiff, axis=0)

	# Difference above 2 pixels
	imgOutlierMap = np.uint8(np.where(imgAbsDiff >= threshold, 255, 0))
	imgOutlierMap_3D = np.expand_dims(imgOutlierMap, axis=0)
	
	# Stack layers to generate one 3D volume
	print("Generating 3D volume...")
	imgAll = np.concatenate((imgTargetDisp_3D,imgGroundTruth_3D), axis = 0)
	imgAll = np.concatenate((imgAll,imgPred_3D), axis = 0)
	imgAll = np.concatenate((imgAll,imgAbsDiff_3D), axis = 0)
	imgAll = np.concatenate((imgAll,imgOutlierMap_3D), axis = 0)
	if imgMask_name:
		imgAll = np.concatenate((imgAll,imgMask_3D), axis = 0)
	
	# Output image description
	outputimage_description = 'QC image with 5 layers: TargetDisp, GroundTruth, Pred, AbsDiff, OutlierMap'
	if imgMask_name:
		outputimage_description = outputimage_description + ' DataFilteringMask'
	
	print("Saving output volume...")
	print('\t imgAll shape: ', imgAll.shape)
	#imageio.mimwrite(output_name,imgAll)
	tifffile.imwrite(output_name,imgAll, description=outputimage_description)


	Time2 = time.time()
	TimeDiff = Time2 - Time1
	print("Execution Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

