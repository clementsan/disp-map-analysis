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
	required.add_argument('--confidence', type=str, required=True,
						  help='2D confidence TIFF file (single layer)')
	required.add_argument('--output', type=str, required=True,
						  help='CSV file')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	imgConfidence_name = args.confidence
	imgConfidence_basename = os.path.basename(imgConfidence_name)
	output_name = args.output

	# Read confidence image - all layers
	#print("Reading confidence file...")
	imgConfidence = imageio.imread(imgConfidence_name)
	print('\nimgConfidence type: ', imgConfidence.dtype)
	print('imgConfidence shape: ', imgConfidence.shape)

	Bool_NaN = np.isnan(imgConfidence)
	Nb_NaN = np.count_nonzero(Bool_NaN)
	imgConfidence_NotNaN = imgConfidence[~np.isnan(imgConfidence)]
	print('\nNb_NaN: ', Nb_NaN)
	print('\nimgConfidence_NotNaN type: ', imgConfidence_NotNaN.dtype)
	print('imgConfidence_NotNaN shape: ', imgConfidence_NotNaN.shape)

	# Compute Image info: mean,min,max values
	imgConfidence_mean = np.mean(imgConfidence_NotNaN)
	imgConfidence_min = np.min(imgConfidence_NotNaN)
	imgConfidence_max = np.max(imgConfidence_NotNaN)	
	print('\nimgConfidence_mean : ', imgConfidence_mean)
	print('imgConfidence_min : ', imgConfidence_min)
	print('imgConfidence_max : ', imgConfidence_max)
	
	# Save information to CSV file
	QC_data = np.array([[imgConfidence_basename,imgConfidence.shape[0],imgConfidence.shape[1], \
		Nb_NaN, imgConfidence_mean, imgConfidence_min, imgConfidence_max]
	])
	Columns = ['FileName','imgConfidence_Shape0','imgConfidence_Shape1', \
	'Nb_NaN','imgConfidence_mean','imgConfidence_min','imgConfidence_max']
	df = pd.DataFrame(QC_data,columns=Columns)
	print(df.head())
	df.to_csv(output_name, index=False)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	#print("Computing Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

