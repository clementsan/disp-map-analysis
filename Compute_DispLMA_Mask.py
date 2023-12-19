#!/usr/bin/env python3


import imageio 
import numpy as np
import pandas as pd

import sys
import argparse
import time 
import os 

def arg_parser():
	parser = argparse.ArgumentParser(description='Data analysis - NaN computation')
	required = parser.add_argument_group('Required')
	required.add_argument('--disp_lma', type=str, required=True,
		help='Prediction TIFF file (single layer)')
	required.add_argument('--output', type=str, required=True,
		help='Output CSV file')
	options = parser.add_argument_group('Options')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	options.add_argument('--mask', type=str,
						  help='verbose mode')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	DispLMA_name = args.disp_lma
	output_name = args.output

	# Read Pred and GroundTruth images
	#print("Reading Pred file...")
	imgDispLMA = imageio.imread(DispLMA_name)

	# Verbose mode
	if args.verbose:
		print('imgDispLMA type: ', imgDispLMA.dtype)
		print('imgDispLMA shape: ', imgDispLMA.shape)		
		# print('imgGT type: ', imgGT.dtype)

	# Compute NaN
	Bool_NaN = np.isnan(imgDispLMA)
	Nb_NaN = np.count_nonzero(Bool_NaN)
	Nb_NotNaN = np.count_nonzero(~Bool_NaN)
	Nb_Total = imgDispLMA.shape[0] * imgDispLMA.shape[1]
	NaN_Percent = Nb_NaN / Nb_Total
	
	if args.verbose:
		print('Nb_NaN: ', Nb_NaN)
		print('Nb_NotNaN: ', Nb_NotNaN)
		print('Nb_Total: ', Nb_Total)
		print('NaN_Percent: ', NaN_Percent)


	# Mask
	DispLMA_Mask = np.uint8(np.reshape(~Bool_NaN, imgDispLMA.shape))
	if args.verbose:
		print('DispLMA_Mask type: ', DispLMA_Mask.dtype)
		print('DispLMA_Mask shape: ', DispLMA_Mask.shape)		
	

	QC_data = np.array([[DispLMA_name, Nb_NaN, Nb_NotNaN, Nb_Total, NaN_Percent]])
	Columns = ['FileName','Nb_NaN', 'Nb_NotNaN', 'Nb_Total', 'NaN_Percent']
	df = pd.DataFrame(QC_data,columns=Columns)
	print(df.head())
	df.to_csv(output_name, index=False)

	if (args.mask is not None):
		Mask_FileName = args.mask
		imageio.imwrite(Mask_FileName, DispLMA_Mask * 255)


	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	#print("Computing Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

