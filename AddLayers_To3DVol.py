#!/usr/bin/env python3


import imageio 
import numpy as np

import sys
import argparse
import time 

def arg_parser():
	parser = argparse.ArgumentParser(description='Concatenating layer to 3D volume')
	required = parser.add_argument_group('Required')
	required.add_argument('--input', type=str, required=True,
						  help='3D TIFF file (multi-layer)')
	required.add_argument('--layer1', type=str, required=True,
						  help='2D TIFF file (single layer)')
	required.add_argument('--output', type=str, required=True,
						  help='Combined TIFF file (multi-layer)')
	options = parser.add_argument_group('Options')
	required.add_argument('--layer2', type=str, required=True,
						  help='2D TIFF file (single layer)')
	options.add_argument('--repeat', type=int, default=15,
						  help='Repeat tile number (default 15)')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	imgInput_name = args.input
	imgLayer1_name = args.layer1
	imgLayer2_name = args.layer2
	output_name = args.output
	repeatNb = args.repeat
	
	# Read Combined image - all layers
	print("\nReading input file - 3D volume...")
	imgInput = imageio.mimread(imgInput_name,memtest=False)
	imgInput = np.array(imgInput)
	# print('\nimgInput type: ', imgInput.dtype)
	print('\t imgInput shape: ', imgInput.shape)

	# Read Layer1 image
	print("Reading Layer1 file...")
	imgLayer1 = imageio.imread(imgLayer1_name)
	# print('imgLayer1 type: ', imgLayer1.dtype)
	print('\t imgLayer1 shape: ', imgLayer1.shape)

	# Resample layer1 (repeating values, to match input size)
	imgLayer1_repeat0 = np.repeat(imgLayer1, repeatNb, axis=0)
	imgLayer1_repeat = np.repeat(imgLayer1_repeat0, repeatNb, axis=1)
	imgLayer1_repeat = np.expand_dims(imgLayer1_repeat, axis=0)
	print('\t imgLayer1_repeat shape: ', imgLayer1_repeat.shape)

	# Stack layer to imgInput, to generate one 3D volume
	print("Adding 2D layer to 3D volume...")
	imgAll = np.concatenate((imgInput,imgLayer1_repeat), axis = 0)

	if args.layer2:
		print("Reading Layer2 file...")
		imgLayer2 = imageio.imread(imgLayer2_name)
		# print('imgLayer2 type: ', imgLayer2.dtype)
		print('\t imgLayer2 shape: ', imgLayer2.shape)

		imgLayer2_repeat0 = np.repeat(imgLayer2, repeatNb, axis=0)
		imgLayer2_repeat = np.repeat(imgLayer2_repeat0, repeatNb, axis=1)
		imgLayer2_repeat = np.expand_dims(imgLayer2_repeat, axis=0)
		print('\t imgLayer2_repeat shape: ', imgLayer2_repeat.shape)

		# Stack layer to imgInput, to generate one 3D volume
		print("Adding 2D layer to 3D volume...")
		imgAll = np.concatenate((imgAll,imgLayer2_repeat), axis = 0)

	print("Saving output volume...")
	print('\t imgAll shape: ', imgAll.shape)
	imageio.mimwrite(output_name,imgAll)

	Time2 = time.time()
	TimeDiff = Time2 - Time1
	print("Execution Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

