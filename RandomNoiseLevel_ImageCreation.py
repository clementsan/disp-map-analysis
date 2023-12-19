#!/usr/bin/env python3


import imageio 
import numpy as np

import sys
import argparse
import time 

def arg_parser():
	parser = argparse.ArgumentParser(description='Generate combined image with random noise level')
	required = parser.add_argument_group('Required')
	required.add_argument('--template', type=str, required=True,
						  help='Combined TIFF file (multi-layer)')
	required.add_argument('--offsetTemplate', type=float, required=True,
						  help='Template noise offset level')
	required.add_argument('--noiseLevel', type=float, required=True,
						  help='Noise offset level')
	required.add_argument('--output', type=str, required=True,
						  help='Output combined TIFF file (multi-layer)')
	options = parser.add_argument_group('Options')
	options.add_argument('--tileSize', type=int, default=15,
						  help='Tile Size')
	options.add_argument('--outputTargetDisp', type=str,
						  help='Output TargetDisp (2D image)')
	options.add_argument('--outputGroundTruth', type=str,
						  help='Output GroundTruth (2D image)')
	return parser


# Steps
# Option to define offset noise level
# Assess offset images based on noise level
# Load all images as list
# for each tile, pick among available images
# Save image

#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	template_name = args.template
	offset_template = args.offsetTemplate
	noise_level = args.noiseLevel
	output_name = args.output
	tile_size = args.tileSize
	output_TargetDisp = args.outputTargetDisp
	output_GroundTruth = args.outputGroundTruth

	# Offsets
	List_AvailableOffsets = np.array([-5.000, -4.003, -3.116, -2.341, -1.676, -1.122, -0.679, -0.346, -0.125, -0.014, \
		0.014, 0.125, 0.346, 0.679, 1.122, 1.676, 2.341, 3.116, 4.003, 5.000])
	#print('List_AvailableOffsets',List_AvailableOffsets)

	# Filter offsets based on offsetLevel
	List_Offsets = List_AvailableOffsets[np.logical_and(List_AvailableOffsets >= -noise_level,List_AvailableOffsets <= noise_level)]
	print('\n List_Offsets',List_Offsets)

	# Nb images
	nb_images = len(List_Offsets)
	print('nb_images',nb_images)

	# Read template combined image
	print("\n Reading template file...")
	imgTemplate_layerList = imageio.mimread(template_name,memtest=False)
	imgTemplate = np.stack(imgTemplate_layerList, axis=0)
	print('\t imgTemplate type: ', imgTemplate.dtype)
	print('\t imgTemplate shape: ', imgTemplate.shape)

	print("\n Creating image stack...")
	image_stack = np.zeros((nb_images, imgTemplate.shape[0], imgTemplate.shape[1], imgTemplate.shape[2]), dtype = imgTemplate.dtype)
	# print('image_stack type: ', image_stack.dtype)
	# print('image_stack shape: ', image_stack.shape)

	# List_ImageNames
	List_ImageNames = []
	for i, offset in enumerate(List_Offsets):
		print('\t Reading Image {} with offset {:.3f} ...'.format(i,offset))
		offset_string = "{:.3f}".format(offset)
		FileName_current = template_name.replace(str(offset_template),offset_string)
		List_ImageNames.append(FileName_current)
		img_current_layerList = imageio.mimread(FileName_current,memtest=False)
		img_current = np.stack(img_current_layerList, axis=0)
		image_stack[i,...] = img_current
	#print('List_ImageNames',List_ImageNames)


	# Generating image stack
	# image_stack = np.stack(List_Images, axis=0)
	print('\t image_stack type: ', image_stack.dtype)
	print('\t image_stack shape: ', image_stack.shape)

	# Output image
	print("\n Generating output image by random input patch selection...")
	imgOutput = np.zeros(imgTemplate.shape, dtype = imgTemplate.dtype)
	print('\t imgOutput type: ', imgOutput.dtype)
	print('\t imgOutput shape: ', imgOutput.shape)

	i = j = 0
	for i in range(0,imgTemplate.shape[1], tile_size):
		for j in range(0,imgTemplate.shape[2], tile_size):
			random_selection = np.random.randint(nb_images)
			#print('i j rnd: {} {} {}'.format(i,j,random_selection))
			#imgOutput[:,i:i+tile_size,j:j+tile_size] = tile_size * patchNb
			imgOutput[:,i:i+tile_size,j:j+tile_size] = image_stack[random_selection,:,i:i+tile_size,j:j+tile_size]

	imgOutput_TargetDisp = imgOutput[-2,:,:]
	imgOutput_TargetDisp = imgOutput_TargetDisp[::tile_size,::tile_size]
	print('\t imgOutput_TargetDisp shape: ', imgOutput_TargetDisp.shape)

	imgOutput_GroundTruth = imgOutput[-1,:,:]
	imgOutput_GroundTruth = imgOutput_GroundTruth[::tile_size,::tile_size]
	print('\t imgOutput_GroundTruth shape: ', imgOutput_GroundTruth.shape)

	# Write output combined image (multi-layer)
	print("\n Writing output file - combined image...")
	imageio.mimwrite(output_name,imgOutput)

	# Write output TargetDisp image (2D)
	if (output_TargetDisp is not None):
		print("\nWriting output file - TargetDisp image...")
		imageio.imwrite(output_TargetDisp,imgOutput_TargetDisp)

	# Write output GroundTruth image (2D)
	if (output_GroundTruth is not None):
		print("\nWriting output file - GroundTruth image...")
		imageio.imwrite(output_GroundTruth,imgOutput_GroundTruth)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	print("Reading Time: "+str(TimeDiff))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

