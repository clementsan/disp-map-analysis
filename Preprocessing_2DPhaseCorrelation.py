#!/usr/bin/env python3



'''
  Notes:
	- Pillow 5.1.0. Version 4.1.1 throws error (VelueError):
	  ~$ (sudo) pip3 install Pillow --upgrade
	  ~$ python3
	  >>> import PIL
	  >>> PIL.PILLOW_VERSION
	  '5.1.0'
'''

from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import xml.dom.minidom as minidom
import time

# Added by Clement
import imageio
#import tifffile
import argparse

#http://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[38;5;214m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	BOLDWHITE = '\033[1;37m'
	UNDERLINE = '\033[4m'

# Data cropping (border of specific size)
def crop_border(image, size):
	List_FirstRows = list(range(0,size))
	List_LastRows = list(range(image.shape[0]-size,image.shape[0]))
	List_Rows = List_FirstRows + List_LastRows
	#print('ListRows: ', List_Rows)

	List_FirstCols = list(range(0,size))
	List_LastCols = list(range(image.shape[1]-size,image.shape[1]))
	List_Cols = List_FirstCols + List_LastCols
	#print('List_Cols: ', List_Cols)

	CroppedImage = np.delete(image,List_Rows,axis=0)
	CroppedImage = np.delete(CroppedImage,List_Cols,axis=1)

	return CroppedImage

# 2D Corr data cropping
def delete_tile_splits(image, tile_size):
	List_RowSplit = list(range(tile_size-1,image.shape[0],tile_size))
	List_ColSplit = list(range(tile_size-1,image.shape[1],tile_size))
	#print('List_RowSplit: ', List_RowSplit)
	#print('List_ColSplit: ', List_ColSplit)

	CroppedImage = np.delete(image,List_RowSplit,axis=0)
	CroppedImage = np.delete(CroppedImage,List_ColSplit,axis=1)

	return CroppedImage

def save_metadata_layer(image, filename, index, is_cropping, verbose):
		#metadata_layer = tiles_meta[0,:,:,index]
		metadata_layer = image.getMeta(index = index, layer = 0)
		if is_cropping:
			metadata_layer = crop_border(metadata_layer, 1)
		if verbose:
			print('\metadata_layer shape: ', metadata_layer.shape)
		if not os.path.exists(os.path.dirname(filename)):
			os.makedirs(os.path.dirname(filename))
		imageio.imwrite(filename, metadata_layer)

# reshape to tiles (e.g. 16x16)
def get_tile_images(image, width=8, height=8):
	_nrows, _ncols, depth = image.shape
	_size = image.size
	_strides = image.strides
	
	nrows, _m = divmod(_nrows, height)
	ncols, _n = divmod(_ncols, width)
	if _m != 0 or _n != 0:
		return None

	return np.lib.stride_tricks.as_strided(
		np.ravel(image),
		shape=(nrows, ncols, height, width, depth),
		strides=(height * _strides[0], width * _strides[1], *_strides),
		writeable=False
	)

# TiffFile has no len exception
#import imageio

#from libtiff import TIFF
'''
Description:
	Reads TIFF files with multiple layers that were saved by imagej
Methods:
	.getstack(items=[])
		returns np.array, layers are stacked along depth - think of RGB channels
		@items - if empty = all, if not - items[i] - can be layer index or layer's label name
	.channel(index)
		returns np.array of a single layer
	.show_images(items=[])
		@items - if empty = all, if not - items[i] - can be layer index or layer's label name
	.show_image(index)
Examples:
#1

'''
class imagej_tiff:
	# imagej stores labels lengths in this tag
	__TIFF_TAG_LABELS_LENGTHS = 50838
	# imagej stores labels contents in this tag
	__TIFF_TAG_LABELS_STRINGS = 50839
	# init
	def __init__(self,filename, layers = None, tile_list = None):
		# file name
		self.fname = filename
		tif = Image.open(filename)
		# total number of layers in tiff
		self.nimages = tif.n_frames
		# labels array
		self.labels = []
		# infos will contain xml data Elphel stores in some of tiff files
		self.infos = []
		# dictionary from decoded infos[0] xml data
		self.props = {}
	
		# bits per sample, type int
		self.bpp = tif.tag[258][0]

		# Extract and parse header information
		self.__split_labels(tif.n_frames,tif.tag)
		self.__parse_info()
		try:
			self.nan_bug = self.props['VERSION']== '1.0' # data between min and max is mapped to 0..254 instead of  1.255
		except:
			self.nan_bug = False # other files, not ML ones
		# image layers stacked along depth - (think RGB)
		self.image = []
	
		if layers is None:
		# fill self.image
			for i in range(self.nimages):
				tif.seek(i)
				a = np.array(tif)
				a = np.reshape(a,(a.shape[0],a.shape[1],1))
				
				#a = a[:,:,np.newaxis]
				# exclude layer named 'other'
				if self.bpp==8:
					_min = self.data_min
					_max = self.data_max
					_MIN = 1
					_MAX = 255
					if (self.nan_bug):
						_MIN = 0
						_MAX = 254
					else:
						if self.labels[i]!='other':
							a[a==0]=np.nan
					a = a.astype(float)
					if self.labels[i]!='other':
						a = (_max-_min)*(a-_MIN)/(_MAX-_MIN)+_min
				if i==0:
					self.image = a
					# stack along depth (think of RGB channels)
				else:
					self.image = np.append(self.image,a,axis=2)
		else:
			if tile_list is None:
				indx = 0
				for layer in layers:
					tif.seek(self.labels.index(layer)) 
					a = np.array(tif)
					if not indx:
						self.image = np.empty((a.shape[0],a.shape[1],len(layers)),a.dtype)
					self.image[...,indx] = a
					indx += 1
			else:
				other_label = "other"
	#            print(tile_list)
				num_tiles =  len(tile_list)
				num_layers = len(layers)
				tiles_corr = np.empty((num_tiles,num_layers,self.tileH*self.tileW),dtype=float)
	#            tiles_other=np.empty((num_tiles,3),dtype=float)
				tiles_other=self.gettilesvalues(
						 tif = tif,
						 tile_list=tile_list,
						 label=other_label)
				for nl,label in enumerate(layers):
					tif.seek(self.labels.index(label))
					layer = np.array(tif) # 8 or 32 bits
					tilesX = layer.shape[1]//self.tileW
					for nt,tl in enumerate(tile_list):
						ty = tl // tilesX
						tx = tl % tilesX
	#                    tiles_corr[nt,nl] = np.ravel(layer[self.tileH*ty:self.tileH*(ty+1),self.tileW*tx:self.tileW*(tx+1)])
						a = np.ravel(layer[self.tileH*ty:self.tileH*(ty+1),self.tileW*tx:self.tileW*(tx+1)])
						#convert from int8
						if self.bpp==8:
							a = a.astype(float)
							if np.isnan(tiles_other[nt][0]):
								# print("Skipping NaN tile ",tl)
								a[...] = np.nan
							else:
								_min = self.data_min
								_max = self.data_max
								_MIN = 1
								_MAX = 255
								if (self.nan_bug):
									_MIN = 0
									_MAX = 254
								else:    
									a[a==0] = np.nan
								a = (_max-_min)*(a-_MIN)/(_MAX-_MIN)+_min
						tiles_corr[nt,nl] = a    
						pass
					pass
				self.corr2d =           tiles_corr
				self.target_disparity = tiles_other[...,0]
				self.gt_ds =            tiles_other[...,1:3]
				pass
		# init done, close the image
		if (self.props['VERSION']== 2.0):
#            self.tileH = self.image.shape[0]//self.props['tileStepY']
#            self.tileW = self.image.shape[1]//self.props['tileStepX']
			self.tileH = self.props['tileStepY']
			self.tileW = self.props['tileStepX']
			pass
		
		tif.close()
	#   label == tiff layer name
	def getvalues(self,label=""):
		l = self.getstack([label],shape_as_tiles=True)
		res = np.empty((l.shape[0],l.shape[1],3))
	
		for i in range(res.shape[0]):
			for j in range(res.shape[1]):
				# 9x9 -> 81x1
				m = np.ravel(l[i,j])
				if self.bpp==32:
					res[i,j,0] = m[0]
					res[i,j,1] = m[2]
					res[i,j,2] = m[4]
				elif self.bpp==8:
					res[i,j,0] = ((m[0]-128)*256+m[1])/128
					res[i,j,1] = ((m[2]-128)*256+m[3])/128
					res[i,j,2] = (m[4]*256+m[5])/65536.0
				else:
					res[i,j,0] = np.nan
					res[i,j,1] = np.nan
					res[i,j,2] = np.nan
	
		# NaNize
		a = res[:,:,0]
		a[a==-256] = np.nan
		b = res[:,:,1]
		b[b==-256] = np.nan
		c = res[:,:,2]
		c[c==0] = np.nan
		return res

	# 3 values per tile: target disparity, GT disparity, GT confidence
	def gettilesvalues(self,
					 tif,
					 tile_list,
					 label=""):
		res = np.empty((len(tile_list),3),dtype=float)
		tif.seek(self.labels.index(label))
		layer = np.array(tif) # 8 or 32 bits
		tilesX = layer.shape[1]//self.tileW
		for i,tl in enumerate(tile_list):
			ty = tl // tilesX
			tx = tl % tilesX
			m = np.ravel(layer[self.tileH*ty:self.tileH*(ty+1),self.tileW*tx:self.tileW*(tx+1)])
			if self.bpp==32:
				res[i,0] = m[0]
				res[i,1] = m[2]
				res[i,2] = m[4]
			elif self.bpp==8:
				res[i,0] = ((m[0]-128)*256+m[1])/128
				res[i,1] = ((m[2]-128)*256+m[3])/128
				res[i,2] = (m[4]*256+m[5])/65536.0
			else:
				res[i,0] = np.nan
				res[i,1] = np.nan
				res[i,2] = np.nan
		# NaNize
		a = res[...,0]
		a[a==-256] = np.nan
		b = res[...,1]
		b[b==-256] = np.nan
		c = res[...,2]
		c[c==0] = np.nan
		
		return res

	# get ordered stack of images by provided items
	# by index or label name. Divides into [self.tileH][self.tileW] tiles
	def getstack(self,items=[],shape_as_tiles=False):
		a = ()
		if len(items)==0:
			b = self.image
		else:
			for i in items:
				if type(i)==int:
					a += (self.image[:,:,i],)
				elif type(i)==str:
					j = self.labels.index(i)
					a += (self.image[:,:,j],)
			# stack along depth
			b = np.stack(a,axis=2)
		if shape_as_tiles:
			b = get_tile_images(b,self.tileW,self.tileH)
	
		return b

	# Trim stack to correlation data (e.g 15x15 instead of 16x16)
	def trimStack (self, stack, radius = 0):
		if (radius == 0):
			radius=self.props['corrRadius']
		corr_side = 2*radius+1
		return stack[:,:,:,:corr_side,:corr_side]
	# get np.array of a channel
	# * do not handle out of bounds
	def channel(self,index):
		return self.image[:,:,index]


	# extract 2D corr data
	def get2DCorr(self):
		im_2DCorr = crop_border(self.image, self.tileH)
		#print('im_2DCorr shape', im_2DCorr.shape)
		im_2DCorr = delete_tile_splits(im_2DCorr, self.tileH)
		#print('im_2DCorr shape', im_2DCorr.shape)
		# Delete last 2 layers
		# im_2DCorr = np.delete(im_2DCorr,im_2DCorr.shape[2]-1,axis=2)
		im_2DCorr = np.delete(im_2DCorr,[120,121],axis=2)
		return im_2DCorr


	# Retrieve specific meta image defined by index & layer
	def getMeta(self, index, layer = 0):
		_, meta = self.getCorrsMeta(items=[layer])
		meta = np.squeeze(meta)
		return meta[:,:,index]


	# Retrieve correlation information (tiles) and meta-data
	def getCorrsMeta(self,items=[]):
		stack0 =  self.getstack(items,shape_as_tiles=True)
		stack = np.moveaxis(stack0, 4, 0) # slices - first index
		
		radius=self.props['corrRadius']
		num_meta=self.props['numMeta']
		corr_side = 2*radius+1
		corr_tiles = stack[:,:,:,:corr_side,:corr_side]
		
		meta = stack[:,:,:,-1,:num_meta]
		return corr_tiles, meta/self.props['tileMetaScale'] 

	# display images by index or label
	def show_images(self,items=[]):

		# show listed only
		if len(items)>0:
			for i in items:
				if type(i)==int:
					self.show_image(i)
				elif type(i)==str:
					j = self.labels.index(i)
					self.show_image(j)
		# show all
		else:
			for i in range(self.nimages):
				self.show_image(i)


	# display single image
	def show_image(self,index):
	# display using matplotlib

		t = self.image[:,:,index]
		mytitle = "("+str(index+1)+" of "+str(self.nimages)+") "+self.labels[index]
		fig = plt.figure()
		fig.canvas.set_window_title(self.fname+": "+mytitle)
		fig.suptitle(mytitle)
		#plt.imshow(t,cmap=plt.get_cmap('gray'))
		plt.imshow(t)
		plt.colorbar()
	
		# display using Pillow - need to scale
	
		# remove NaNs - no need
		#t[np.isnan(t)]=np.nanmin(t)
		# scale to [min/max*255:255] range
		#t = (1-(t-np.nanmax(t))/(t-np.nanmin(t)))*255
		#tmp_im = Image.fromarray(t)
		#tmp_im.show()

	# puts etrees in infos
	def __parse_info(self):

		infos = []
		for info in self.infos:
			infos.append(ET.fromstring(info))
		self.infos = infos
	
		# specifics
		# properties dictionary
		pd = {}
	
		if infos:
			for child in infos[0]:
				#print(child.tag+"::::::"+child.text)
				pd[child.tag] = child.text
		
			self.props = pd
			file_version = float(self.props['VERSION'])
			if (file_version < 2.0):
				# tiles are squares (older version
				self.tileW = int(self.props['tileWidth'])
				self.tileH = int(self.props['tileWidth'])
				self.data_min = float(self.props['data_min'])
				self.data_max = float(self.props['data_max'])
			else:
				floats=['dispOffsetLow','tileMetaScale','disparity_low','dispOffset',
				'fatZero','disparity_pwr','VERSION','dispOffsetHigh',
				'disparity_high']
				ints = ['metaGTConfidence','tileMetaSlice','indexReference','metaLastDiff',
				 'metaGTDisparity','metaFracValid', 'numScenes','metaTargetDisparity',
				 'disparity_steps','tileStepX',    'tileStepY', "corrRadius","numMeta"]
				bools=['randomize_offsets']
				for key in pd:
					val = pd[key]
					if key in bools:
						if (val == '1') or (val == 'true') or (val == 'True'):
							pd[key] = 1
						else:
							pd[key] = 0  
						pass
					elif key in ints:
						pd[key] = int(pd[key])
					elif  key in floats:   
						pd[key] = float(pd[key])
				try:       
					pd['corrRadius'] = pd['corrRadius'] # not yet exists
				except:
					pd['corrRadius'] = 7              
				try:       
					pd['numMeta'] = pd['numMeta'] # not yet exists
				except:
					pd['numMeta'] = 6              
						
			pass        

	# makes arrays of labels (strings) and unparsed xml infos
	def __split_labels(self,n,tag):
		# list
		tag_lens = tag[self.__TIFF_TAG_LABELS_LENGTHS]
		# string
		tag_labels = tag[self.__TIFF_TAG_LABELS_STRINGS].decode()
		# remove 1st element: it's something like IJIJlabl..
		tag_labels = tag_labels[tag_lens[0]:]
		tag_lens = tag_lens[1:]
		# the last ones are images labels
		# normally the difference is expected to be 0 or 1
		skip = len(tag_lens) - n
	
		self.labels = []
		self.infos = []
		for l in tag_lens:
			string = tag_labels[0:l].replace('\x00','')
			if skip==0:
				self.labels.append(string)
			else:
				self.infos.append(string)
				skip -= 1
			tag_labels = tag_labels[l:]


def arg_parser():
	parser = argparse.ArgumentParser(description='Extract images from specific multi-layer tiff file')
	required = parser.add_argument_group('Required')
	required.add_argument('-i', '--input', type=str, required=True,
						  help='input TIFF file (specific multi-layer)')
	options = parser.add_argument_group('Options')
	options.add_argument('--crop_border', action="store_true",
						  help='Crop image border (removing NaN border)')
	options.add_argument('--corr', type=str,
						  help='Save 2D Corr TIFF file (multi-layer)')
	options.add_argument('--targetdisp', type=str,
						  help='Save Target Disparity TIFF file (single layer)')
	options.add_argument('--groundtruth', type=str,
						  help='Save Ground Truth TIFF file (single layer)')
	options.add_argument('--confidence', type=str, 
						  help='Save ground truth confidence image (single layer')
	options.add_argument('--disp_lma', type=str, 
						  help='Save LMA disparity image (single layer')
	options.add_argument('--frac_valid', type=str, 
						  help='Save fraction valid image (single layer')
	options.add_argument('--last_diff', type=str, 
						  help='Save image (single layer')
	options.add_argument('--disp_bg', type=str, 
						  help='Save image (single layer')
	options.add_argument('--confidence_bg', type=str, 
						  help='Save image (single layer')
	options.add_argument('--disp_lma_bg', type=str, 
						  help='Save image (single layer')
	options.add_argument('--last_diff_bg', type=str, 
						  help='Save image (single layer')
	options.add_argument('--disp_fg', type=str, 
						  help='Save image (single layer')
	options.add_argument('--disp_bg_all', type=str, 
						  help='Save image (single layer')
	options.add_argument('--blue_sky', type=str, 
						  help='Save image (single layer')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)

	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	print('Reading Tiff image...')
	ijt = imagej_tiff(args.input)

	Time2 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time2 - Time1
	print(bcolors.BOLDWHITE+"Reading Time: "+str(TimeDiff)+bcolors.ENDC)

	# Image header info
	rough_string = ET.tostring(ijt.infos[0], "utf-8")
	reparsed = minidom.parseString(rough_string)
	print('\n Image Header info:')
	print(reparsed.toprettyxml(indent="\t"))

	print("\n TIFF stack labels: "+str(ijt.labels))
	print("TIFF stack labels length: "+str(len(ijt.labels)))
	#print(ijt.infos)


	# Image shape information and tiles information
	print("\nImage shape: ",ijt.image.shape)
	print("Image size: ",ijt.image.size)
	print("Image strides: ",ijt.image.strides)

	# needed properties:
	print("Tiles shape: "+str(ijt.tileW)+"x"+str(ijt.tileH))

	try:
		print("Data min: "+str(ijt.data_min))
		print("Data max: "+str(ijt.data_max))
	except:
		print("Data min/max are not provided")


#    tiles,tiles_meta = ijt.getCorrsMeta(['0-1','1-2','2-3','3-4'])
	# nrows, ncols, height, width, depth
	tiles,tiles_meta = ijt.getCorrsMeta([])
	print("Corr stack shape: "+str(tiles.shape))
	print("Meta stack shape: "+str(tiles_meta.shape))


	# Reorder numpy array
	# img_reordered = np.transpose(ijt.image, axes=[2,0,1])
	# print('img_reordered shape',img_reordered.shape)
	# imageio.mimwrite('test_imageio.tiff',img_reordered)
	# #tifffile.imwrite('test_tifffile.tiff',img_reordered)


	# Extract and crop 2d correlation data
	if (args.corr is not None):
		Corr_FileName = args.corr
		im_2DCorr = ijt.get2DCorr()
		im_2DCorr = np.moveaxis(im_2DCorr, 2, 0)
		if args.verbose:
			print("\nImage shape: ",ijt.image.shape)
			print('im_2DCorr shape', im_2DCorr.shape)
		print('\nSaving 2DCorr image... ')
		if not os.path.exists(os.path.dirname(Corr_FileName)):
			os.makedirs(os.path.dirname(Corr_FileName))
		imageio.mimwrite(Corr_FileName, im_2DCorr)

	# Extract and crop target disparity
	if (args.targetdisp is not None):
		#TargetDisparity = tiles_meta[0,:,:,0]
		save_metadata_layer(ijt, args.targetdisp, 0, args.crop_border, args.verbose)
		print('Saving targetdisp layer...')
	
	# Extract and crop Ground Truth disparity
	if (args.groundtruth is not None):
		#GroundTruth = tiles_meta[0,:,:,1]
		save_metadata_layer(ijt, args.groundtruth, 1, args.crop_border, args.verbose)
		print('Saving groundtruth layer...')
		
	# Extract and crop ground truth confidence
	if (args.confidence is not None):
		#Confidence = tiles_meta[0,:,:,2]
		save_metadata_layer(ijt, args.confidence, 2, args.crop_border, args.verbose)
		print('Saving confidence layer...')
	
	# Extract and crop LMA disparity image
	if (args.disp_lma is not None):
		#DispLMA = tiles_meta[0,:,:,3]
		save_metadata_layer(ijt, args.disp_lma, 3, args.crop_border, args.verbose)
		print('Saving disp_lma layer...')
	
	# Extract and crop frac_valid
	if (args.frac_valid is not None):
		#FracValid = tiles_meta[0,:,:,4]
		save_metadata_layer(ijt, args.frac_valid, 4, args.crop_border, args.verbose)
		print('Saving frac_valid layer...')
	
	# Extract and crop last_diff
	if (args.last_diff is not None):
		#LastDiff = tiles_meta[0,:,:,5]
		save_metadata_layer(ijt, args.last_diff, 5, args.crop_border, args.verbose)
		print('Saving last_diff layer...')
	
	# Extract and crop disp_bg
	if (args.disp_bg is not None):
		#DispBg = tiles_meta[0,:,:,6]
		save_metadata_layer(ijt, args.disp_bg, 6, args.crop_border, args.verbose)
		print('Saving disp_bg layer...')
	
	# Extract and crop confidence_bg
	if (args.confidence_bg is not None):
		#ConfidenceBg = tiles_meta[0,:,:,7]
		save_metadata_layer(ijt, args.confidence_bg, 7, args.crop_border, args.verbose)
		print('Saving confidence_bg layer...')
	
	# Extract and crop disp_lma_bg
	if (args.disp_lma_bg is not None):
		#DispLMABg = tiles_meta[0,:,:,8]
		save_metadata_layer(ijt, args.disp_lma_bg, 8, args.crop_border, args.verbose)
		print('Saving disp_lma_bg layer...')
	
	# Extract and crop last_diff_bg
	if (args.last_diff_bg is not None):
		#LastDiffBg = tiles_meta[0,:,:,9]
		save_metadata_layer(ijt, args.last_diff_bg, 9, args.crop_border, args.verbose)
		print('Saving last_diff_bg layer...')
	
	# Extract and crop disp_fg
	if (args.disp_fg is not None):
		#DispFg = tiles_meta[0,:,:,10]
		save_metadata_layer(ijt, args.disp_fg, 10, args.crop_border, args.verbose)
		print('Saving disp_fg layer...')
	
	# Extract and crop disp_bg_all
	if (args.disp_bg_all is not None):
		#DispBgAll = tiles_meta[0,:,:,11]
		save_metadata_layer(ijt, args.disp_bg_all, 11, args.crop_border, args.verbose)
		print('Saving disp_bg_all layer...')
	
	# Extract and crop blue_sky
	if (args.blue_sky is not None):
		#BlueSky = tiles_meta[0,:,:,12]
		save_metadata_layer(ijt, args.blue_sky, 12, args.crop_border, args.verbose)
		print('Saving blue_sky layer...')
	
	Time3 = time.time()
	#print(bcolors.BOLDWHITE+"Time2: "+str(Time2)+bcolors.ENDC)
	TimeDiff = Time3 - Time1
	print(bcolors.BOLDWHITE+"Execution Time: "+str(TimeDiff)+bcolors.ENDC)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
