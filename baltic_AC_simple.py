import matplotlib.pyplot as plt
# from netCDF4 import Dataset
import numpy as np
# from scipy.ndimage.filters import uniform_filter
from keras.models import load_model
import pandas as pd
import snappy as snp
import os
import time
import json
import sys
from snappy import Product
from snappy import ProductData
from snappy import ProductIO
from snappy import ProductUtils
from snappy import ProgressMonitor
from snappy import FlagCoding
from snappy import jpy


def get_band_or_tiePointGrid(product, name, dtype='float32', reshape=True):
	##
	# This function reads a band or tie-points, identified by its name <name>, from SNAP product <product>
	# The fuction returns a numpy array of shape (height, width)
	##
	height = product.getSceneRasterHeight()
	width = product.getSceneRasterWidth()
	# print(height, width)
	var = np.zeros(width * height, dtype=dtype)
	if name in list(product.getBandNames()):
		product.getBand(name).readPixels(0, 0, width, height, var)
	elif name in list(product.getTiePointGridNames()):
		product.getTiePointGrid(name).readPixels(0, 0, width, height, var)
	else:
		raise Exception('{}: neither a band nor a tie point grid'.format(name))

	if reshape:
		var = var.reshape((height, width))

	return var


def read_NN_metadata(nnpath):
	##
	# read the metadata:
	# nnpath = 'D:\WORK\IdePix\\NN_training_S2\I13x11x9x6x4x3xO1_sqrt_Radical2TrainingSelection_Relu_NoScaler\\'
	meta_fnames = os.listdir(nnpath)
	meta_fn = [fn for fn in meta_fnames if 'Metadata_' in fn]
	with open(nnpath + meta_fn[0], "r") as f:
		d = f.read()
	training_meta = json.loads(d)
	f.close()
	model_fn = [fn for fn in meta_fnames if 'MetadataModel_' in fn]
	with open(nnpath + model_fn[0], "r") as f:
		d = f.read()
	model_meta = json.loads(d)
	f.close()
	return training_meta, model_meta


def radianceToReflectance_Reader(product, sensor = '', print_info=False):

	if print_info:
		height = product.getSceneRasterHeight()
		width = product.getSceneRasterWidth()
		name = product.getName()
		description = product.getDescription()
		band_names = product.getBandNames()

		print("Sensor:      %s" % sensor)
		print("Product:     %s, %s" % (name, description))
		print("Raster size: %d x %d pixels" % (width, height))
		print("Start time:  " + str(product.getStartTime()))
		print("End time:    " + str(product.getEndTime()))
		print("Bands:       %s" % (list(band_names)))


	if sensor == 'OLCI':
		rad = Level1_Reader(product, sensor, band_group='radiance')
		solar_flux = Level1_Reader(product, sensor, band_group='solar_flux')
		SZA = get_band_or_tiePointGrid(product, 'SZA', reshape=False)
		refl = np.zeros(rad.shape)
		for j in range(rad.shape[1]):
			refl[:, j] = rad[:, j] * np.pi / (solar_flux[:, j] * np.cos(SZA*np.pi/180.))
	elif sensor == 'S2':
		refl = Level1_Reader(product, sensor)

	return refl


def Level1_Reader(product, sensor, band_group='radiance'):
	input_label = []
	if sensor == 'S2':
		input_label = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
	elif sensor == 'OLCI':
		if band_group == 'radiance':
			input_label = ["Oa01_radiance", "Oa02_radiance", "Oa03_radiance", "Oa04_radiance", "Oa05_radiance",
					   "Oa06_radiance", "Oa07_radiance", "Oa08_radiance", "Oa09_radiance", "Oa10_radiance",
					   "Oa11_radiance", "Oa12_radiance", "Oa13_radiance", "Oa14_radiance", "Oa15_radiance",
					   "Oa16_radiance", "Oa17_radiance", "Oa18_radiance", "Oa19_radiance", "Oa20_radiance", "Oa21_radiance"]
		elif band_group == 'solar_flux':
			input_label = ["solar_flux_band_" + str(i+1) for i in range(21)]

	# Initialise and read all bands contained in input_label into variable X
	# X is re-organised to serve as valid input to the neural net. Each row contains one spectrum.
	band = get_band_or_tiePointGrid(product, input_label[0])
	X = np.zeros((band.shape[0] * band.shape[1], len(input_label)))
	print(X.shape)
	X[:, 0] = band.reshape((X.shape[0],))
	for i, bn in enumerate(input_label[1:]):
		#print(bn)
		band = get_band_or_tiePointGrid(product, bn)
		X[:, i + 1] = band.reshape((X.shape[0],))

	return X


def check_valid_pixel_expression_L1(product, sensor):
	if sensor == 'OLCI':
		height = product.getSceneRasterHeight()
		width = product.getSceneRasterWidth()
		quality_flags = np.zeros(width * height, dtype='uint32')
		product.getBand('quality_flags').readPixels(0, 0, width, height, quality_flags)
		#quality_flags = quality_flags.reshape((height, width))

		# Masks OLCI L1
		## flags: 31=land 30=coastline 29=fresh_inland_water 28=tidal_region 27=bright 26=straylight_risk 25=invalid
		## 24=cosmetic 23=duplicated 22=sun-glint_risk 21=dubious 20->00=saturated@Oa01->saturated@Oa21
		invalid_mask = np.bitwise_and(quality_flags, 2 ** 25) == 2 ** 25
		land_mask = (np.bitwise_and(quality_flags, 2 ** 31) == 2 ** 31) | \
					(np.bitwise_and(quality_flags, 2 ** 30) == 2 ** 30)
					#(np.bitwise_and(quality_flags, 2 ** 29) == 2 ** 29)
		bright_mask = np.bitwise_and(quality_flags, 2 ** 27) == 2 ** 27

		invalid_mask = np.logical_or(invalid_mask , np.logical_or( land_mask , bright_mask))
		valid_pixel_flag = np.logical_not(invalid_mask)

	elif sensor == 'S2':
		#todo: set valid pixel expression L1C S2
		height = product.getSceneRasterHeight()
		width = product.getSceneRasterWidth()
		valid_pixel_flag = np.ones(width * height, dtype='uint32')

	return valid_pixel_flag


def apply_forwardNN_IOP_to_rhow_keras(X, sensor):
	start_time = time.time()
	###
	# read keras NN + metadata
	NN_path = '...'  # full path to NN file.
	metaNN_path = '...'  # folder with metadata files from training
	model = load_model(NN_path)
	training_meta, model_meta = read_NN_metadata(metaNN_path)

	X_trans = np.copy(X)
	###
	# transformation of input data
	transform_method = training_meta['transformation_method']
	if transform_method == 'sqrt':
		X_trans = np.sqrt(X_trans)
	elif transform_method == 'log':
		X_trans = np.log10(X_trans)

	###
	if model_meta['scaling']:
		scaler_path = os.listdir(metaNN_path)
		scaler_path = [sp for sp in scaler_path if 'scaling' in sp][0]
		print(scaler_path)
		scaler = pd.read_csv(metaNN_path + '/' + scaler_path, header=0, sep="\t", index_col=0)
		for i in range(X.shape[1]):
			X_trans[:, i] = (X_trans[:, i] - scaler['mean'].loc[i]) / scaler['var'].loc[i]

	###
	# Application of the NN to the data.
	prediction = model.predict(X_trans)
	print(len(prediction.shape))

	print("model load, transform, predict: %s seconds " % round(time.time() - start_time, 2))
	return prediction


def apply_forwardNN_IOP_to_rhow(iop, sensor):
	#todo: use existing forward NN from c2rcc and derive rhow from iops.
	# will only give MERIS bands! (11 bands)

	return np.ones(11)*0.03


def apply_NN_to_scene(scene_path='', filename='', outpath='', sensor=''):
	###
	# Initialising a product for Reading with snappy
	##
	product = snp.ProductIO.readProduct(scene_path + filename)

	###
	# Read L1B product and convert Radiance to reflectance (if necessary).
	# returns: numpy array with shape(pixels, wavelength).
	###
	refl = radianceToReflectance_Reader(product, sensor=sensor)



	###
	# classification of pixels
	# returns: boolean array (pixels).
	###
	valid_L1 = check_valid_pixel_expression_L1(product, sensor)


	###
	# IOP to rhow
	# todo: which is the specific order for NN input ?
	# at the moment just fixed value, single pixel spectrum shape(11)
	# todo input: numpy array (pixels, iops) ?? or single pixel?
	# todo returns: numpy array (pixels, wavelength) ?? or single pixel spectrum?
	###
	iop = 0.
	rhow_mod = apply_forwardNN_IOP_to_rhow(iop, sensor)


	###
	# todo Normalisation
	###



	###
	# Writing a product
	###

	width = product.getSceneRasterWidth()
	height = product.getSceneRasterHeight()
	bandShape = (height, width)

	baltic__product_path = outpath + 'baltic_' + filename
	balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
	balticPACProduct.setFileLocation(baltic__product_path)

	ProductUtils.copyGeoCoding(product, balticPACProduct)  # geocoding is copied when tie point grids are copied,
	ProductUtils.copyTiePointGrids(product, balticPACProduct)

	if (sensor == 'OLCI'):
		nbands = 21
		band_name = ["Oa01_radiance"]
		for i in range(1, nbands):
			if (i < 9):
				band_name += ["Oa0" + str(i + 1) + "_radiance"]
			else:
				band_name += ["Oa" + str(i + 1) + "_radiance"]

	# Create rhow, rhown, uncertainties for rhow, angles
	for i in range(nbands):
		bsource = product.getBand(band_name[i]) # TOA radiance

		brtoa_name = "rtoa_" + str(i + 1)
		rtoaBand = balticPACProduct.addBand(brtoa_name, ProductData.TYPE_FLOAT32)
		ProductUtils.copySpectralBandProperties(bsource, rtoaBand)
		rtoaBand.setNoDataValue(np.nan)
		rtoaBand.setNoDataValueUsed(True)
		out = np.array(refl[:, i]).reshape(bandShape)
		#plt.imshow(out)
		#plt.show()
		# rtoaBand.setData(snp.ProductData.createInstance(np.float32(out)))
		#targetBand.setData(snp.ProductData.createInstance(np.float32(prediction.reshape(band.shape))))
		rtoaBand.writeRasterData(0, 0, width, height, snp.ProductData.createInstance(np.float32(out)), ProgressMonitor.NULL)

		brhow_name = "rhow_" + str(i + 1)
		rhowBand = balticPACProduct.addBand(brhow_name, ProductData.TYPE_FLOAT32)
		ProductUtils.copySpectralBandProperties(bsource, rhowBand)
		rhowBand.setNoDataValue(np.nan)
		rhowBand.setNoDataValueUsed(True)
		# out = np.array(prediction[:, i]).reshape(bandShape)
		# rhowBand.setData(snp.ProductData.createInstance(np.float32(out)))
		#
		brhown_name = "rhown_" + str(i + 1)
		rhownBand = balticPACProduct.addBand(brhown_name, ProductData.TYPE_FLOAT32)
		ProductUtils.copySpectralBandProperties(bsource, rhownBand)
		rhownBand.setNoDataValue(np.nan)
		rhownBand.setNoDataValueUsed(True)
		# out = np.array(prediction[:, i]).reshape(bandShape)
		# rhownBand.setData(snp.ProductData.createInstance(np.float32(out)))

		bunc_rhow_name = "unc_rhow_" + str(i + 1)
		unc_rhowBand = balticPACProduct.addBand(bunc_rhow_name, ProductData.TYPE_FLOAT32)
		ProductUtils.copySpectralBandProperties(bsource, unc_rhowBand)
		unc_rhowBand.setNoDataValue(np.nan)
		unc_rhowBand.setNoDataValueUsed(True)
		# out = np.array(prediction[:, i]).reshape(bandShape)
		# unc_rhowBand.setData(snp.ProductData.createInstance(np.float32(out)))

	writer = ProductIO.getProductWriter('BEAM-DIMAP')
	balticPACProduct.setProductWriter(writer)

	balticPACProduct.writeHeader(baltic__product_path)

	# set datarhow, rhown, uncertainties for rhow, angles
	for i in range(nbands):
		bsource = product.getBand(band_name[i]) # TOA radiance

		brtoa_name = "rtoa_" + str(i + 1)
		rtoaBand = balticPACProduct.getBand(brtoa_name)
		out = np.array(refl[:, i]).reshape(bandShape)
		#plt.imshow(out)
		#plt.show()
		# rtoaBand.setData(snp.ProductData.createInstance(np.float32(out)))
		#targetBand.setData(snp.ProductData.createInstance(np.float32(prediction.reshape(band.shape))))
		rtoaBand.writeRasterData(0, 0, width, height, snp.ProductData.createInstance(np.float32(out)), ProgressMonitor.NULL)

		######
		#need to be adapted
		# brhow_name = "rhow_" + str(i + 1)
		# rhowBand = balticPACProduct.addBand(brhow_name, ProductData.TYPE_FLOAT32)
		# ProductUtils.copySpectralBandProperties(bsource, rhowBand)
		# rhowBand.setNoDataValue(np.nan)
		# rhowBand.setNoDataValueUsed(True)
		# # out = np.array(prediction[:, i]).reshape(bandShape)
		# # rhowBand.setData(snp.ProductData.createInstance(np.float32(out)))
		# #
		# brhown_name = "rhown_" + str(i + 1)
		# rhownBand = balticPACProduct.addBand(brhown_name, ProductData.TYPE_FLOAT32)
		# ProductUtils.copySpectralBandProperties(bsource, rhownBand)
		# rhownBand.setNoDataValue(np.nan)
		# rhownBand.setNoDataValueUsed(True)
		# # out = np.array(prediction[:, i]).reshape(bandShape)
		# # rhownBand.setData(snp.ProductData.createInstance(np.float32(out)))
		#
		# bunc_rhow_name = "unc_rhow_" + str(i + 1)
		# unc_rhowBand = balticPACProduct.addBand(bunc_rhow_name, ProductData.TYPE_FLOAT32)
		# ProductUtils.copySpectralBandProperties(bsource, unc_rhowBand)
		# unc_rhowBand.setNoDataValue(np.nan)
		# unc_rhowBand.setNoDataValueUsed(True)
		# # out = np.array(prediction[:, i]).reshape(bandShape)
		# # unc_rhowBand.setData(snp.ProductData.createInstance(np.float32(out)))

	balticPACProduct.setAutoGrouping('rtoa:rhow:rhown:unc_rhow')

	# # Create flag coding
	# raycorFlagsBand = balticPACProduct.addBand('raycor_flags', ProductData.TYPE_UINT8)
	# raycorFlagCoding = FlagCoding('raycor_flags')
	# raycorFlagCoding.addFlag("testflag_1", 1, "Flag 1 for Rayleigh Correction")
	# raycorFlagCoding.addFlag("testflag_2", 2, "Flag 2 for Rayleigh Correction")
	# group = balticPACProduct.getFlagCodingGroup()
	# group.add(raycorFlagCoding)
	# raycorFlagsBand.setSampleCoding(raycorFlagCoding)

	# add geocoding and create the product on disk (meta data, empty bands)

	writer = ProductIO.getProductWriter('BEAM-DIMAP')
	balticPACProduct.setProductWriter(writer)

	balticPACProduct.writeHeader(baltic__product_path)
	# snp.ProductIO.writeProduct(balticPACProduct, baltic__product_path, 'BEAM-DIMAP')




	###
	# Adding new bands to the product and writing a new product as output.
	# if len(prediction.shape) > 1:
	# 	print(prediction.shape)
	# 	for i in range(prediction.shape[1]):
	# 		# targetBand = product.addBand('nn_value_'+str(i)+'_'+modelN, snp.ProductData.TYPE_FLOAT32)
	# 		targetBand = product.addBand(training_meta['output_label'][i], snp.ProductData.TYPE_FLOAT32)
	# 		targetBand.setNoDataValue(np.nan)
	# 		targetBand.setNoDataValueUsed(True)
	# 		out = np.array(prediction[:, i]).reshape(band.shape)
	# 		targetBand.setData(snp.ProductData.createInstance(np.float32(out)))
	# elif len(prediction.shape) == 1:
	# 	targetBand = product.addBand(training_meta['output_label'][0], snp.ProductData.TYPE_FLOAT32)
	# 	targetBand.setNoDataValue(np.nan)
	# 	targetBand.setNoDataValueUsed(True)
	# 	targetBand.setData(snp.ProductData.createInstance(np.float32(prediction.reshape(band.shape))))
	#
	# snp.ProductIO.writeProduct(product, outpath + filename[:-4] + '_NNTest.dim', 'BEAM-DIMAP')
	product.closeProductReader()
	balticPACProduct.closeIO()


def main(args=sys.argv[1:]):
	if len(args) != 1:
		print("usage: baltic_AC_simple <SENSOR>")
		sys.exit(1)
	sensor = args[0]

	outpath = 'E:\Documents\projects\Baltic+\WP3_AC\\test_data\\results\\'
	current_path = os.path.dirname(__file__)
	path = current_path + '\\test_data\\'

	fnames = os.listdir(path)
	fnames = [fn for fn in fnames if '.dim' in fn]  # OLCI

	print(len(fnames))

	for fn in fnames[:1]:
		print(fn)
		apply_NN_to_scene(scene_path=path, filename=fn, outpath=outpath, sensor=sensor)


if __name__ == '__main__':
	main()
