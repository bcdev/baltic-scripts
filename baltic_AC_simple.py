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
	# NN inputs (10): SZA, VZA, diffAzi, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit
	# NN output (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754

	# return: should probably be an array of shape (pixels, wavelengths) ??

	nnFilePath = "forwardNN_c2rcc/olci/olci_20161012/iop_rw/17x97x47_464.3.net"

	#nnFilePath = "forwardNN_c2rcc/olci/olci_20161012/iop_rw/47x37x27_443.0.net"
	NNffbpAlphaTabFast = jpy.get_type('org.esa.snap.core.nn.NNffbpAlphaTabFast')
	nnfile = open(nnFilePath, 'r')
	nnCode = nnfile.read()
	nn_iop_rw = NNffbpAlphaTabFast(nnCode)

	####
	# is functioning - > reading of NN file is correct.
	###
	# mi = np.array(nn_iop_rw.getOutmin())
	# print(mi)
	#
	# mi = np.array(nn_iop_rw.getInmin())
	# ma = np.array(nn_iop_rw.getInmax())
	# print(mi)
	# print(ma)

	#
	# #// (9.5.4)
	# #check if log_IOPs out of range
	# mi = nn_rw_iop.get().getOutmin();
	# ma = nn_rw_iop.get().getOutmax();
	# boolean iop_oor_flag = false;
	# for (int iv = 0; iv < log_iops_nn1.length; iv++) {
	# 	if (log_iops_nn1[iv] < mi[iv] | log_iops_nn1[iv] > ma[iv]) {
	# 		iop_oor_flag = true;
	# 	}
	# }
	# flags = BitSetter.setFlag(flags, FLAG_INDEX_IOP_OOR, iop_oor_flag);
	#
	# #// (9.5.5)
	# # check if log_IOPs	at limit
	# int firstIopMaxFlagIndex = FLAG_INDEX_APIG_AT_MAX;
	# for (int i = 0; i < log_iops_nn1.length; i++) {
	# 	final boolean iopAtMax = log_iops_nn1[i] > (ma[i] - log_threshfak_oor);
	# 	flags = BitSetter.setFlag(flags, i + firstIopMaxFlagIndex, iopAtMax);
	# }
	#
	# int	firstIopMinFlagIndex = FLAG_INDEX_APIG_AT_MIN;
	# for (int i = 0; i < log_iops_nn1.length; i++) {
	# 	final boolean iopAtMin = log_iops_nn1[i] < (mi[i] + log_threshfak_oor);
	# 	flags = BitSetter.setFlag(flags, i + firstIopMinFlagIndex, iopAtMin);
	# }
	#
	#

	###
	# SZA, VZA, diffAzimuthA, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit
	Double = jpy.get_type('java.lang.Double')
	#nn_in_for = np.array((30., 30., 120., 15., 25., 0., -1., -1., -1., -3.)) # caution: dtype=np.float32 leads to errror in next line.
	#log_rw_nn2 = np.array(nn_iop_rw.calc(nn_in_for), dtype=np.float32)
	#print(log_rw_nn2)

	#double_nn_in_for = np.array((Double(45.), Double(30.), Double(120.), Double(15.), Double(25.),
	#							 Double(-2.), Double(-2.), Double(-2.), Double(-2.), Double(-3.)))
	#log_rw_nn2 = np.array(nn_iop_rw.calc(double_nn_in_for), dtype=np.float32)

#Name	X	Y	Lon	Lat							log_adet	log_agelb	log_apig	log_bpart	log_bwit	rhow_1	rhow_2	rhow_3	rhow_4	rhow_5	rhow_6	rhow_7	rhow_8	rhow_9	rhow_10	rhow_11	rhow_12	OAA	OZA	SAA	SZA
#pin_1	437.5	235.5	3.825742	56.397962	-4.956355	-3.7658699	-4.3414865	-1.8608053	-2.6944041	0.009959362	0.01134242	0.014025537	0.01487543	0.010059207	0.005442682	0.0010344568	5.945379E-4	5.734044E-4	5.45633E-4	2.7334824E-4	7.638413E-5	106.33723	8.41573	155.42921	51.091805
#pin_2	315.5	189.5	3.369955	56.597397	-5.088239	-4.04009	-4.5042415	-1.5932024	-3.473691	0.013867372	0.015864057	0.019084657	0.019100675	0.01233714	0.006331922	0.0011564749	6.558736E-4	6.272484E-4	5.924552E-4	2.9530638E-4	8.18309E-5	105.94441	10.97436	154.92598	51.3791
#pin_3	261.5	291.5	3.013032	56.374021	-3.9385555	-3.2776499	-3.4899228	-1.7674016	-0.70727175	0.0077605355	0.008688468	0.010997476	0.014358465	0.012292771	0.009077219	0.0021156792	0.0012308386	0.0011759225	0.0011434247	6.446244E-4	1.8506091E-4	105.67807	12.095442	154.42883	51.258068
#pin_4	581.5	90.5	4.624798	56.662986	-5.0377655	-3.6873133	-4.332324	-2.0587044	-2.844635	0.008768552	0.009983219	0.012415685	0.013173553	0.008856366	0.004725398	9.098557E-4	5.228566E-4	5.064532E-4	4.8198106E-4	2.3771799E-4	6.621309E-5	106.967384	5.360547	156.48013	51.16074

	lam = np.array((400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 665.0, 673.75, 681.25, 708.75, 753.75))

	double_nn_in_for = np.array((Double(51.091805), Double(8.41573), Double(155.42921- 106.33723), Double(15.), Double(30.),
								 Double(-4.3414865), Double(-4.956355), Double(-3.7658699), Double(-1.8608053), Double(-2.6944041)))
	log_rw_nn2 = np.array(nn_iop_rw.calc(double_nn_in_for), dtype=np.float32) # returns always the same numbers!!

	print((log_rw_nn2))
	print(np.exp(log_rw_nn2))

	return np.ones(10)*0.03


def write_BalticP_AC_Product(product, baltic__product_path, sensor, data_dict):
	File = jpy.get_type('java.io.File')
	width = product.getSceneRasterWidth()
	height = product.getSceneRasterHeight()
	bandShape = (height, width)

	balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
	balticPACProduct.setFileLocation(File(baltic__product_path))

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

	# Create empty bands for rhow, rhown, uncertainties for rhow
	for i in range(nbands):
		bsource = product.getBand(band_name[i])  # TOA radiance

		for key in data_dict.keys():
			brtoa_name = key + "_" + str(i + 1)
			rtoaBand = balticPACProduct.addBand(brtoa_name, ProductData.TYPE_FLOAT32)
			ProductUtils.copySpectralBandProperties(bsource, rtoaBand)
			rtoaBand.setNoDataValue(np.nan)
			rtoaBand.setNoDataValueUsed(True)


	dataNames = [*data_dict.keys()]
	autoGroupingString = dataNames[0]
	for key in dataNames[1:]:
		autoGroupingString += ':' + key
	balticPACProduct.setAutoGrouping(autoGroupingString)

	writer = ProductIO.getProductWriter('BEAM-DIMAP')
	balticPACProduct.setProductWriter(writer)
	balticPACProduct.writeHeader(baltic__product_path)

	# set datarhow, rhown, uncertainties for rhow
	for key in data_dict.keys():
		x = data_dict[key].get('data')
		if not x is None:
			for i in range(nbands):
				brtoa_name = key + "_" + str(i + 1)
				rtoaBand = balticPACProduct.getBand(brtoa_name)
				out = np.array(x[:, i]).reshape(bandShape)
				rtoaBand.writeRasterData(0, 0, width, height, snp.ProductData.createInstance(np.float32(out)),
										 ProgressMonitor.NULL)


	# # Create flag coding
	# raycorFlagsBand = balticPACProduct.addBand('raycor_flags', ProductData.TYPE_UINT8)
	# raycorFlagCoding = FlagCoding('raycor_flags')
	# raycorFlagCoding.addFlag("testflag_1", 1, "Flag 1 for Rayleigh Correction")
	# raycorFlagCoding.addFlag("testflag_2", 2, "Flag 2 for Rayleigh Correction")
	# group = balticPACProduct.getFlagCodingGroup()
	# group.add(raycorFlagCoding)
	# raycorFlagsBand.setSampleCoding(raycorFlagCoding)

	balticPACProduct.closeIO()


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
	# at the moment just fixed value, single pixel spectrum shape(12)
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
	# input: data_dict holds the band
	###
	baltic__product_path = outpath + 'baltic_' + filename
	data_dict = {
		'rtoa':{
			'data': refl
		},
		'rhow':{
			'data': None
		},
		'rhown': {
			'data': None
		},
		'unc_rhow': {
			'data': None
		}
	}

	write_BalticP_AC_Product(product, baltic__product_path, sensor, data_dict)

	product.closeProductReader()


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
