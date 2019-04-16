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

def angle_Reader(product, sensor):
	if sensor == 'OLCI':
		band = get_band_or_tiePointGrid(product, 'OAA')
		oaa = np.zeros((band.shape[0] * band.shape[1]))
		oaa = band.reshape((oaa.shape[0],))

		band = get_band_or_tiePointGrid(product, 'OZA')
		oza = np.zeros((band.shape[0] * band.shape[1]))
		oza = band.reshape((oza.shape[0],))

		band = get_band_or_tiePointGrid(product, 'SZA')
		sza = np.zeros((band.shape[0] * band.shape[1]))
		sza = band.reshape((sza.shape[0],))

		band = get_band_or_tiePointGrid(product, 'SAA')
		saa = np.zeros((band.shape[0] * band.shape[1]))
		saa = band.reshape((saa.shape[0],))

	return oaa, oza, sza, saa


def calculate_diff_azim(oaa, saa):
	###
	# azimuth difference as input to the NN is defined in a range between 0° and 180°

	x = np.array(oaa - saa)
	ID = np.array( x < 0)
	if np.sum(ID)>0.:
		x[ID] = 360. + x[ID]
	ID = np.array(x > 180.)
	if np.sum(ID)>0.:
		x[ID] = x[ID] -180.

	return x

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


def apply_forwardNN_IOP_to_rhow_example():
	# NN inputs (10): SZA, VZA, diffAzi, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit
	# NN output (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754

	nnFilePath = "forwardNN_c2rcc/olci/olci_20161012/iop_rw/17x97x47_464.3.net"
	NNffbpAlphaTabFast = jpy.get_type('org.esa.snap.core.nn.NNffbpAlphaTabFast')
	nnfile = open(nnFilePath, 'r')
	nnCode = nnfile.read()
	nn_iop_rw = NNffbpAlphaTabFast(nnCode)

	###
	# IMPORTANT: initialise input as np.zeros in the correct length!!!
	# SZA, VZA, diffAzimuthA, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit
	a = np.zeros(10)
	a[0] = 51.091805
	a[1] = 8.41573
	a[2] = 155.42921 - 106.33723
	a[3] = 15.
	a[4] = 35.
	a[5] = -4.3414865
	a[6] = -4.956355
	a[7] = -3.7658699
	a[8] = -1.8608053
	a[9] = -2.6944041
	log_rw_nn2 = np.array(nn_iop_rw.calc(a), dtype=np.float32)  # returns always the same numbers!!

	print(np.exp(log_rw_nn2))
	test_rhow_fromTOA = np.array((0.009959362, 0.01134242, 0.014025537, 0.01487543, 0.010059207, 0.005442682, 0.0010344568, 5.945379E-4, 5.734044E-4,
	5.45633E-4, 2.7334824E-4, 7.638413E-5))

	lam = np.array((400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754))

	plt.plot(lam, test_rhow_fromTOA, 'c-')
	plt.plot(lam, np.exp(log_rw_nn2), 'c--')
	plt.show()

### Example 1 c2rcc 2016/2017
# Name	X	Y	Lon	Lat							log_adet	log_agelb	log_apig	log_bpart	log_bwit	rhow_1	rhow_2	rhow_3	rhow_4	rhow_5	rhow_6	rhow_7	rhow_8	rhow_9	rhow_10	rhow_11	rhow_12	OAA	OZA	SAA	SZA
	# pin_1	437.5	235.5	3.825742	56.397962	-4.956355	-3.7658699	-4.3414865	-1.8608053	-2.6944041	0.009959362	0.01134242	0.014025537	0.01487543	0.010059207	0.005442682	0.0010344568	5.945379E-4	5.734044E-4	5.45633E-4	2.7334824E-4	7.638413E-5	106.33723	8.41573	155.42921	51.091805
	# pin_2	315.5	189.5	3.369955	56.597397	-5.088239	-4.04009	-4.5042415	-1.5932024	-3.473691	0.013867372	0.015864057	0.019084657	0.019100675	0.01233714	0.006331922	0.0011564749	6.558736E-4	6.272484E-4	5.924552E-4	2.9530638E-4	8.18309E-5	105.94441	10.97436	154.92598	51.3791
	# pin_3	261.5	291.5	3.013032	56.374021	-3.9385555	-3.2776499	-3.4899228	-1.7674016	-0.70727175	0.0077605355	0.008688468	0.010997476	0.014358465	0.012292771	0.009077219	0.0021156792	0.0012308386	0.0011759225	0.0011434247	6.446244E-4	1.8506091E-4	105.67807	12.095442	154.42883	51.258068
	# pin_4	581.5	90.5	4.624798	56.662986	-5.0377655	-3.6873133	-4.332324	-2.0587044	-2.844635	0.008768552	0.009983219	0.012415685	0.013173553	0.008856366	0.004725398	9.098557E-4	5.228566E-4	5.064532E-4	4.8198106E-4	2.3771799E-4	6.621309E-5	106.967384	5.360547	156.48013	51.16074

### Example 2 c2r 201904
#Name	X	Y	Lon	Lat							iop_adet	iop_agelb	iop_apig	iop_bpart	iop_bwit	rhow_1	rhow_2	rhow_3	rhow_4	rhow_5	rhow_6	rhow_7	rhow_8	rhow_9	rhow_11	rhow_12	OAA	OZA	SAA	SZA
#pin_1	476.5	231.5	3.993265	56.381533	0.027343486	0.021865448	0.02313981	0.16197646	0.11456131	0.0096926335	0.009946768	0.009698673	0.009246618	0.0071290717	0.0045980685	0.0011407222	6.9232716E-4	6.9392053E-4	6.8216515E-4	3.2724423E-4	9.311986E-5	106.47686	7.592013	155.63219	51.038136
#pin_2	791.5	144.5	5.430028	56.375004	0.06849463	0.049048793	0.042024814	0.737513	0.65764487	0.014872396	0.015654514	0.016917245	0.019503482	0.017883737	0.014397462	0.0044253394	0.002696726	0.0026146006	0.0025478457	0.0013499877	3.8966344E-4	107.70104	0.86169	157.40138	50.720493
#pin_3	1087.5	140.5	6.659461	56.159049	0.047663715	0.052653775	0.08010797	0.55395997	0.42619917	0.009425671	0.009672539	0.009948069	0.011273121	0.010529963	0.008871017	0.0027428195	0.0016406277	0.0016266676	0.001634283	8.882042E-4	2.5643408E-4	-71.3413	5.483421	158.88246	50.267815
#pin_4	893.5	153.5	5.840092	56.27681	0.07499708	0.08046007	0.080823615	1.7166388	1.4494008	0.022707473	0.023912517	0.026493134	0.03272301	0.03230792	0.029830737	0.010091868	0.005850822	0.0055907085	0.005481569	0.0029984459	8.490639E-4	-72.05109	1.3295213	157.88736	50.542614

def load_example_data( exampleNo = 1):
	lam = np.array((400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754))
	if exampleNo == 2: # c2r 201904
		iop = np.zeros((2, 5))
		# log_apig, log_adet, log a_gelb, log_bpart, log_bwit
		iop[0, :] = (0.02313981, 0.027343486, 0.021865448, 0.16197646, 0.11456131)
		iop[1, :] = (0.042024814, 0.06849463, 0.049048793, 0.737513, 0.65764487)
		iop = np.log(iop)

		sza = np.array((51.038136, 50.720493))
		vza = np.array((7.592013, 0.86169))
		diff_azim = np.array((155.63219 - 106.47686 , 157.40138 - 107.70104))
		test_rhow_fromTOA = np.zeros((2, 12))
		test_rhow_fromTOA[0, :] = (0.0096926335, 0.009946768, 0.009698673, 0.009246618, 0.0071290717, 0.0045980685,
								   0.0011407222, 6.9232716E-4, 6.9392053E-4, 6.8216515E-4, 3.2724423E-4, 9.311986E-5)
		test_rhow_fromTOA[1, :] = (0.014872396,	0.015654514, 0.016917245, 0.019503482, 0.017883737, 0.014397462,
								   0.0044253394, 0.002696726, 0.0026146006, 0.0025478457, 0.0013499877, 3.8966344E-4)

	if exampleNo == 1:  # c2rcc 2016
		iop = np.zeros((2, 5))
		# log_apig, log_adet, log a_gelb, log_bpart, log_bwit
		iop[0, :] = (-4.3414865, -4.956355, - 3.7658699, - 1.8608053, - 2.6944041)
		iop[1, :] = (-4.5042415, -5.088239, -4.04009, -1.5932024, -3.473691)
		sza = np.array((51.091805, 51.3791))
		vza = np.array((8.41573, 10.97436))
		diff_azim = np.array((155.42921 - 106.33723, 154.92598 - 105.94441))
		test_rhow_fromTOA = np.zeros((2, 12))
		test_rhow_fromTOA[0, :] = (
			0.009959362, 0.01134242, 0.014025537, 0.01487543, 0.010059207, 0.005442682, 0.0010344568, 5.945379E-4,
			5.734044E-4, 5.45633E-4, 2.7334824E-4, 7.638413E-5)
		test_rhow_fromTOA[1, :] = (
			0.013867372, 0.015864057, 0.019084657, 0.019100675, 0.01233714, 0.006331922, 0.0011564749, 6.558736E-4,
			6.272484E-4, 5.924552E-4, 2.9530638E-4, 8.18309E-5)

	return iop, sza, vza, diff_azim, test_rhow_fromTOA, lam

def apply_forwardNN_IOP_to_rhow_arrayExample(sensor, exampleNo=1):

	if exampleNo == 1:
		iop, sza, vza, diff_azim, test_rhow_fromTOA, lam = load_example_data(1)
		rhow_mod = apply_forwardNN_IOP_to_rhow(iop, sza, vza, diff_azim, sensor)
		label = 'c2rcc_2016'

	if exampleNo == 2:
		iop, sza, vza, diff_azim, test_rhow_fromTOA, lam = load_example_data(2)
		rhow_mod = apply_forwardNN_IOP_to_rhow(iop, sza, vza, diff_azim, sensor, nn='new')
		label = 'c2rcc_2019'


	mycol = np.array(('c', 'r'))

	for i in range(rhow_mod.shape[0]):
		plt.plot(lam, rhow_mod[i, :], '--', color=mycol[i], label=label)
		#plt.plot(lam, rhow_mod_new[i, :], '-.', color=mycol[i], label='nn_2019')
		plt.plot(lam, test_rhow_fromTOA[i, :], '-', color=mycol[i], label='c2rcc_rhow_from_TOA')

	plt.legend()
	plt.show()


def apply_forwardNN_IOP_to_rhow(iop, sun_zenith, view_zenith, diff_azimuth, sensor, T=15, S=35, nn=''):
	# iop : pixels x (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)
	# sun_zenith, view_zenith, diff_azimuth (0-180°) : pixels
	# T, S: currently constant.
	# valid ranges can be found at the beginning of the .net-file.

	# NN inputs (10): SZA, VZA, diffAzi, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit
	# NN output (12 bands): log_rw at lambda = 400, 412, 443, 489, 510, 560, 620, 665, 674, 681, 709, 754
	# return: should probably be an array of shape (pixels, wavelengths) ??

	nBands = None
	if sensor == 'OLCI':
		nBands=12

	output = np.zeros((iop.shape[0], nBands))

	nnFilePath = "forwardNN_c2rcc/olci/olci_20161012/iop_rw/17x97x47_464.3.net"
	#if nn == 'new':
	#	nnFilePath = "forwardNN_c2rcc/olci/olci_20190414/iop_rw/55x55x55_40.3.net"

	NNffbpAlphaTabFast = jpy.get_type('org.esa.snap.core.nn.NNffbpAlphaTabFast')
	nnfile = open(nnFilePath, 'r')
	nnCode = nnfile.read()
	nn_iop_rw = NNffbpAlphaTabFast(nnCode)

	for i in range(iop.shape[0]):
		###
		# IMPORTANT: initialise input as np.zeros in the correct length!!! otherwise, the function will not work!
		# SZA, VZA, diffAzimuthA, T, S, log_apig, log_adet, log a_gelb, log_bpart, log_bwit
		inputNN = np.zeros(10)
		inputNN[0] = sun_zenith[i]
		inputNN[1] = view_zenith[i]
		inputNN[2] = diff_azimuth[i]
		inputNN[3] = T
		inputNN[4] = S
		for j in range(iop.shape[1]): # log_apig, log_adet, log a_gelb, log_bpart, log_bwit
			inputNN[j+5] = iop[i, j]

		log_rw_nn2 = np.array(nn_iop_rw.calc(inputNN), dtype=np.float32)  # returns always the same numbers!!

		print(np.exp(log_rw_nn2))
		output[i, :] = np.exp(log_rw_nn2)

	# #// (9.5.4)
	# #check if log_IOPs out of range
	# mi = nn_rw_iop.getOutmin();
	# ma = nn_rw_iop.getOutmax();
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

	return output


def write_BalticP_AC_Product(product, baltic__product_path, sensor, data_dict, singleBand_dict=None):
	File = jpy.get_type('java.io.File')
	width = product.getSceneRasterWidth()
	height = product.getSceneRasterHeight()
	bandShape = (height, width)

	balticPACProduct = Product('balticPAC', 'balticPAC', width, height)
	balticPACProduct.setFileLocation(File(baltic__product_path))

	ProductUtils.copyGeoCoding(product, balticPACProduct)
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

	if not singleBand_dict is None:
		for key in singleBand_dict.keys():
			singleBand = balticPACProduct.addBand(key, ProductData.TYPE_FLOAT32)
			singleBand.setNoDataValue(np.nan)
			singleBand.setNoDataValueUsed(True)

	writer = ProductIO.getProductWriter('BEAM-DIMAP')
	balticPACProduct.setProductWriter(writer)
	balticPACProduct.writeHeader(baltic__product_path)
	writer.writeProductNodes(balticPACProduct, baltic__product_path)

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

	if not singleBand_dict is None:
		for key in singleBand_dict.keys():
			x = singleBand_dict[key].get('data')
			if not x is None:
				singleBand = balticPACProduct.getBand(key)
				out = np.array(x).reshape(bandShape)
				singleBand.writeRasterData(0, 0, width, height, snp.ProductData.createInstance(np.float32(out)),
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
	# forwardNN: IOP to rhow
	# input: numpy array iop, shape=(Npixels x iops= (log_apig, log_adet, log a_gelb, log_bpart, log_bwit)),
	#		np.array sza, shape = (Npixels,)
	#		np.array vza, shape = (Npixels,)
	#		np.array diff_azim, shape = (Npixels,); range: 0-180
	# returns: numpy array, shape=(Npixels, wavelengths)
	###
	# Examples:
	# apply_forwardNN_IOP_to_rhow_example()
	apply_forwardNN_IOP_to_rhow_arrayExample(sensor, 1)
	apply_forwardNN_IOP_to_rhow_arrayExample(sensor, 2)

	###
	# iop = np.zeros((2,5))
	# # Please keep this order! Necessary for NN application : log_apig, log_adet, log a_gelb, log_bpart, log_bwit
	# iop[0,:] = (-4.3414865, -4.956355, - 3.7658699 ,- 1.8608053, - 2.6944041)
	# iop[1,:] = ( -4.5042415, -5.088239, -4.04009, -1.5932024, -3.473691)
	# sza = np.array((51.091805, 51.3791))
	# vza = np.array((8.41573, 10.97436))
	# diff_azim = np.array((155.42921-106.33723, 154.92598-105.94441)) # between 0 - 180.
	#
	# rhow_mod = apply_forwardNN_IOP_to_rhow(iop, sza, vza, diff_azim, sensor)





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

	###
	# use SNAP with option S3TBX pixelGeoCoding turned off!
	fnames = os.listdir(path)
	fnames = [fn for fn in fnames if '.dim' in fn]  # OLCI

	print(len(fnames))

	for fn in fnames[:1]:
		print(fn)
		apply_NN_to_scene(scene_path=path, filename=fn, outpath=outpath, sensor=sensor)


if __name__ == '__main__':
	main()
