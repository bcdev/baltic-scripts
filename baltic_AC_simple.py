# import matplotlib.pyplot as plt
# from netCDF4 import Dataset
import numpy as np
# from scipy.ndimage.filters import uniform_filter
from keras.models import load_model
import pandas as pd
import snappy as snp
import os
import time
import json


def get_var(product, name, dtype='float32'):
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
	return var.reshape((height, width))


def read_metadata(nnpath):
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


def apply_NN_to_scene(scene_path='', filename='', outpath='', sensor=''):
	###
	# Initialising a product for Reading with snappy
	##
	product = snp.ProductIO.readProduct(scene_path + filename)

	input_label = []
	if sensor == 'S2':
		input_label = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
	elif sensor == 'OLCI':
		input_label = ["Oa01_reflectance", "Oa02_reflectance", "Oa03_reflectance", "Oa04_reflectance",
					   "Oa05_reflectance",
					   "Oa06_reflectance", "Oa07_reflectance", "Oa08_reflectance", "Oa09_reflectance",
					   "Oa10_reflectance",
					   "Oa11_reflectance", "Oa12_reflectance", "Oa13_reflectance", "Oa14_reflectance",
					   "Oa15_reflectance",
					   "Oa16_reflectance", "Oa17_reflectance", "Oa18_reflectance", "Oa19_reflectance",
					   "Oa20_reflectance", "Oa21_reflectance"]

	# Initialise and read all bands contained in input_label into variable X
	# X is re-organised to serve as valid input to the neural net
	band = get_var(product, input_label[0])
	X = np.zeros((band.shape[0] * band.shape[1], len(input_label)))
	print(X.shape)
	X[:, 0] = band.reshape((X.shape[0],))
	for i, bn in enumerate(input_label[1:]):
		band = get_var(product, bn)
		X[:, i + 1] = band.reshape((X.shape[0],))

	start_time = time.time()

	###
	# read keras NN + metadata
	NN_path = '...'  # full path to NN file.
	metaNN_path = '...'  # folder with metadata files from training
	model = load_model(NN_path)
	training_meta, model_meta = read_metadata(metaNN_path)

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

	###
	# Adding new bands to the product and writing a new product as output.
	if len(prediction.shape) > 1:
		print(prediction.shape)
		for i in range(prediction.shape[1]):
			# targetBand = product.addBand('nn_value_'+str(i)+'_'+modelN, snp.ProductData.TYPE_FLOAT32)
			targetBand = product.addBand(training_meta['output_label'][i], snp.ProductData.TYPE_FLOAT32)
			targetBand.setNoDataValue(np.nan)
			targetBand.setNoDataValueUsed(True)
			out = np.array(prediction[:, i]).reshape(band.shape)
			targetBand.setData(snp.ProductData.createInstance(np.float32(out)))
	elif len(prediction.shape) == 1:
		targetBand = product.addBand(training_meta['output_label'][0], snp.ProductData.TYPE_FLOAT32)
		targetBand.setNoDataValue(np.nan)
		targetBand.setNoDataValueUsed(True)
		targetBand.setData(snp.ProductData.createInstance(np.float32(prediction.reshape(band.shape))))

	snp.ProductIO.writeProduct(product, outpath + filename[:-4] + '_NNTest.dim', 'BEAM-DIMAP')
	product.closeProductReader()


def main(args=sys.argv[1:]):
	if len(args) != 1:
		print("usage: baltic_AC_simple <SENSOR>")
		sys.exit(1)
	sensor = args[0]

	outpath = ''

	path = "E:\Documents\projects\IdePix\data\S3_NN_test\L1_reproc_O2harm\\"

	fnames = os.listdir(path)
	fnames = [fn for fn in fnames if 'homogenIdepix.dim' in fn]  # OLCI

	print(len(fnames))

	for fn in fnames[:5]:
		print(fn)
		apply_NN_to_scene(scene_path=path, filename=fn, outpath=outpath, sensor=sensor)


if __name__ == '__main__':
	main()
