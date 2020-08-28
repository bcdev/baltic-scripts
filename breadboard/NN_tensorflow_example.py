#!/usr/bin/python
# -*- coding: utf-8 -*-
import datetime
import json
import locale
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

#sys.path.append('/home/cmazeran/.snap/snap-python')
sys.path.append('C:\\Users\\Dagmar\\snap7\\snappy')
import snappy as snp

from baltic_AC_backwardNN import angle_Reader, calculate_diff_azim, check_valid_pixel_expression_L1


def run_bwNN_fwNN_loop(rhow, sza, oza, nn_raa, valid):
    # bands_forwardNN = [400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754]
    # bands_backwardNN = [400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754]

    bwpath = "NN_reciprocal/bwNNc2rcc_LossLogRhow_fwNN97_2e6_I15x77x77x77xO5batch300_epoch100000_loss0.00242.h5"
    fwpath = "NN_reciprocal/fwNNc2rcc_linearLossI8x97x97x97xO12batch300_epoch200000_loss0.0.h5"
    model_fw = tf.keras.models.load_model(fwpath)
    model_bw = tf.keras.models.load_model(bwpath)

    ## transforming all angles by cosine, transform rhow -> log(rhow)
    bwNN_input = np.zeros((rhow.shape[0], rhow.shape[1] + 3))
    bwNN_input[:, 0] = np.cos(sza * np.pi / 180.)  # sza
    bwNN_input[:, 1] = np.cos(oza * np.pi / 180.)  # oza
    bwNN_input[:, 2] = np.cos(nn_raa * np.pi / 180.)  # nn_raa
    bwNN_input[:, 3:] = np.log(rhow)
    bwNN_input = np.array(bwNN_input[valid, :], dtype='float32')

    ## run backwardNN
    iop_pred_bw = model_bw.predict(bwNN_input)
    ## combine bwNN[rhow] with angles
    fwNN_input = tf.concat([iop_pred_bw, bwNN_input[:, :3]], axis=1)
    ## run forwardNN
    rhow_pred = model_fw(fwNN_input)

    ## change back to full
    iop_pred_reci_out = np.ones((rhow.shape[0], 5)) * np.nan
    iop_pred_reci_out[valid, :] = iop_pred_bw

    # mae_recipr = np.ones(rhow.shape[0]) * np.nan
    # mae_recipr[valid] = mean_squared_error(rhow_pred, np.log(rhow[valid, :]))
    rhow_out_rec = np.ones(rhow.shape) * np.nan
    for j in range(rhow.shape[1]):
        rhow_out_rec[valid, j] = np.exp(rhow_pred[:, j])

    return rhow_out_rec

def apply_bwNN_OLCIscene(productPath, filename, sensor='OLCI'):

    # Initialising a product for Reading with snappy
    product = snp.ProductIO.readProduct(os.path.join(productPath, filename))
    # Get scene size
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    npix = width * height

    ### I tested it on a level 2 c2rcc product, but the code is not included here.
    #  rhow = Level2_Reader(product, sensor, band_group='reflectance', reshape=False)

    # calculate some rhow for the product.

    oaa, oza, saa, sza = angle_Reader(product, sensor)
    raa, nn_raa = calculate_diff_azim(oaa, saa)

    valid = check_valid_pixel_expression_L1(product, sensor)

    print('valid', np.sum(valid))

    # takes: rhow, sza, oza, nn_raa, linear/no transformation.
    rhow_recipr = run_bwNN_fwNN_loop(rhow, sza, oza, nn_raa, valid)

