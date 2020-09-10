#!/usr/bin/env python
import numpy as np
import os
from scipy import ndimage

default_ADF = {
        'OLCI': { 
            'S3A': {
                'file_adf_ppp': os.path.join('auxdata',
                    'S3A_OL_2_PPP_AX_20160216T000000_20991231T235959_20190109T120000___________________MPC_O_AL_005.SEN3',
                    'OL_2_PPP_AX.nc'),
                'file_adf_acp': os.path.join('auxdata',
                    'S3A_OL_2_ACP_AX_20160216T000000_20991231T235959_20190125T120000___________________MPC_O_AL_004.SEN3',
                    'OL_2_ACP_AX.nc'),
                'file_adf_clp': os.path.join('auxdata',
                    'S3A_OL_2_CLP_AX_20160216T000000_20991231T235959_20170210T120000___________________MPC_O_AL_003.SEN3',
                    'OL_2_CLP_AX.nc'),
                'SRF_wavelength': os.path.join('auxdata', 'SRF_OLCI_S3A_wavelength_table_perBand.txt'),
                'SRF_weights': os.path.join('auxdata', 'SRF_OLCI_S3A_weights_table_perBand.txt')
                },
            'S3B': { #TODO
                 'file_adf_ppp': os.path.join('auxdata',
                    'S3A_OL_2_PPP_AX_20160216T000000_20991231T235959_20190109T120000___________________MPC_O_AL_005.SEN3',
                    'OL_2_PPP_AX.nc'),
                'file_adf_acp': os.path.join('auxdata',
                    'S3A_OL_2_ACP_AX_20160216T000000_20991231T235959_20190125T120000___________________MPC_O_AL_004.SEN3',
                    'OL_2_ACP_AX.nc'),
                'file_adf_clp': os.path.join('auxdata',
                    'S3A_OL_2_CLP_AX_20160216T000000_20991231T235959_20170210T120000___________________MPC_O_AL_003.SEN3',
                    'OL_2_CLP_AX.nc'),
                'SRF_wavelength': os.path.join('auxdata', 'SRF_OLCI_S3B_wavelength_table_perBand.txt'),
                'SRF_weights': os.path.join('auxdata', 'SRF_OLCI_S3B_weights_table_perBand.txt')
                }
            },

        'S2MSI': { # TODO, currently copy S3A for ADF
             'S2A': {
                'file_adf_ppp': os.path.join('auxdata',
                    'S3A_OL_2_PPP_AX_20160216T000000_20991231T235959_20190109T120000___________________MPC_O_AL_005.SEN3',
                    'OL_2_PPP_AX.nc'),
                'file_adf_acp': os.path.join('auxdata',
                    'S3A_OL_2_ACP_AX_20160216T000000_20991231T235959_20190125T120000___________________MPC_O_AL_004.SEN3',
                    'OL_2_ACP_AX.nc'),
                'file_adf_clp': os.path.join('auxdata',
                    'S3A_OL_2_CLP_AX_20160216T000000_20991231T235959_20170210T120000___________________MPC_O_AL_003.SEN3',
                    'OL_2_CLP_AX.nc'),
                'SRF_wavelength': os.path.join('auxdata', 'SRF_MSI_S2A_wavelength_table_perBand.txt'),
                'SRF_weights': os.path.join('auxdata', 'SRF_MSI_S2A_weights_table_perBand.txt')
                },
            'S2B': {
                 'file_adf_ppp': os.path.join('auxdata',
                    'S3A_OL_2_PPP_AX_20160216T000000_20991231T235959_20190109T120000___________________MPC_O_AL_005.SEN3',
                    'OL_2_PPP_AX.nc'),
                'file_adf_acp': os.path.join('auxdata',
                    'S3A_OL_2_ACP_AX_20160216T000000_20991231T235959_20190125T120000___________________MPC_O_AL_004.SEN3',
                    'OL_2_ACP_AX.nc'),
                'file_adf_clp': os.path.join('auxdata',
                    'S3A_OL_2_CLP_AX_20160216T000000_20991231T235959_20170210T120000___________________MPC_O_AL_003.SEN3',
                    'OL_2_CLP_AX.nc'),
                'SRF_wavelength': os.path.join('auxdata', 'SRF_MSI_S2A_wavelength_table_perBand.txt'),
                'SRF_weights': os.path.join('auxdata', 'SRF_MSI_S2A_weights_table_perBand.txt')
                }
            },

        'GENERIC': {
            'file_HYGEOS': os.path.join('auxdata','LUT.hdf')
            }
    }

def nlinear(x, yp, axes):
    '''
    multidimensional LUT interpolation
    (interpolate n-dimensional LUT over array of m dimensions)
        x: iterable of n * (scalar or m-dim indices)
        yp: the n-dim array to interpolate
        axes: list of m axes
    '''
    assert [len(ax) for ax in axes] == list(yp.shape), 'incompatible axes'
    coords = []
    for i, xi in enumerate(x):
        coords.append(np.interp(xi, axes[i], np.arange(len(axes[i]))))
    coords = np.array(*np.broadcast_arrays(coords))

    return ndimage.map_coordinates(yp, coords, order=1)


def scale_minmax(x, input_range, reverse=False):
    if reverse:
        for j in range(x.shape[1]):
            diff = input_range[j, 1] - input_range[j, 0]
            x[:, j] = x[:,j]*diff + input_range[j, 0]
    else:
        for j in range(x.shape[1]):
            diff = input_range[j, 1] - input_range[j, 0]
            x[:, j] = (x[:,j]- input_range[j, 0])/diff
    return x
