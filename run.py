#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

from baltic_AC_forwardNN_TF import baltic_AC

def main(args=sys.argv[1:]):
    if len(args) != 1:
        print("usage: baltic_AC_simple <SENSOR>")
        sys.exit(1)

    sensor = args[0]

    # Define i/o
    current_path = os.path.dirname(__file__)
    inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test/'
    outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test_output/'

    ###
    # Insitu Helsinki Lighthouse
    fnames = ['subset_S3A_OL_1_ERR___20160520T090131_Helsinki.dim',
              'subset_S3A_OL_1_ERR___20160621T091456_Helsinki.dim',
              'subset_S3A_OL_1_ERR___20160713T090052_Helsinki.dim',
              'subset_S3A_OL_1_ERR___20160729T084649_Helsinki.dim']
    # subset = [(139 - 4, 139 + 4, 823 - 4, 823 + 4), (126 - 4, 126 + 4, 449 - 4, 449 + 4),
    #           (164 - 4, 164 + 4, 823 - 4, 823 + 4), (75 - 4, 75 + 4, 637 - 4, 637 + 4)]
    subset = [(139 - 1, 139 + 1, 823 - 1, 823 + 1), (126 - 1, 126 + 1, 449 - 1, 449 + 1),
              (164 - 1, 164 + 1, 823 - 1, 823 + 1), (75 - 1, 75 + 1, 637 - 1, 637 + 1)]


    # Define output fields
    outputSpectral = {
        'rho_toa': 'rho_toa',
        'rho_ng': 'rho_ng',
        'rho_gc': 'rho_gc',
        'rho_r': 'rho_r',
        'rho_rc': 'rho_rc',
        # 'rho_molgli' : 'rho_molgli', # for HYGEOS Rayleigh correction only
        # 'rho_ag': 'rho_ag',
        # 'rho_ag_mod': 'rho_ag_mod',
        # 'td': 'td',
        #  'tau_r': 'tau_r',
        # 'tau_r_SRF': 'tau_r_SRF',
        # 'rho_w': 'rho_w',               # AC output
        # 'rho_wmod': 'rho_wmod',         # AC output
        # 'rho_wn': 'rho_wn'              # AC output
    }
    outputScalar = {
        # 'iteration' : 'iterations',
        # 'chi2_AC': 'chi2_AC'
        # 'log_apig': 'log_iop[:,0]',     # AC output
        # 'log_adet': 'log_iop[:,1]',     # AC output
        # 'log_agelb': 'log_iop[:,2]',    # AC output
        # 'log_bpart': 'log_iop[:,3]',    # AC output
        # 'log_bwit': 'log_iop[:,4]'      # AC output
        # 'raa': 'raa'
    }

    # Launch the AC
    for i, fn in enumerate(fnames[:]):
        print(fn)
        baltic_AC(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, platform='S3A',
                  addName='_fwNNHL_50x40x40_test',
                  NNversion='baltic+_v1', NNIOPversion='c2rcc_20171221',
                  outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
                  add_Idepix_Flags=True,
                  correction='HYGEOS',
                  add_c2rccIOPs=False,
                  outputProductFormat='BEAM-DIMAP')
        # baltic_AC(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, platform='S2A',
        #           addName='_fwNNHL_50x40x40_test',
        #           atmosphericAuxDataPath="E:\Documents\projects\Baltic+\WP3_AC\\baltic-scripts\S2_atmosphericAux\\",
        #           NNversion='baltic+_v1', NNIOPversion='std_s2_20160502',
        #           outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
        #           add_Idepix_Flags=False,
        #           correction='HYGEOS',
        #           add_c2rccIOPs=False,
        #           outputProductFormat='BEAM-DIMAP')



if __name__ == '__main__':
    main()
