#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append("C:\\Users\Telpecarne\.snap\snap-python")

from baltic_AC_forwardNN_TF import baltic_AC

def main(args=sys.argv[1:]):
    if len(args) != 1:
        print("usage: baltic_AC_simple <SENSOR>")
        sys.exit(1)
    sensor = args[0]

    # Define i/o
    current_path = os.path.dirname(__file__)
    # inpath = os.path.join(current_path, 'test_data')
    # outpath = '/home/cmazeran/Documents/solvo/Projets/ESRIN/BALTIC+/data'

    inpath = "E:\\work\projects\\baltic-scripts\\breadboard\\test_data"
    #inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test/'
    outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test_output/'

    # inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_baltic_test/'
    # outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_baltic_test_output/'

    # Input data - use SNAP with option S3TBX pixelGeoCoding turned off!
    # fnames = os.listdir(path)
    # fnames = [fn for fn in fnames if '.dim' in fn]
    # fnames = ['subset_S3A_OL_1_ERR____20180531T084955_20180531T093421_20180531T113749_2666_031_378______MAR_O_NR_002.dim']
    # subset = (20,70,50,200) # (start_line, end_line, start_col, end_col) or None

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

    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\AC_Helsinki_L2std\AC_Helsinki_L2std_extracts\\"
    # fnames = ["pixEx_S3A_L1__OL_1_ERR_measurements_forSNAPinut2test.csv"]

    inpath = "E:\Documents\projects\Baltic+\WP3_AC\Insitu_data\AERONET-OC\Palgrunden\OLCI_extracts\\"
    outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Palgrunden_test_output/'
    # fnames = ['OLCI_Palgrunden_extraction_all_SNAPreadable2.txt']
    fnames = ['OLCI_Palgrunden_extraction_all_SNAPreadable2_test.csv'] #corrected flag names.

    ###
    # RAyleigh correction only:
    # inpath = "E:\Documents\projects\IdePix\pixbox\S3-OLCI\collection2018_rBRR\\"
    # outpath = inpath
    # fnames = ['pixEx_OLCI2018__OL_1_EFR_measurements_subset_SNAPreadable.csv',
    #           "pixEx_OLCI2018_20170315_SNAPreadable.csv",
    #           "pixEx_OLCI2018_again__OL_1_EFR_SNAPreadable.csv"]

    ##
    # Rayleigh test:
    # inpath = "E:\Documents\projects\S3MPC\Rayleigh\Test_SNAP5_SNAP7\\"
    # outpath = inpath
    # fnames = ['subset_0_of_S3A_OL_1_EFR____20160802T103602_20160802T103902_20180226T144340_0179_007_108_1980_LR2_R_NT_002.dim']
    # inpath = "E:\Documents\projects\S3MPC\Rayleigh\Test_SNAP5_SNAP7\MarcExample\\"
    # outpath = inpath
    # fnames = ['S3A_OL_1_EFR____20190605T230509_20190605T230809_20190606T005515_0179_045_272_3600_MAR_O_NR_002.dim']

    ###
    # Insitu Sampsa
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\Sampsa_Insitu_20170814\\"
    # outpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\Sampsa_Insitu_20170814_output\\"
    # fnames = ['subset_S3A_OL_1_EFR____20170814T092115.dim', 'subset_S3A_OL_1_EFR____20170814T092115.dim',
    #           'subset_S3A_OL_1_EFR____20170814T092115.dim', 'subset_S3A_OL_1_EFR____20170814T092115.dim',
    #           'subset_S3A_OL_1_EFR____20170814T092115.dim', 'subset_S3A_OL_1_EFR____20170814T092115.dim',
    #           'subset_S3A_OL_1_EFR____20170814T092115.dim']
    # subset = [(492-4, 492 +4, 144-4, 144+4), (500-4, 500 +4, 139-4, 139+4),
    #           (517-4, 517 +4, 134-4, 134+4), (558-4, 558 +4, 127-4, 127+4),
    #           (605-4, 605 +4, 121-4, 121+4), (631-4, 631 +4, 106-4, 106+4),
    #           (583 - 4, 583 + 4, 105 - 4, 105 + 4)]
    # delta_old=4
    # delta = 1
    # for i in range(len(subset)):
    #     x1, x2, y1, y2 = subset[i]
    #     x1 = x1 + delta_old -delta
    #     y1 = y1 + delta_old - delta
    #     x2 = x2 - delta_old + delta
    #     y2 = y2 - delta_old + delta
    #     subset[i] = (x1, x2, y1, y2)


    ###
    # FR subset 20190415 + 20190418
    # inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/'
    # outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/FR_output_test'
    # fnames = ['subset_0_of_S3A_OL_1_EFR____20190415T092940_20190415T093240_20190415T112537_0179_043_307_1800_LN1_O_NR_002.dim',
    # 		   'subset_0_of_S3A_OL_1_EFR____20190418T095207_20190418T095507_20190419T143029_0179_043_350_1800_LN1_O_NT_002.dim']
    # subset = [(59, 79, 0, 230)]

    # fnames = ['subset_Baltic_S3A_OL_1_ERR____20190415T092333.dim', 'subset_Baltic_S3B_OL_1_ERR____20190418T090611.dim',
    #          'subset_baltic_S3B_OL_1_ERR____20190415T084404.dim', 'subset_Baltic_S3A_OL_1_ERR____20190418T094541.dim',
    #          'subset_NorthSea_S3A_OL_1_ERR____20190418T094541.dim']

    inpath = "E:\\work\projects\\baltic-scripts\\breadboard\\test_data"
    #inpath = "E:\Documents\projects\Baltic+\WP3_AC\\baltic-scripts\\test_data\\"
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\"
    outpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\results\\"
    # fnames = ["subset_0_of_S3A_OL_1_EFR____20190415T092940_20190415T093240_20190415T112537_0179_043_307_1800_LN1_O_NR_002.dim"]
    # fnames = ['subset_S3A_OL_1_EFR____20180531T090233_20180531T090533_20180601T124046_0179_031_378_1980_LN1_O_NT_002.dim']
    fnames = ['subset_S3A_OL_1_ERR____20180531T084955_20180531T093421_20180531T113749_2666_031_378______MAR_O_NR_002.dim']
    #
    #
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\OLCI_testscenes\\ROI\\"
    # outpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\OLCI_testscenes\\L2\\"
    #
    # # fnames = os.listdir(inpath)
    # # fnames = [fn for fn in fnames if '.dim' in fn]
    # fnames = ["subset_Pori_S3A_OL_1_ERR____20180602T093828_20180602T102254_20180603T144028_2666_032_022______LN1_O_NT_002.dim"]
    # fnames = ["subset_GulfBothnia_S3A_OL_1_ERR____20180602T093828_20180602T102254_20180603T144028_2666_032_022______LN1_O_NT_002.dim"]
    # fnames =["subset_RiverDischarge_S3B_OL_1_ERR____20190415T084404_20190415T092817_20190416T132956_2653_024_164______LN1_O_NT_002.dim"]


    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\S3A_OL_1_EFR____20160503T081354_20160503T081654_20180207T105015_0179_003_349_1980_LR2_R_NT_002.SEN3\\"
    # fnames = ["xfdumanifest.xml"]


    ###
    # CIWAWA OLCI test
    # inpath = "E:\Documents\projects\CIWAWA\\testdata\\"
    # outpath = inpath
    # fnames = ['subset_S3A_OL_1_EFR____20180505T101722.dim',
    #           'subset_S3A_OL_1_EFR____20180803T094342.dim',
    #           'subset_S3A_OL_1_EFR____20190423T092513.dim']
    # subset = [(299,381, 514, 650), (271, 351, 420, 552), (374, 480, 388, 520)] # (start_line, end_line, start_col, end_col)
    #subset = [(374, 480, 388, 520)]
    # Define output fields
    outputSpectral = {
        'rho_toa': 'rho_toa',
        # 'rho_ng': 'rho_ng',
        # 'rho_gc': 'rho_gc',
        # 'rho_r': 'rho_r',
        # 'rho_rc': 'rho_rc',
        # 'rho_molgli' : 'rho_molgli',
        # 'rho_ag': 'rho_ag',
        # 'rho_ag_mod': 'rho_ag_mod',
        # 'td': 'td',
        #  'tau_r': 'tau_r',
        # 'tau_r_SRF': 'tau_r_SRF',
        'rho_w': 'rho_w',
        'rho_wmod': 'rho_wmod',
        'rho_wn': 'rho_wn'
    }
    outputScalar = {
                # 'iteration' : 'iterations',
        # 'chi2_AC': 'chi2_AC'
        'log_apig': 'log_iop[:,0]',
        'log_adet': 'log_iop[:,1]',
        'log_agelb': 'log_iop[:,2]',
        'log_bpart': 'log_iop[:,3]',
        'log_bwit': 'log_iop[:,4]'
        # 'raa': 'raa'
    }



    # Launch the AC
    for i, fn in enumerate(fnames[:]):
        print(fn)

        ## forwardNN TF
        # baltic_AC(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor,
        #           addName='_fwNNHL_50x40x40_',
        #           NNversion='TF',
        #           outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
        #           add_Idepix_Flags=False,
        #           correction='HYGEOS',
        #           add_c2rccIOPs=False,
        #           outputProductFormat='CSV')

        baltic_AC(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor,
                  # addName='_fwNNHL_50x40x40Noise_',
                  # NNversion='TF_n',
                  addName='_fwNNHL_50x40x40Noise_',
                  NNversion='TF_n',
                  outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
                  add_Idepix_Flags=True,
                  correction='HYGEOS',
                  add_c2rccIOPs=False,
                  outputProductFormat='BEAM-DIMAP')

        # ## Rayleigh + gaseous correction only!!
        # baltic_AC(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor,
        #           # addName='_fwNNHL_50x40x40_',
        #           addName='_rBRR_',
        #           NNversion='TF',
        #           outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
        #           add_Idepix_Flags=False,
        #           correction='HYGEOS',
        #           add_c2rccIOPs=False,
        #           runAC=False, copyOriginalProduct=True,  # for Rayleigh Correction on CSV data.
        #           outputProductFormat='BEAM-DIMAP') #CSV


        ## OLCI matchup data.
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_fwNN97c2rcc_3x3_startVal1_niop5',
        #                     outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS', NNversion= 'TF', niop=5,
        #                     subset=subset[i],
        #                     outputProductFormat="CSV", add_Idepix_Flags=False)




if __name__ == '__main__':
    main()
