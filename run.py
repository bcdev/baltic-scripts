#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys

from baltic_AC_forwardNN import baltic_AC_forwardNN


def main(args=sys.argv[1:]):
    if len(args) != 1:
        print("usage: baltic_AC_simple <SENSOR>")
        sys.exit(1)
    sensor = args[0]

    # Define i/o
    current_path = os.path.dirname(__file__)
    inpath = os.path.join(current_path, 'test_data')
    outpath = '/home/cmazeran/Documents/solvo/Projets/ESRIN/BALTIC+/data'

    inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test/'
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
    # fnames = ['subset_S3A_OL_1_ERR___20160520T090131_Helsinki.dim',
    #           'subset_S3A_OL_1_ERR___20160621T091456_Helsinki.dim',
    #           'subset_S3A_OL_1_ERR___20160713T090052_Helsinki.dim',
    #           'subset_S3A_OL_1_ERR___20160729T084649_Helsinki.dim']
    # subset = [(139 - 4, 139 + 4, 823 - 4, 823 + 4), (126 - 4, 126 + 4, 449 - 4, 449 + 4),
    #           (164 - 4, 164 + 4, 823 - 4, 823 + 4), (75 - 4, 75 + 4, 637 - 4, 637 + 4)]

    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\AC_Helsinki_L2std\AC_Helsinki_L2std_extracts\\"
    # fnames = ["pixEx_S3A_L1__OL_1_ERR_measurements_forSNAPinut2test.csv"]

    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\Insitu_data\AERONET-OC\Palgrunden\OLCI_extracts\\"
    # outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Palgrunden_test_output/'
    # fnames = ['OLCI_Palgrunden_extraction_all_SNAPreadable2.txt']
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

    inpath = "E:\Documents\projects\Baltic+\WP3_AC\\baltic-scripts\\test_data\\"
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\"
    outpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\results\\"
    # fnames = ["subset_0_of_S3A_OL_1_EFR____20190415T092940_20190415T093240_20190415T112537_0179_043_307_1800_LN1_O_NR_002.dim"]
    # fnames = ['subset_S3A_OL_1_EFR____20180531T090233_20180531T090533_20180601T124046_0179_031_378_1980_LN1_O_NT_002.dim']
    fnames = ['subset_S3A_OL_1_ERR____20180531T084955_20180531T093421_20180531T113749_2666_031_378______MAR_O_NR_002.dim']

    ###
    # Rayleigh tests:
    #inpath = "E:\Documents\projects\S3MPC\OLCI_smile_correction\\rayleigh_tests\\"
    #inpath = "E:\Documents\projects\S3MPC\data\\"
    #outpath = "E:\Documents\projects\S3MPC\OLCI_smile_correction\\rayleigh_tests\\"
   # fnames = ["subset_S3A_OL_1_ERR___20190605T085717_Alps.dim", "subset_S3A_OL_1_ERR___20190605T085717_Marocco.dim"]
    #fnames = ["subset_S3A_OL_1_ERR___20190717T130807_RioDeLaPlata.dim"]

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
       #  'rho_ng': 'rho_ng',
       # # 'rho_gc': 'rho_gc',
       # # 'rho_r': 'rho_r',
       #  'rho_rc': 'rho_rc',
        # 'rho_ag': 'rho_ag',
        # 'rho_ag_mod': 'rho_ag_mod',
        #'td': 'td',
        #'tau_r': 'tau_r'
        #  'rho_w': 'rho_w',
        #  'rho_wmod': 'rho_wmod',
        # 'rho_wn': 'rho_wn'
    }
    outputScalar = {
        # 'log_apig': 'iop[:,0]',
        # 'log_adet': 'iop[:,1]',
        # 'log_agelb': 'iop[:,2]',
        # 'log_bpart': 'iop[:,3]',
        # 'log_bwit': 'iop[:,4]',
        # 'raa': 'raa'
    }

    # for S2 test:
    ## needs resampling
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\"
    # fnames = ['S2A_MSIL1C_20180526T100031_N0206_R122_T34VDK_20180526T134414.SAFE',
    #           'S2A_MSIL1C_20180529T101031_N0206_R022_T34VDL_20180529T110633.SAFE']
    ## resampled data
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\AC_S2_test\\"
    # fnames = ["S2A_MSIL1C_20180526T100031_N0206_R122_T34VDK_20180526T134414_s2resampled.dim"]
    # path_auxdata_repository = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\AC_S2_test\AUX_DATA\\"
    #path_auxdata_repository = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\AC_S2_test\\auxdata_test\\"

    ## matchup extraction including auxdata.
    # inpath = "E:\Documents\projects\Baltic+\WP3_AC\Insitu_data\AERONET-OC\\"
    # fnames = ["baltic_S2_extracts_AERONETOC_SNAPproduct.txt"]
    #
    #
    # outpath = 'E:\Documents\projects\Baltic+\WP3_AC\\test_data\AC_S2_test\\'
    # outputSpectral = {
    #     'rho_toa': 'rho_toa',
    #     'rho_w': 'rho_w',
    #     'rho_wmod': 'rho_wmod',
    #     'rho_wn': 'rho_wn'
    # }
    # outputScalar = {
    #     'sza': 'sza',
    #     'saa': 'saa',
    #     'oza': 'oza',
    #     'oaa': 'oaa'
    # }

    # Launch the AC
    for i, fn in enumerate(fnames[:1]):
        print(fn)
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, subset=subset[i],
        #                     addName='_FNNv2constr', outputSpectral=outputSpectral, outputScalar=outputScalar,
        #                     correction='IPF')
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_FNNv2constr_IPF',
        #         outputSpectral=outputSpectral, outputScalar=outputScalar, correction='IPF')
        baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_withIdePix',
                             outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS')
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor,
        #                     addName='_RayleighOnlyAltitude',
        #                     outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS')
        #baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, subset=subset[i], addName='_FNNv2constrHighChl_IPF',
        #        outputSpectral=outputSpectral, outputScalar=outputScalar, correction='IPF')
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, subset=subset[i], addName='_FNNv2constr_HYGEOS2',
        #                     outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS')

        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_outTest_HYGEOS_v1',
        #                     outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS', copyOriginalProduct=True, outputProductFormat="BEAM-DIMAP")
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor,
        #                     addName='_FNNv2constr_IPF_csv_azimuth3',
        #                     outputSpectral=outputSpectral, outputScalar=outputScalar, correction='IPF',
        #                     copyOriginalProduct=True, outputProductFormat="BEAM-DIMAP")

        ## OLCI matchup data.
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_NN20190414',
        #                      outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS', outputProductFormat="BEAM-DIMAP")
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_NN20171221test',
        #                     outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS',
        #                     outputProductFormat="CSV")


        ## S2 scenes:
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_S2test',
        #                      outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS',
        #                     copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP", atmosphericAuxDataPath = path_auxdata_repository)
        ## S2 matchup data
        # baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_S2test20190828',
        #                      outputSpectral=outputSpectral, outputScalar=outputScalar, correction='HYGEOS',
        #                     copyOriginalProduct=False, outputProductFormat="BEAM-DIMAP", atmosphericAuxDataPath = '')

if __name__ == '__main__':
    main()
