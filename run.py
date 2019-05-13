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
    inpath = os.path.join(current_path,'test_data')
    outpath = '/home/cmazeran/Documents/solvo/Projets/ESRIN/BALTIC+/data'

    inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test/'
    outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_Helsinki_test_output/'

    inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_baltic_test/'
    outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/AC_baltic_test_output/'

    # Input data - use SNAP with option S3TBX pixelGeoCoding turned off!
    #fnames = os.listdir(path)
    #fnames = [fn for fn in fnames if '.dim' in fn]
    #fnames = ['subset_S3A_OL_1_ERR____20180531T084955_20180531T093421_20180531T113749_2666_031_378______MAR_O_NR_002.dim']
    #subset = (20,70,50,200) # (start_line, end_line, start_col, end_col) or None

    ###
    # Insitu Helsinki Lighthouse
    #fnames = ['subset_S3A_OL_1_ERR___20160520T090131_Helsinki.dim', 'subset_S3A_OL_1_ERR___20160621T091456_Helsinki.dim',
    #          'subset_S3A_OL_1_ERR___20160713T090052_Helsinki.dim', 'subset_S3A_OL_1_ERR___20160729T084649_Helsinki.dim']
    #subset = [(139-4, 139+4, 823-4, 823 +4), (126-4, 126+4, 449-4, 449 +4),
    #          (164-4, 164+4, 823-4, 823 +4), (75-4, 75+4, 637-4, 637 +4)]

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
    inpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/'
    outpath = 'E:/Documents/projects/Baltic+/WP3_AC/test_data/FR_output_test'
    fnames = ['subset_0_of_S3A_OL_1_EFR____20190415T092940_20190415T093240_20190415T112537_0179_043_307_1800_LN1_O_NR_002.dim',
              'subset_0_of_S3A_OL_1_EFR____20190418T095207_20190418T095507_20190419T143029_0179_043_350_1800_LN1_O_NT_002.dim']



    #fnames = ['subset_Baltic_S3A_OL_1_ERR____20190415T092333.dim', 'subset_Baltic_S3B_OL_1_ERR____20190418T090611.dim',
    #          'subset_baltic_S3B_OL_1_ERR____20190415T084404.dim', 'subset_Baltic_S3A_OL_1_ERR____20190418T094541.dim',
    #          'subset_NorthSea_S3A_OL_1_ERR____20190418T094541.dim']

    for i,fn in enumerate(fnames[:]):
        print(fn)
        #baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, subset=subset[i],addName=str(i)+'_Forwardc2rccv2')
        baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, addName='_Forwardc2rccv2')

if __name__ == '__main__':
    main()
