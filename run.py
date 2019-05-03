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

    # Input data - use SNAP with option S3TBX pixelGeoCoding turned off!
    #fnames = os.listdir(path)
    #fnames = [fn for fn in fnames if '.dim' in fn]
    fnames = ['subset_S3A_OL_1_ERR____20180531T084955_20180531T093421_20180531T113749_2666_031_378______MAR_O_NR_002.dim']
    subset = (20,70,50,200) # (start_line, end_line, start_col, end_col) or None

    for fn in fnames:
        print(fn)
        baltic_AC_forwardNN(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor, subset=subset)

if __name__ == '__main__':
    main()
