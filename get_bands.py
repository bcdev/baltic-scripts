#!/usr/bin/python


""" Define bands used by BALTIC+ AC """

import re
import sys

def main(sensor,proc):

    if sensor == 'OLCI':
        bands_sat  = [400,412,443,490,510,560,620,665,674,681,709,754,760,764,767,779,865,885,900,940,1020]
        bands_rw   = [400,412,443,490,510,560,620,665,674,681,709,754,            779,865,885,        1020]
        bands_corr = [    412,443,490,510,560,620,665,674,681,    754] # forwardNN but 400 and 709
        bands_chi2 = [400,412,443,490,510,560,620,665,674,681,    754] # forwardNN but 709
        bands_forwardNN = [400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709, 754]
        bands_abs = [760, 764, 767, 900, 940]
        #bands_corr = [    412,443,490,510,560,620,665,            754,            779,865                 ] # POLYMER
        #bands_corr = [400,412,443,490,510,560,620,665,674,681,709,754,            779,865,            1020] #C2RCC
    elif sensor == 'S2':
        #TODO add the missing wavelengths
        bands_sat = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
        bands_rw  = []
        bands_corr = []
        bands_forwardNN = [443, 490, 560, 665, 705, 740]
        bands_abs = []
    else:
        print( "Unknown sensor %s"%sensor)
        sys.exit(1)

    return(bands_sat, bands_rw, bands_corr, bands_chi2, bands_forwardNN, bands_abs)

if __name__ == "__main__":
    import sys
    arg = ' '.join(sys.argv[1:])
    main(arg)


