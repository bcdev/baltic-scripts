#!/usr/bin/python


""" Define bands used by BALTIC+ AC """

import re
import sys

def main(sensor,proc):

    if sensor == 'OLCI':
        bands_sat  = [400,412,443,490,510,560,620,665,674,681,709,754,760,764,767,779,865,885,900,940,1020]
        bands_rw   = [400,412,443,490,510,560,620,665,674,681,709,754,            779,865,885,        1020]
        bands_corr = [    412,443,490,510,560,620,665,674,681,    754] # forwardNN but 400 and 709
        #bands_corr = [    412,443,490,510,560,620,665,            754,            779,865                 ] # POLYMER
        #bands_corr = [400,412,443,490,510,560,620,665,674,681,709,754,            779,865,            1020] #C2RCC
    #elif sensor == 'S2' #TODO
    else:
        print( "Unknown sensor %s"%sensor)
        sys.exit(1)

    return(bands_sat, bands_rw, bands_corr)

if __name__ == "__main__":
    import sys
    arg = ' '.join(sys.argv[1:])
    main(arg)


