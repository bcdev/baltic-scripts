#!/usr/bin/python
""" Define bands used by BALTIC+ AC """
import re
import sys
def main(sensor):

    if sensor == 'OLCI':
        bands_sat  =       [400,412,443,490,510,560,620,665,674,681,709,754,760,764,767,779,865,885,900,940,1020]
        bands_rw   =       [400,412,443,490,510,560,620,665,674,681,709,754,            779,865,885,        1020]
        bands_chi2 =       [400,412,443,490,510,560,620,665,674,681,    754,            779,865,885]#,        1020]
        bands_corr =       [400,412,443,490,510,560,620,665,674,681,    754,            779,865,885]#,        1020]
        bands_abs =        [                                                760,764,767,            900,940]
    elif sensor == 'S2MSI':
        #TODO Refine the wavelengths used in the AC
        bands_sat = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
        bands_rw  = [443, 490, 560, 665, 705, 740]
        bands_corr = [443, 490, 560, 665, 705, 740]
        bands_chi2 = [443, 490, 560, 665, 705, 740]
        bands_abs = []
        bands_sat =  [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190]
        bands_corr = [443, 490, 560, 665, 705, 740, 783,      865]
        bands_chi2 = [443, 490, 560, 665, 705, 740, 783,      865]
        bands_abs =  []
    else:
        print( "Unknown sensor %s"%sensor)
        sys.exit(1)

    return(bands_sat, bands_rw, bands_corr, bands_chi2, bands_abs)
    return(bands_sat, bands_corr, bands_chi2, bands_abs)

if __name__ == "__main__":
    import sys
    arg = ' '.join(sys.argv[1:])
    main(arg)
