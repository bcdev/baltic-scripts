import os
import platform
import sys
import tempfile
import numpy as np
import snappy
from snappy import GPF
from snappy import jpy
from snappy import ProgressMonitor
from snappy import ProductData

import baltic_ac_algorithm
from baltic_AC_forwardNN_TF import baltic_AC


class BalticOp:
    """
    The Baltic+ AC GPF operator for OLCI L1b.

    Authors: D.Müller, C.Mazeran (breadboard); R.Shevchuk, O.Danne, 2020
    """

    def __init__(self):
        self.sourceProduct = None
        self.targetProduct = None
        self.format = "BEAM-DIMAP"
        self.outputPath = None
        pass

    def initialize(self, operator):
        """
        GPF initialize method
        :param operator
        :return:
        """
        print('start initializng...')
        resource_root = os.path.dirname(__file__)
        f = open(tempfile.gettempdir() + '/baltic_.log', 'w')

        sys.path.append(resource_root)

        f.write('Python module location: ' + __file__ + '\n')
        f.write('Python module location parent: ' + resource_root + '\n')

        print('platform.system(): ' + platform.system() + '\n')
        print('sys.version_info(): ' + str(sys.version_info) + '\n')

        # get  source product:
        sourceProduct = operator.getSourceProduct('sourceProduct')

        if not sourceProduct:
            raise RuntimeError('No source product specified or product not found - cannot continue.')
        if 'S3A_OL' not in sourceProduct.getName():
            raise RuntimeError('Source product does not seem to be an OLCI L1b product - cannot continue.')

        #######
        f.write("Start initialize: source product is " + sourceProduct.getName() + '\n')
        print('Start initialize: source product is ' + sourceProduct.getName() + '\n')


        width = sourceProduct.getSceneRasterWidth()
        height = sourceProduct.getSceneRasterHeight()
        f.write('Source product width, height = ...' + str(width) + ', ' + str(height) + '\n')
        print('Source product width, height = ...' + str(width) + ', ' + str(height) + '\n')
        #######
        sensor = "OLCI"
        outputSpectral = {'rho_toa': 'rho_toa',
                          'rho_w': 'rho_w',
                          'rho_wmod': 'rho_wmod',
                          'rho_wn': 'rho_wn'
                          }
        outputScalar = {'log_apig': 'log_iop[:,0]',
            'log_adet': 'log_iop[:,1]',
            'log_agelb': 'log_iop[:,2]',
            'log_bpart': 'log_iop[:,3]',
            'log_bwit': 'log_iop[:,4]'
        }
        self.format = operator.getParameter('format')
        self.outputPath = operator.getParameter('outputPath')
        targetProduct = baltic_AC(sourceProduct=sourceProduct, sensor='OLCI',
                                  addName='_fwNNHL_50x40x40Noise_',
                                  NNversion='TF_n',
                  outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
                  add_Idepix_Flags=True,
                  correction='HYGEOS',
                  add_c2rccIOPs=False)


        File = jpy.get_type('java.io.File')
        GPF.writeProduct(targetProduct, File(self.outputPath), self.format, False, ProgressMonitor.NULL)



        f.write('end initialize.')
        f.close()
        operator.setTargetProduct(targetProduct)


    def doExecute(self, pm):
        ####### commented for now, since all is done in initialize
        pass

    def dispose(self, operator):
        """
        The GPF dispose method. Nothing to do here.
        :param operator:
        :return:
        """
        pass

    @staticmethod
    def get_band(input_product, band_name):
        """
        Gets band from input product by name
        :param input_product
        :param band_name
        :return:
        """
        band = input_product.getBand(band_name)
        if not band:
            band = input_product.getTiePointGrid(band_name)
            if not band:
                raise RuntimeError('Product has no band or tpg with name', band_name)
        return band
