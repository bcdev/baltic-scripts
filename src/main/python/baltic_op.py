import os
import platform
import sys
import tempfile
sys.path.append("C:\\Users\Telpecarne\.snap\snap-python")
import numpy as np
import snappy

from snappy import ProductData

import baltic_ac_algorithm
from baltic_AC_forwardNN_TF import baltic_AC


class BalticOp:
    """
    The Baltic+ AC GPF operator for OLCI L1b.

    Authors: D.Müller, C.Mazeran (breadboard); R.Shevchuk, O.Danne, 2020
    """

    def __init__(self):
        self.source_product = None
        pass

    def initialize(self, operator):
        """
        GPF initialize method
        :param operator
        :return:
        """
        resource_root = os.path.dirname(__file__)
        f = open(tempfile.gettempdir() + '/baltic_.log', 'w')

        sys.path.append(resource_root)

        f.write('Python module location: ' + __file__ + '\n')
        f.write('Python module location parent: ' + resource_root + '\n')

        print('platform.system(): ' + platform.system() + '\n')
        print('sys.version_info(): ' + str(sys.version_info) + '\n')

        # get  source product:
        #source_product = operator.getSourceProduct('l1b')
        source_product = operator.getSourceProduct('source')
        fn = source_product
        inpath = None
        if not source_product:
            raise RuntimeError('No source product specified or product not found - cannot continue.')

        #######parameters#########
        sensor = operator.getParameter('sensor')
        if sensor not in ['OLCI']:
            raise RuntimeError('Baltic_OP supports only OLCI sensors')

        outputFormat = operator.getParameter('outputFormat')
        if outputFormat not in ['BEAM-DIMAP']:
            raise RuntimeError('Currently Baltic_OP supports only BEAM-DIMAP')

        ######## Copy from breadboard
        #######
        #inpath = "E:\\work\projects\\baltic-scripts\\breadboard\\test_data"
        #fn = 'subset_S3A_OL_1_ERR____20180531T084955_20180531T093421_20180531T113749_2666_031_378______MAR_O_NR_002.dim'
        #outpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\results\\"
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

        targetProduct = baltic_AC(scene_path=inpath, filename=fn, outpath=None, sensor=sensor,
                  addName='_fwNNHL_50x40x40Noise_',
                  NNversion='TF_n',
                  outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
                  add_Idepix_Flags=True,
                  correction='HYGEOS',
                  add_c2rccIOPs=False,
                  outputProductFormat=outputFormat)


        #######
        #######
        f.write("Start initialize: source product is " + source_product.getName() + '\n')
        print('Start initialize: source product is ' + source_product.getName() + '\n')

        if 'S3A_OL' not in source_product.getName():
            raise RuntimeError('Source product does not seem to be an OLCI L1b product - cannot continue.')

        width = source_product.getSceneRasterWidth()
        height = source_product.getSceneRasterHeight()
        f.write('Source product width, height = ...' + str(width) + ', ' + str(height) + '\n')
        print('Source product width, height = ...' + str(width) + ', ' + str(height) + '\n')

        # get source bands:

        # setup target product:
        #balticac_product = snappy.Product('pyBALTICAC', 'pyBALTICAC', width, height)
        targetProduct.setDescription('Baltic+ AC product')
        targetProduct.setStartTime(source_product.getStartTime())
        targetProduct.setEndTime(source_product.getEndTime())

        # setup target bands:
        # todo
        # test: define one target band.
        # Roman: It's all done inside baltic_AC_forward
        # self.test_band = balticac_product.addBand('test_band', ProductData.TYPE_FLOAT64)
        # self.test_band.setDescription('Test band: radiance0 * 2')

        #snappy.ProductUtils.copyTiePointGrids(source_product, balticac_product)
        #source_product.transferGeoCodingTo(balticac_product, None)
        #self.baltic_ac_algo = baltic_ac_algorithm.BalticAcAlgorithm()
        #self.baltic_ac_algo.run(None)

        operator.setTargetProduct(targetProduct)
        f.write('end initialize.')
        f.close()

    def doExecute(self, pm):
        ######## Copy from breadboard
        #######
        inpath = "E:\\work\projects\\baltic-scripts\\breadboard\\test_data"
        fn = 1
        outpath = "E:\Documents\projects\Baltic+\WP3_AC\\test_data\\results\\"
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

        targetProduct = baltic_AC(scene_path=inpath, filename=fn, outpath=outpath, sensor=sensor,
                                  addName='_fwNNHL_50x40x40Noise_',
                                  NNversion='TF_n',
                                  outputSpectral=outputSpectral, outputScalar=outputScalar, niop=5,
                                  add_Idepix_Flags=True,
                                  correction='HYGEOS',
                                  add_c2rccIOPs=False,
                                  outputProductFormat='BEAM-DIMAP')
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
