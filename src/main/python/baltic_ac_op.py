import os
import platform
import sys
import tempfile

import numpy as np
import snappy

from src.main.python import baltic_ac_algorithm


class BalticAcOp:
    """
    The Baltic+ AC GPF operator for OLCI L1b.

    Authors: D.Müller, C.Mazeran (breadboard); R.Shevchuk, O.Danne, 2020
    """

    def __init__(self):
        pass

    def initialize(self, operator):
        """
        GPF initialize method
        :param operator
        :return:
        """
        resource_root = os.path.dirname(__file__)
        f = open(tempfile.gettempdir() + '/balticac.log', 'w')

        sys.path.append(resource_root)

        f.write('Python module location: ' + __file__ + '\n')
        f.write('Python module location parent: ' + resource_root + '\n')

        print('platform.system(): ' + platform.system() + '\n')
        print('sys.version_info(): ' + str(sys.version_info) + '\n')

        # get L1b source product:
        source_product = operator.getSourceProduct('l1b')
        if not source_product:
            raise RuntimeError('No source product specified or product not found - cannot continue.')

        f.write("Start initialize: source product is " + source_product.getName() + '\n')
        print('Start initialize: source product is ' + source_product.getName() + '\n')

        if 'S3A_OL' not in source_product.getName():
            raise RuntimeError('Source product does not seem to be an OLCI L1b product - cannot continue.')

        width = source_product.getSceneRasterWidth()
        height = source_product.getSceneRasterHeight()
        f.write('Source product width, height = ...' + str(width) + ', ' + str(height) + '\n')
        print('Source product width, height = ...' + str(width) + ', ' + str(height) + '\n')

        # get source bands:
        self.sza_band = self.get_band(source_product, 'SZA')
        self.oza_band = self.get_band(source_product, 'OZA')

        self.rad_bands = []
        self.lambda0_bands = []
        self.fwhm_bands = []
        self.solar_flux_bands = []
        for b in range(1, 21):
            self.rad_bands.append(self.get_band(source_product, 'Oa%02d' % b + '_radiance'))
            self.lambda0_bands.append(self.get_band(source_product, 'lambda0_band_' + str(b)))
            self.fwhm_bands.append(self.get_band(source_product, 'FWHM_band_' + str(b)))
            self.solar_flux_bands.append(self.get_band(source_product, 'solar_flux_band_' + str(b)))

        # setup target product:
        o2corr_product = snappy.Product('pyBALTICAC', 'pyBALTICAC', width, height)
        o2corr_product.setDescription('O2 correction product')
        o2corr_product.setStartTime(source_product.getStartTime())
        o2corr_product.setEndTime(source_product.getEndTime())

        # setup target bands:
        # todo

        snappy.ProductUtils.copyTiePointGrids(source_product, o2corr_product)
        source_product.transferGeoCodingTo(o2corr_product, None)

        operator.setTargetProduct(o2corr_product)

        self.baltic_ac_algo = baltic_ac_algorithm.BalticAcAlgorithm()

        f.write('end initialize.')
        f.close()

    def compute(self, operator, target_tiles, target_rectangle):
        """
        GPF compute method
        :param operator
        :param target_tiles
        :param target_rectangle
        :return:
        """
        print('enter compute: rectangle = ', target_rectangle.toString())
        # todo: implement and run algorithm...

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

    def intern_read(self, bnd, rect, typ=np.float32, stride=(1, 1)):
        out = np.empty(rect.width * rect.height, typ)
        bnd.readPixels(rect.x, rect.y, rect.width, rect.height, out)
        out.shape = (rect.height, rect.width)
        out = out[::stride[0], ::stride[1]] * 1.

        return out
