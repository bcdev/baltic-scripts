import os
import platform
import sys
import tempfile

import numpy as np
import snappy

from snappy import ProductData

import baltic_ac_algorithm


class BalticOp:
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
        balticac_product = snappy.Product('pyBALTICAC', 'pyBALTICAC', width, height)
        balticac_product.setDescription('Baltic+ AC product')
        balticac_product.setStartTime(source_product.getStartTime())
        balticac_product.setEndTime(source_product.getEndTime())

        # setup target bands:
        # todo
        # test: define one target band
        self.test_band = balticac_product.addBand('test_band', ProductData.TYPE_FLOAT64)
        self.test_band.setDescription('Test band: radiance0 * 2')

        snappy.ProductUtils.copyTiePointGrids(source_product, balticac_product)
        source_product.transferGeoCodingTo(balticac_product, None)

        operator.setTargetProduct(balticac_product)

        self.baltic_ac_algo = baltic_ac_algorithm.BalticAcAlgorithm()
        self.baltic_ac_algo.run(None)

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

        # test: write one target band
        src_rad0_tile = operator.getSourceTile(self.rad_bands[0], target_rectangle)
        src_rad0_samples = src_rad0_tile.getSamplesFloat()
        src_rad0_data = np.array(src_rad0_samples, dtype=np.float32)

        target_test_data = np.empty(src_rad0_data.shape[0], dtype=np.float32)
        for i in range(0, src_rad0_data.shape[0]):
            target_test_data[i] = src_rad0_data[i] * 2.0

        target_test_tile = target_tiles.get(self.test_band)
        target_test_tile.setSamples(target_test_data)

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
