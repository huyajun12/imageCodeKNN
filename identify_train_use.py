#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from imageCodeKNN.imgIdentify_C import ProcessImg
from PIL import Image,ImageFilter,ImageEnhance


class ImageProcessor(ProcessImg):
    def __init__(self):
        super(ImageProcessor, self).__init__()

    def process_img(self, im_path):
        im_obj = Image.open(im_path)
        img_array = numpy.array(im_obj)
        binaryzation = []
        for line in img_array:
            sub_bin = []
            for i in line:
                if 135 < i[0] and 135 < i[1] and 135 < i[2]:
                    sub_bin.append(0)
                else:
                    sub_bin.append(1)
            binaryzation.append(sub_bin)
        bin_arr = numpy.array(binaryzation)
        return #bin_arr

