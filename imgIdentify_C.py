#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import operator
import os
import os.path

import numpy
from PIL import Image

logger = logging.getLogger('filehander')


class ProcessImg(object):
    def get_data(self, imgPath):
        try:
            img_array = self.process_img(imgPath)
            im_array = self.clean_img(img_array)
            X_value = self.Cut_X(im_array)
            Y_value = self.Cut_Y(im_array, X_value)
            tData = self.crop_img(im_array, X_value, Y_value)
        except Exception as  e:
            logger.exception(e)
            return
        return tData

    def process_img(self, imgPath):
        raise NotImplementedError('function process_img is not implemented when process {}'.format(imgPath))

    def clean_img(self, img_array):
        '''
        :return: 去噪后的数组
        '''
        col = numpy.zeros((img_array.shape[0]))
        img_array = numpy.column_stack((img_array, numpy.array(col)))
        width = img_array.shape[1]
        height = img_array.shape[0]
        for _ in range(4):
            # 有些噪音1与非噪音1连在一起，如果一行或一列只有一个1，那么可视为噪音
            for i in range(height):
                for j in range(width):
                    if img_array[i][j] == 1 and (sum(img_array[:, j]) == 1 or sum(img_array[i, :]) == 1):
                        img_array[i][j] = 0

            for i in range(height):
                for j in range(width):
                    # 四个角
                    if i == 0 and j == 0 and img_array[i][j] == 1 and sum(
                            [img_array[i + 1][j], img_array[i][j + 1]]) == 0:  # 左上角
                        img_array[i][j] = 0

                    elif j == (width - 1) and i == 0 and img_array[i][j] == 1 and sum(
                            [img_array[i][j - 1], img_array[i + 1][j]]) == 0:  # 右上角
                        img_array[i][j] = 0

                    elif j == 0 and i == (height - 1) and img_array[i][j] == 1 and sum(
                            [img_array[i - 1][j], img_array[i][j + 1]]) == 0:  # 左下角
                        img_array[i][j] = 0

                    elif j == (width - 1) and i == (height - 1) and img_array[i][j] == 1 and sum(
                            [img_array[i][j - 1], img_array[i - 1][j]]) == 0:  # 右下角
                        img_array[i][j] = 0

                    # 四条边
                    elif 1 <= j <= width - 2 and i == 0 and img_array[i][j] == 1 and sum(
                            [img_array[i][j - 1], img_array[i + 1][j], img_array[i][j + 1]]) == 0:  # 上边
                        img_array[i][j] = 0

                    elif 1 <= j <= width - 2 and i == (height - 1) and img_array[i][j] == 1 and sum(
                            [img_array[i][j - 1], img_array[i - 1][j], img_array[i][j + 1]]) == 0:  # 下边
                        img_array[i][j] = 0

                    elif 1 <= i <= height - 2 and j == 0 and img_array[i][j] == 1 and sum(
                            [img_array[i][j - 1], img_array[i][j + 1], img_array[i + 1][j]]) == 0:  # 左边
                        img_array[i][j] = 0

                    elif 1 <= i <= height - 2 and j == (width - 1) and img_array[i][j] == 1 and sum(
                            [img_array[i - 1][j], img_array[i + 1][j], img_array[i][j - 1]]) == 0:  # 右边
                        img_array[i][j] = 0

                    # 其余部分
                    elif 1 <= i <= height - 2 and 1 <= j <= width - 2 and img_array[i][j] == 1 and (sum(
                            [img_array[i - 1][j - 1], img_array[i - 1][j], img_array[i - 1][j + 1],
                             img_array[i][j - 1], img_array[i][j + 1],
                             img_array[i + 1][j - 1], img_array[i + 1][j], img_array[i + 1][j + 1]]) == 1 or sum(
                        [img_array[i][j - 1], img_array[i][j + 1], img_array[i - 1][j], img_array[i + 1][j]]) == 0):
                        img_array[i][j] = 0

        return img_array

    def Cut_X(self, im_array):
        '''
        :return: 8元素列表
        '''
        X_value = []
        List0 = []
        List1_num = []
        ListRow0 = []
        ListRow1 = []

        l = len(im_array[0])
        for i in range(l):
            if sum(im_array[:, i]) <= 1 and len(ListRow1) == 0:  # 数字左侧有空白
                ListRow0.append(i)
            elif sum(im_array[:, i]) <= 1 and len(ListRow1) > 0:  # 数字右侧有空白
                List1_num.append(ListRow1)
                ListRow1 = []
                ListRow0.append(i)
            elif sum(im_array[:, i]) > 1 and len(ListRow0) > 0:  # 数字列
                List0.append(ListRow0)
                ListRow0 = []
                ListRow1.append(i)
            elif sum(im_array[:, i]) > 1 and len(ListRow0) == 0:  # 数字列
                ListRow1.append(i)

        # 去除List1_num中有问题的元素，也就是说这个元素是噪声
        List1 = []
        for sub_list in List1_num:
            if len(sub_list) > 2:
                List1.append(sub_list)

        # ListRow1不为空那么ListRow0一定为空，但是如果ListRow1不为空但是噪声呢？
        if ListRow1 and len(ListRow1) <= 2:
            ListRow0.append(ListRow1)
        # 在相信List1中的元素都是干净的情况下，做出如下判断：

        if len(List1) == 1:
            len1 = len(List1[0])
            len2 = int(len1 / 4)
            j = 0
            for i in range(4):
                X_value.append(List1[0][j])
                j += (len2 - 2)
                if i != 3:
                    X_value.append(List1[0][j])
                    j += 2
                else:
                    X_value.append(List1[0][-1])

        elif len(List1) == 2:
            len0 = len(List1[0])
            len1 = len(List1[1])
            #print(len0, len1)
            if 0.6 <= float(len1) / float(len0) <= 1.0 or 0.6 <= float(len0) / float(len1) <= 1.0:
                mid0 = int(len0 / 2)
                mid1 = int(len1 / 2)
                # 将list[0]拆成两段
                X_value.append(List1[0][0])
                X_value.append(List1[0][mid0 - 1])

                X_value.append(List1[0][mid0 + 1])
                X_value.append(List1[0][(len(List1[0]) - 1)])
                # 将list[1]拆成两段
                X_value.append(List1[1][0])
                X_value.append(List1[1][mid1 - 1])

                X_value.append(List1[1][mid1 + 1])
                X_value.append(List1[1][(len(List1[1]) - 1)])

            elif len0 > len1 and float(len1) / float(len0) < 0.5:  # 将List[0]拆成3段
                mid = int(round(len0 / 3.0, 0))
                X_value.append(List1[0][0])
                X_value.append(List1[0][mid - 1])

                X_value.append(List1[0][mid + 1])
                X_value.append(List1[0][mid * 2 - 1])

                X_value.append(List1[0][mid * 2 + 1])
                X_value.append(List1[0][(len(List1[0]) - 1)])

                X_value.append(List1[1][0])
                X_value.append(List1[1][len(List1[1]) - 1])

            elif len0 < len1 and float(len0) / float(len1) < 0.5:  # 将List[1]拆成3段
                mid = int(round(len1 / 3.0, 0))
                X_value.append(List1[0][0])
                X_value.append(List1[0][len(List1[0]) - 1])

                X_value.append(List1[1][0])
                X_value.append(List1[1][mid - 1])

                X_value.append(List1[1][mid + 1])
                X_value.append(List1[1][mid * 2 - 1])

                X_value.append(List1[1][mid * 2 + 1])
                X_value.append(List1[1][(len(List1[1]) - 1)])

        # elif len(List1) == 3 and not ListRow0:
        #     # 最后一个数字右侧没有空白，那么最后一个数字所对应的所占的List1在还没有被清空的ListRow1中
        #     X_value = [List1[0][0], List1[0][(len(List1[0]) - 1)], List1[1][0], List1[1][(len(List1[1]) - 1)],
        #                List1[2][0], List1[2][(len(List1[2]) - 1)], ListRow1[0], ListRow1[(len(ListRow1) - 1)]]

        elif len(List1) == 3:  # 有两个数字连在一起了
            len_list = [len(List1[i]) for i in range(len(List1))]
            max_len = max(len_list)
            max_index = int(len_list.index(max_len))
            mid_index = int(max_len / 2)
            for i in range(len(List1)):
                if i == max_index:
                    X_value.append(List1[i][0])
                    X_value.append(List1[i][mid_index - 1])

                    X_value.append(List1[i][mid_index + 1])
                    X_value.append(List1[i][(len(List1[i]) - 1)])
                else:
                    X_value.append(List1[i][0])
                    X_value.append(List1[i][(len(List1[i]) - 1)])

        elif len(List1) == 4:  # 4个数字右侧均有空白
            for i in range(len(List1)):
                X_value.append(List1[i][0])
                X_value.append(List1[i][(len(List1[i]) - 1)])

        elif len(List1) >= 5:  # 取长度最长的4段。
            mm = []
            for i, sub_list in enumerate(List1):
                mm.append([])
                mm[i].append(len(sub_list))
                mm[i].append([sub_list[0], sub_list[len(sub_list) - 1]])
            sorted_mm = sorted(mm, key=lambda x: x[0], reverse=True)[:4]

            for i in sorted_mm:
                X_value.append(i[1][0])
                X_value.append(i[1][1])
            X_value = sorted(X_value)

        else:
            logger.debug(u'X_value erroe: %s' % (X_value))

        return X_value

    def Cut_Y(self, im_array, X_value):
        '''
        :return:
        '''
        Y_value = []
        if len(X_value) == 8:
            for k in range(4):
                Image_Value = []
                for j in range(im_array.shape[0]):
                    count = 0
                    for i in range(X_value[(2 * k)], (X_value[(2 * k + 1)] + 1)):
                        if im_array[j, i] == 1:
                            count += 1
                    Image_Value.append(count)
                # ++++++++++++++++++++
                # 去除im_array中连续的噪音1
                l = len(Image_Value)
                for i in range(l):
                    if i == 0 and Image_Value[i] != 0 and Image_Value[i + 1] == 0:
                        Image_Value[i] = 0
                    elif i != 0 and i != l - 1 and Image_Value[i] != 0 and Image_Value[i - 1] == 0 and Image_Value[
                                i + 1] == 0:
                        Image_Value[i] = 0
                    elif i == l - 1 and Image_Value[i] != 0 and Image_Value[i - 1] == 0:
                        Image_Value[i] = 0
                # ++++++++++++++++++++

                for i in range(l):
                    if Image_Value[i] > 1:
                        Y_value.append(i + 1)
                        break
                for i in range((l - 1), 0, (-1)):
                    if Image_Value[i] > 1:
                        Y_value.append(i + 1)
                        break
        else:
            logger.debug(u'Y_value erroe: %s' % (Y_value))
            return

        return Y_value

    def crop_img(self, im_array, X_value, Y_value):
        '''
        :return:最终的样本
        '''
        if len(X_value) == 8 and len(Y_value) == 8:
            p1 = im_array[(Y_value[0] - 1):Y_value[1], X_value[0]:(X_value[1] + 1)]
            p2 = im_array[(Y_value[2] - 1):Y_value[3], X_value[2]:(X_value[3] + 1)]
            p3 = im_array[(Y_value[4] - 1):Y_value[5], X_value[4]:(X_value[5] + 1)]
            p4 = im_array[(Y_value[6] - 1):Y_value[7], X_value[6]:(X_value[7] + 1)]

            pic1 = numpy.array(Image.fromarray(numpy.int8(p1)).resize((32, 32)))
            pic2 = numpy.array(Image.fromarray(numpy.int8(p2)).resize((32, 32)))
            pic3 = numpy.array(Image.fromarray(numpy.int8(p3)).resize((32, 32)))
            pic4 = numpy.array(Image.fromarray(numpy.int8(p4)).resize((32, 32)))

            croped_imglist = [pic1, pic2, pic3, pic4]

        else:
            logger.debug(u'crop_img erroe')
            return

        return croped_imglist


class IdentifyImg(object):
    def identify_image_code(self, tData, traindir, knn=3):
        result = []
        traindata = self.load_train_data(traindir)
        if tData:
            for i, testData in enumerate(tData):
                one_num = self.classify(testData, traindata, knn)
                result.append(one_num)
            theLast = ''.join([result[0], result[1], result[2], result[3]])
            return theLast
        else:
            logger.debug(u'identifyImageCode failed')
            return ''

    def load_train_data(self, traindir):
        '''
        :return: 训练数据集的二维数组列表
        '''
        train_data = []
        dirnames = os.listdir(traindir)

        for dirname in dirnames:
            sub_list = []
            sub_path = os.path.join(traindir, str(dirname))
            filenames = os.listdir(sub_path)

            for j, file in enumerate(filenames):
                sub_list.append([])
                label = file[:file.find('.')]
                img_array = numpy.loadtxt(os.path.join(sub_path, file))
                sub_list[j].append(label)
                sub_list[j].append(img_array)

            [train_data.append(sub_list[k]) for k in range(len(sub_list))]
        return train_data

    def classify(self, testData, traindata, knn):
        '''
        :param testData:
        :return:类别
        '''
        cla_dis = []
        for j, single in enumerate(traindata):
            cla_dis.append([])
            dis = sum(sum((single[1] - testData) ** 2))
            cla_dis[j].append(single[0])
            cla_dis[j].append(dis)
        sorted_cla_dis = sorted(cla_dis, key=lambda cla_dis: cla_dis[1])
        knn_cla_dis = sorted_cla_dis[:knn]

        classCount = {}
        for single in knn_cla_dis:
            label = single[0][:single[0].find('_')]
            classCount[label] = classCount.get(label, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]
