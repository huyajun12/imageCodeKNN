#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import re
import numpy
import operator
import os.path
from PIL import Image, ImageFilter, ImageEnhance



def savefile(dir,iter):
    '''
    :param dir: 要保存文件的目录
    :param iter: 一个二维的可迭代对象
    :return:
    '''
    if dir:
        with open(dir, 'w+') as f:
            for line in iter:
                for i in line:
                    if i:
                        f.write('1 ')
                    else:
                        f.write('0 ')
                f.write('\n')
    else:
        print('save file failed')


def process_img(im_obj):
    '''
    :param im_obj: 图片对象
    :return: 二值化后的数组
    '''
    img_array = numpy.array(im_obj)
    l0 = img_array.shape[0]
    l1 = img_array.shape[1]
    for i in range(l0):
        for j in range(l1):
            if 150 <= img_array[i][j][0] and 160 <= img_array[i][j][1] and 160 <= img_array[i][j][2]:
                img_array[i][j][0] = 255
                img_array[i][j][1] = 255
                img_array[i][j][2] = 255
    img_obj2 = Image.fromarray(img_array)
    im_obj3 = img_obj2.filter(ImageFilter.MedianFilter())
    # im_obj3.save(r'G:\PycharmProjects\machineLearning\imageCodeKNN\CT_jiangsu\imagesArray\0.jpg')
    # img_obj2.show()
    img_array2 = numpy.array(im_obj3)

    bin_list = []
    for line in img_array2:
        sub_bin = []
        for i in line:
            if 150 <= i[0] and 160 <= i[1] and 160 <= i[2]:
                sub_bin.append(0)
            else:
                sub_bin.append(1)
        bin_list.append(sub_bin)

    return numpy.array(bin_list)


def clean_img(img_array):
    '''
    :param img_array: 二值化后的数组
    :return: 去噪后的数组
    '''
    col = numpy.zeros((img_array.shape[0]))
    img_array = numpy.column_stack((img_array, numpy.array(col)))
    width = img_array.shape[1]  # 列
    height = img_array.shape[0]  # 行
    for _ in range(4):
        # 有些噪音1与非噪音1连在一起，那么上面的规则就无法清除，如果一行或一列只有一个1，那么可视为噪音
        for i in range(height):
            for j in range(width):
                if img_array[i][j] == 1 and (sum(img_array[:, j]) == 1 or sum(img_array[i, :]) == 1):
                    img_array[i][j] = 0

        for i in range(height):
            for j in range(width):
                # 四个角：清除如下形式的噪音
                # 1 0
                # 0
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

                # 四条边：清除如下形式的噪音
                # 0 1 0
                #   0
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

                # 其余部分：清除如下形式的噪音
                # 0 0 1      0
                # 0 1 0    0 1 0
                # 0 0 0      0
                elif 1 <= i <= height - 2 and 1 <= j <= width - 2 and img_array[i][j] == 1 and (sum(
                        [img_array[i - 1][j - 1], img_array[i - 1][j], img_array[i - 1][j + 1],
                         img_array[i][j - 1], img_array[i][j + 1],
                         img_array[i + 1][j - 1], img_array[i + 1][j], img_array[i + 1][j + 1]]) == 1 or sum(
                    [img_array[i][j - 1], img_array[i][j + 1], img_array[i - 1][j], img_array[i + 1][j]]) == 0):
                    img_array[i][j] = 0

    return img_array

# ===========================================================================
def Cut_X(im_array):
    '''
    :param im_array: 数组
    :return: 8元素列表
    '''
    X_value=[]
    List0=[]
    List1_num=[]
    ListRow0=[]
    ListRow1=[]

    l = len(im_array[0])
    for i in range(l):
        if sum(im_array[:, i]) <= 1 and len(ListRow1) == 0:  #数字左侧有空白
            ListRow0.append(i)
        elif sum(im_array[:, i]) <= 1 and len(ListRow1) > 0:  #数字右侧有空白
            List1_num.append(ListRow1)
            ListRow1=[]
            ListRow0.append(i)
        elif sum(im_array[:, i]) > 1 and len(ListRow0) > 0:  #数字列
            List0.append(ListRow0)
            ListRow0=[]
            ListRow1.append(i)
        elif sum(im_array[:, i]) > 1 and len(ListRow0) == 0:  #数字列
            ListRow1.append(i)

    #去除List1_num中有问题的元素，也就是说这个元素是噪声
    List1 = []
    for sub_list in List1_num:
        if len(sub_list) >= 2:
            List1.append(sub_list)
    print('list1', List1)
    print('ListRow0', ListRow0)

    #ListRow1不为空那么ListRow0一定为空，但是如果ListRow1不为空但是噪声呢？
    if ListRow1 and len(ListRow1) <= 2:
        ListRow0.append(ListRow1)

    #在相信List1中的元素都是干净的情况下，做出如下判断：
    if len(List1) == 2 and ListRow0:
        print('if len(List1) == 2 and ListRow0')
        len0 = len(List1[0])
        len1 = len(List1[1])
        print(len0)
        print(len1)
        if 0.7 <= float(len1) / float(len0) <= 1.0 or 0.7 <= float(len0) / float(len1) <= 1.0:
            mid0 = int(len0 / 2)
            mid1 = int(len1 / 2)
            #将list[0]拆成两段
            X_value.append(List1[0][0])
            X_value.append(List1[0][mid0 - 1])

            X_value.append(List1[0][mid0 + 1])
            X_value.append(List1[0][(len(List1[0]) - 1)])
            # 将list[1]拆成两段
            X_value.append(List1[1][0])
            X_value.append(List1[1][mid1 - 1])

            X_value.append(List1[1][mid1 + 1])
            X_value.append(List1[1][(len(List1[1]) - 1)])

        elif len0 > len1 and  float(len1) / float(len0) < 0.5:  # 将List[0]拆成3段
            mid =int(round(len0 / 3.0, 0))
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

    elif len(List1) == 3 and not ListRow0:
        #最后一个数字右侧没有空白，那么最后一个数字所对应的所占的List1在还没有被清空的ListRow1中
        X_value = [List1[0][0], List1[0][(len(List1[0])-1)], List1[1][0], List1[1][(len(List1[1])-1)],
                   List1[2][0], List1[2][(len(List1[2])-1)], ListRow1[0], ListRow1[(len(ListRow1)-1)]]

    elif len(List1) == 3 and ListRow0:  # 有两个数字连在一起了
        len_list = [len(List1[i]) for i in range(len(List1))]
        max_len = max(len_list)
        max_index = len_list.index(max_len)
        mid_index = int(max_len/2)
        for i in range(len(List1)):
            if i == max_index:
                X_value.append(List1[i][0])
                X_value.append(List1[i][mid_index - 1])

                X_value.append(List1[i][mid_index + 1])
                X_value.append(List1[i][(len(List1[i]) - 1)])
            else:
                X_value.append(List1[i][0])
                X_value.append(List1[i][(len(List1[i]) - 1)])

    elif len(List1) == 4:  #4个数字右侧均有空白
        for i in range(len(List1)):
            X_value.append(List1[i][0])
            X_value.append(List1[i][(len(List1[i])-1)])

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
        print('X_value%s错误'%(X_value))

    print(X_value)

    return X_value


def Cut_Y(im_array, X_value):
    '''
    :param im_array:数组
    :param X_value:
    :return:
    '''
    Y_value = []
    if len(X_value) == 8:
        for k in range(4):
            Image_Value = [] #记录每一行中1的个数
            for j in range(im_array.shape[0]):
                count = 0
                for i in range(X_value[(2 * k)], (X_value[(2 * k+1)]+1)):  #k=0,j=0 行数,i=3
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
                    Y_value.append(i+1)
                    break
            for i in range((l - 1), 0, (-1)):
                if Image_Value[i] > 1:
                    Y_value.append(i+1)
                    break
    else:
        print('Y_value%s错误'%(Y_value))

    return Y_value

#===========================================================================
# X_value = [3, 13, 26, 37, 51, 63, 81, 101]
# Y_value = [12, 36, 19, 37, 17, 37, 11, 37]

def crop_img(im_array, X_list, Y_list):
    '''
    :param im_array: 数组
    :param X_list:
    :param Y_list:
    :return:
    '''
    pictures = []
    if len(X_list) == 8 and len(Y_list) == 8:
        # crop(起始点的横坐标，起始点的纵坐标，宽度，高度） 没有用crop函数

        tt = Image.fromarray(im_array[(Y_list[0]-1):Y_list[1], X_list[0]:(X_list[1]+1)])
        picture1 = numpy.array(tt.resize((32, 32)))
        picture2 = numpy.array(Image.fromarray(im_array[(Y_list[2]-1):Y_list[3], X_list[2]:(X_list[3]+1)]).resize((32, 32)))
        picture3 = numpy.array(Image.fromarray(im_array[(Y_list[4]-1):Y_list[5], X_list[4]:(X_list[5]+1)]).resize((32, 32)))
        picture4 = numpy.array(Image.fromarray(im_array[(Y_list[6]-1):Y_list[7], X_list[6]:(X_list[7]+1)]).resize((32, 32)))
        pictures = [picture1, picture2, picture3, picture4]
    else:
        print('crop_img:Y_list%s,错误'%(Y_list))

    return pictures


def get_train_data(dir1, dir2=None):
    '''
    :param dir1:原始图片验证码目录
    :param dir2:切割后数组形式验证码保存目录
    :return:None  主要是将切割后数据写入文件
    '''
    image_names = os.listdir(dir1)
    for n, image_name in enumerate(image_names):
        # try:
        # prefix = re.search(r'(img\d+\.{0,1}\d+)', image_name).group(1)
        prefix = re.search(r'(\d+)', image_name).group(1)
        im_obj = Image.open(dir1 + '/' + image_name)

        processed_img_array = process_img(im_obj)
        cleaned_img_array = clean_img(processed_img_array)
        # --------------
        d = r'/Users/huyajun/work/imageCodeKNN/ddd'
        if not os.path.exists(d):
            os.mkdir(d)
        di = ''.join([d, '/', prefix, '.txt'])
        savefile(di, cleaned_img_array)
        # --------------

        X_list = Cut_X(cleaned_img_array)
        Y_list = Cut_Y(cleaned_img_array, X_list)
        croped_imglist = crop_img(cleaned_img_array, X_list, Y_list)

        if croped_imglist and dir2:
            for x, subimg_array in enumerate(croped_imglist):
                dir = ''.join([dir2, '/', prefix, str(x), '.txt'])
                savefile(dir, subimg_array)
        else:
            print('%s croped_imglist保存失败'%(image_name))

        # except Exception as e:
        #     print('%s get_train_data发生异常' % (image_name))
        #     continue

def get_test_data(dir):
    '''
    :param dir:原始图片验证码目录
    :return:切割后的数组列表
    '''
    im_obj = Image.open(dir)

    processed_img_array = process_img(im_obj)
    # # --------------
    # dir1 = r'G:\PycharmProjects\imageCodeKNN\step1\1.txt'
    # savefile(dir1, processed_img_array)
    # # --------------

    cleaned_img_array = clean_img(processed_img_array)
    # # --------------
    # dir2 = r'G:\PycharmProjects\imageCodeKNN\step1\2.txt'
    # savefile(dir2, cleaned_img_array)
    # # --------------

    X_list = Cut_X(cleaned_img_array)
    Y_list = Cut_Y(cleaned_img_array, X_list)
    croped_imglist = crop_img(cleaned_img_array, X_list, Y_list)

    if not croped_imglist:
        print('get_test_data 失败')

    return croped_imglist
# +++++++++++++++++++++++++++++
def load_train_data(dir):
    '''
    :param dir: 训练数据集目录
    :return: 训练数据集的二维数组列表
    '''
    train_data = []
    dirnames = os.listdir(dir)

    for dirname in dirnames:
        sub_list = []
        sub_path = ''.join([dir, '/', str(dirname)])
        filenames = os.listdir(sub_path)

        for j, file in enumerate(filenames):
            sub_list.append([])
            label = file[:file.find('.')]
            img_array = numpy.loadtxt(sub_path + '/' + file)
            sub_list[j].append(label)
            sub_list[j].append(img_array)

        [train_data.append(sub_list[k]) for k in range(len(sub_list))]
    return train_data


def classify(test_data, train_data, knn=3):
    '''
    :param test_data: a list which contains numpy array
    :param train_data: a double list which contains numpy array
    :param knn:
    :return:类别
    '''
    cla_dis = []
    for j, single in enumerate(train_data):
        cla_dis.append([])
        dis = sum(sum((single[1] - test_data) ** 2))
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

# +++++++++++++++++++++++++++++
def identifyImageCode(dir1, dir2):
    '''
    :param dir1: 测试数据集目录
    :param dir2: 训练好的数据集目录
    :return:
    '''
    test_img_names = os.listdir(dir1)
    traindata = load_train_data(dir2)

    for test_img_name in test_img_names:
        testdir = ''.join([dir1, '/', test_img_name])
        testdata = get_test_data(testdir)
        if testdata:
            result = []

            for test in testdata:
                one_num = classify(test, traindata)
                print(one_num,)
                result.append(one_num)
            try:
                new_name = ''.join([dir1, '/', result[0], result[1], result[2], result[3], '.jpg'])
                os.renames(testdir, new_name)
            except Exception as e:
                continue

        else:
            print('%s 识别失败'%(test_img_name))

# +++++++++++++++++++++++++++++
'''
:param dir1:原始图片验证码目录
:param dir2:切割后数组形式验证码保存目录
'''
dir1 = r'/Users/huyajun/work/imageCodeKNN/aaa'
dir2 = r'/Users/huyajun/work/imageCodeKNN/ccc'
if not os.path.exists(dir2):
    os.mkdir(dir2)
get_train_data(dir1, dir2)

# +++++++++++++++++++++++++++++
traindata_dir = r'CT_jiangsu\trainData'
testdata_dir = r'CT_jiangsu\testImages'
#identifyImageCode(testdata_dir, traindata_dir)

