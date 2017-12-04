#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import shutil
import requests
from imgIdentify_C import ProcessImg, IdentifyImg
import traceback
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class GetTrainData():
    def __init__(self, types=None):
        if not types:
            print("请输入项目路径，形如types=['funds', 'hubwi', 'wuhan']")
            return
        self.dir = os.path.join(os.path.dirname(__file__), *types)
        self.trainDir = os.path.join(self.dir, 'trainData')
        self.testDir = os.path.join(self.dir, 'testData')
        self.cropedImgDir = os.path.join(self.dir, 'cropedImages')
        self.rawImgDir = os.path.join(self.dir, 'rawImages')
        self.imgArrayDir = os.path.join(self.dir, 'imagesArray')

        current_dir = os.path.dirname(__file__)

        if not os.path.exists(self.rawImgDir):
            os.makedirs(self.rawImgDir)
        if not os.path.exists(self.trainDir):
            os.mkdir(self.trainDir)
        if not os.path.exists(self.cropedImgDir):
            os.mkdir(self.cropedImgDir)
        if not os.path.exists(self.imgArrayDir):
            os.mkdir(self.imgArrayDir)
        if not os.path.exists(self.testDir):
            os.mkdir(self.testDir)


        dirname = '.'.join(types)
        module_path = '{}.identify'.format(dirname)

        old_path =os.path.join(current_dir, 'identify_train_use.py')
        new_path = os.path.join(self.dir, 'identify.py')
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)
        old_path = os.path.join(current_dir, '__init__.py')
        new_path = os.path.join(self.dir, '__init__.py')
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)

        old_path = os.path.join(current_dir, '__init__.py')
        new_path = os.path.join(self.dir, '__init__.py')
        if not os.path.exists(new_path):
            shutil.copy(old_path, new_path)

        identify_module = __import__(module_path, {}, {}, ['identify', ])
        self.identify_instances = identify_module.ImageProcessor()


    def getImage(self, url, num, dir, append=False):
        if url == None or num == None:
            print('请输入图片下载url和下载图片数量num')
            return
        i = 0
        if append:
            img_num = len(os.listdir(self.rawImgDir))
            i += img_num
            num += img_num
        print('开始下载')
        while i < num:
            resp = requests.get(url, verify=False)
            file_name = ''.join([str(i), '.jpg'])

            if resp.status_code != 200:
                print('第{}张验证码下载失败'.format(i))
            elif dir == self.trainDir and resp.status_code == 200:
                file_dir = os.path.join(self.rawImgDir, file_name)
                with open(file_dir, 'wb') as f:
                    f.write(resp.content)
                i += 1
            elif dir == self.testDir and resp.status_code == 200:
                file_dir = os.path.join(dir, file_name)
                with open(file_dir, 'wb') as f:
                    f.write(resp.content)
                i += 1
        print('验证码下载完毕')

    def make_dir(self):
        dirnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
                    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                    'x', 'y', 'z']
        for dirname in dirnames:
            dir = os.path.join(self.trainDir, dirname)
            if os.path.exists(dir) and os.path.isdir(dir):
                files = os.listdir(dir)
                if not len(files):
                    continue
                for file in files:
                    filePath = os.path.join(dir, file)
                    if os.path.isfile(filePath):
                        try:
                            os.remove(filePath)
                        except os.error:
                            print("remove %s error." % filePath)
                    elif os.path.isdir(filePath):
                        shutil.rmtree(filePath, True)
            else:
                os.mkdir(dir)

    def prepare(self, url, num, append=False):
        if len(os.listdir(self.rawImgDir)) == 0 or append:
            self.getImage(url, num, self.trainDir, append)

        def save_file(dir, iter):
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

        image_names = os.listdir(self.rawImgDir)
        for n, image_name in enumerate(image_names):
            prefix = re.search(r'(\d+)', image_name).group(1)
            tdir = os.path.join(self.rawImgDir, image_name)

            img_array = self.identify_instances.process_img(tdir)
            if img_array == None:
                print('请先完善identify.py文件')
                return
            else:
                im_array = self.identify_instances.clean_img(img_array)
                imgArrayDir = ''.join([self.imgArrayDir, '/', prefix, '.txt'])

                # img_obj = Image.fromarray(np.int8(im_array*255)).convert('L')
                # img_obj.save(imgArrayDir)
                # -------------------------------------------------------------
                aa = []
                bb = []
                l = len(im_array[1])
                for i in range(l):
                    s = sum(im_array[:, i])
                    bb.append(i)
                    aa.append(s)
                print('bb', bb)
                print('aa', aa)
                plt.plot(bb, aa)
                m = []
                laa= len(aa)
                for i in range(laa):
                    if aa[i] != 0:
                        m.append(i)
                        break
                for i in range(laa):
                    if aa[::-1][i] != 0:
                        m.append((laa-i))
                        break

                ad = (m[1] - m[0])/4
                x1 = [m[0],m[0]+ad,m[0]+ad*2,m[1]]
                maa = max(aa)
                y1= [maa,maa,maa,maa]
                plt.show()
                plt.hist(x1,color='r')
                # -------------------------------------------------------------
                with open(imgArrayDir, 'w+') as f:
                    for line in im_array:
                        for i in line:
                            if i:
                                f.write('1')
                            else:
                                f.write('0')
                        f.write('\n')

                save_file(imgArrayDir, im_array)

                X_value = self.identify_instances.Cut_X(im_array)
                print(image_name, 'X_value', X_value)
                Y_value = self.identify_instances.Cut_Y(im_array, X_value)
                print(image_name, 'Y_value', Y_value)


                trainData = self.identify_instances.get_data(tdir)
                if trainData:
                    for x, subimg_array in enumerate(trainData):
                        file_name = ''.join([prefix, str(x), '.txt'])
                        dir = os.path.join(self.cropedImgDir, file_name)
                        if dir:
                            save_file(dir, subimg_array)
                        else:
                            print('save file failed')
                else:
                    print('%s 识别失败' % (image_name))
        print('图片分割完成！')

    def label_img(self, imageCode):
        self.make_dir()

        img_names = sorted(os.listdir(self.cropedImgDir), key=lambda x: int(re.match('(\d+)\.txt', x).group(1)))
        image_code = imageCode.replace(',', '')

        if len(img_names) == len(image_code):
            for i, img_name in enumerate(img_names):
                old_p = os.path.join(self.cropedImgDir, img_name)
                file_name = ''.join([image_code[i], '_', str(i), '.txt'])

                new_p = os.path.join(self.trainDir, image_code[i], file_name)
                shutil.move(old_p, new_p)

            dir_list = os.listdir(self.trainDir)
            for j in dir_list:
                if not len(os.listdir(os.path.join(self.trainDir, j))) and os.path.isdir(
                        os.path.join(self.trainDir, j)):
                    os.rmdir(os.path.join(self.trainDir, j))
        else:
            print('切割后的图片数{}与验证码长度不匹配{}'.format(len(img_names), len(image_code)))
            return
        print('类别标记完成！')

    def test(self, url=None, num=None, imageCode=None,knn=3):
        if len(os.listdir(self.cropedImgDir)) and (not len(os.listdir(self.trainDir))):
            if imageCode:
                self.label_img(imageCode)
            else:
                print('请输入imageCode参数')
                return

        if not len(os.listdir(self.testDir)):
            self.getImage(url, num, self.testDir)

        test_img_names = os.listdir(self.testDir)
        identifyImg = IdentifyImg()
        print('开始测试！')
        for test_img_name in test_img_names:
            print(test_img_name)
            testdir = os.path.join(self.testDir, test_img_name)
            try:
                testData = self.identify_instances.get_data(testdir)
                code = identifyImg.identify_image_code(testData, self.trainDir, knn)
                new_name = os.path.join(self.testDir, ''.join([code, '.jpg']))
                if not os.path.exists(new_name):
                    os.renames(testdir, new_name)
            except Exception as e:
                print(traceback.format_exc())
                return
        print('测试完成！')


if __name__ == '__main__':
    # 分析图片---测试
    import sys

    project_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_dir)
    aa = GetTrainData(types=['regist', 'mxd'])
    url = r'http://mx.nonobank.com/Validate/checkCode?&i=0.0020024917167151823'
    num =150
    aa.prepare(url=url, num=num)
    num = 100
    imageCode = 'zknf,kpj1,qkku,6hz9,wneq,ka3p,zy8m,tyqh,' \
                'vj1t,h629,6kfy,nmcz,t3gv,73zp,tbuy,bgmj,' \
                'ptec,zux9,bdiq,g4w2,2uk6,kc6m,jjvt,b5mx,' \
                'g3ix,2ptp,umaq,2qn5,p6zu,nhhu,w6ad,u131,' \
                'uy5m,f1ix,wr1i,d116,mpbs,se9k,mugd,x8jr,' \
                '12a6,14da,szmg,adfm,yftb,rs91,d67w,xpg4,' \
                'm2vk,kva9,2kbj,g9fc,3qek,brsh,9uk7,swib,' \
                'rwxf,zksy,9jqu,xpkm,6snp,eauh,yqsg,ubns,' \
                '269v,m3if,hb9v,pkzu,tejn,ge8z,scgj,3az8,' \
                'e37h,wm2x,smge,xyy7,fu9h,m99m,57xf,8f4g,' \
                'vztp,msfn,63e8,buyj,xbet,2txn,zjvv,zs2m,' \
                'qzqq,pusq,7msa,556t,vpwd,4kku,rmxe,na28,' \
                '4ye3,5txg,qk8y,8pe5,j5yy,qdix,siw4,ui4d,' \
                'wt97,amv6,485t,6tsu,e39b,hyvs,s6kp,1dj4,' \
                'v2vs,a1fn,aqw3,2hzx,4wig,kxnu,4f4e,ywmt,' \
                'bqmh,z8aw,sfjx,xkvz,mebn,24ey,ay8b,x9gm,' \
                '2rk9,6wag,iidr,4sm8,bb4u,gckq,6v56,vnqk,' \
                'z3zq,2mgt,gfur,1x7x,ajgr,fqcq,gfg3,gzj6,' \
                'fhft,k8fa,9p5q,az5k,b8u8,33e4'



    # aa.test(url,num,imageCode=imageCode)




# def getImage2(dir, url, num):
#     ssl._create_default_https_context = ssl._create_unverified_context
#     for i in range(num):
#         resp = request.urlopen(url).read()
#         file_dir = ''.join([dir, '/', str(i), '.jpg'])
#         with open(file_dir, 'wb') as f:
#             f.write(resp)
