#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import ssl
import shutil
import requests
# from urllib import request

def makedir(dir, dirnames):
    if not os.path.exists(dir):
        os.mkdir(dir)
    for dirname in dirnames:
        dir3 = ''.join([dir, '/', dirname])
        os.mkdir(dir3)


def classifyCropedImages(dir1, dir2, imageCode):
    img_names = sorted(os.listdir(dir1), key=lambda x: int(re.match('(\d+)\.txt', x).group(1)))
    image_code = imageCode.replace(',', '')

    for i, img_name in enumerate(img_names):
        old_p = ''.join([dir1, '/', img_name])
        new_p = ''.join([dir2, '/', image_code[i], '/', image_code[i], '_', str(i+800), '.txt'])
        shutil.move(old_p, new_p)

def rename(dir):
    img_names = sorted(os.listdir(dir))
    for i, img_name in enumerate(img_names):
        old_n = ''.join([dir, '/', img_name])
        new_n = ''.join([dir, '/', str(i), '.jpg'])
        os.renames(old_n, new_n)


def getImage1(dir, url, num):
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(0,num):
        resp = requests.get(url, verify=False)
        if resp.status_code != 200:
            print('请求失败！')
        file_dir = ''.join([dir, '/', str(i), '.jpg'])
        with open(file_dir, 'wb') as f:
            f.write(resp.content)

def getImage2(dir, url, num):
    ssl._create_default_https_context = ssl._create_unverified_context
    for i in range(num):
        resp = request.urlopen(url).read()
        file_dir = ''.join([dir, '/', str(i), '.jpg'])
        with open(file_dir, 'wb') as f:
            f.write(resp)



# ++++++++++++++
#dir0 = r'CT_anhui\testImages'
dir0 = r'/Users/huyajun/work/imageCodeKNN/aaa'
imgNum = 200
url2 = r'http://old.ddhong.com//authImg.do?tt=1512305572182'
# getImage1(dir0, url2, imgNum)

# ++++++++++++++

dir = r'/Users/huyajun/work/imageCodeKNN/bbb'

# dirnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# dirnames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
#             'W', 'X', 'Y', 'Z']

dirnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
            'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
            'x', 'y', 'z']

# dirnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
#             'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
#             'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
#             'X', 'Y', 'Z']


# dirnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a',
#             'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
#             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
#             'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
#             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
#             'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# makedir(dir, dirnames)

# ++++++++++++++

dir1 = r'CT_jiangsu\cropedImages'
dir2 = r'CT_jiangsu\trainData'

CT = 'qdnf,FJWQ,HNFQ,RKNB,AYVY,CULK,NKPB,NURH,NHLA,UCFR,XXYB,EPSA,LHWW,SEGP,' \
     'LVXP,ESGV,NECH,FUSS,PECL,XVQU,AMBP,BWCQ,TVQR,YKLK,BEYX,GRHK,QLMR,GNQX,' \
     'XSCJ,PXXM,LLXN,LFXS,LKNV,GLWW,RHKY,VCNV,XWTQ,YVMU,MFAS,YSGN,CAGR,VXLT,' \
     'QRMA,XJHR,PNKT,VHWH,FBWK,CNUS,RVCN,HTUA,BJWH,CAGF,HJYG,WBWC,AWGL,ACUK,' \
     'KWXA,NNXL,HTRG,YGKV,BRBT,BBHH,EEEQ,KCAF,FQSM,NWQN,HQTT,GGKV,XJHW,GLBH,' \
     'MMRE,PQNQ,QAKK,AVMQ,QUBR,TBBL,GACP,STHG,LGWR,BGEJ,TSGB,AASQ,FBQC,TFRE,' \
     'GJYS,KYUG,BETE,HSSA,LERV,CPMK,YYBY,TFLC,TWTW,XUAG,QMNH,FJUR,BXQS,ACWS,' \
     'USPC,VTEX,FXFP,SWEW,XVYX,QFFP,HQYG,PUXH,QUBS,LBMA,NLSY,CUFM,FXQQ,VUAW,' \
     'EGRE,VYCP,GPLU,WXHV,YEMW,XUVX,WAXL,FXXC,HCAW,XYJN,NETU,WBKN,NFTC,XEGF,' \
     'GSTM,PBWU,MVYF,HYJH,MGGM,WSVV,HHCC,WGRY,MTCV,QYHK,QCCY,XLTH,MHAQ,RKJN,' \
     'WFEX,KELM,NYRE,TQNT,NYEK,GBRX,RANG,RVXK,XVCE,NXHM,UEBB,MNES,PNKK,YPSF,' \
     'NVMT,XFLW,VYSB,PWMW,LTFE,NCAJ,XRCM,LVYJ,XFRT,SYER,LGYR,STEB,VLUP,GRGR,' \
     'MCXA,SJRA,AEJK,NHEE,XJVE,VGXF,PALR,XYBY,LLJG,NCUQ,ATEH,PBHU,WLTE,EQSS,' \
     'TATN,MEBA,SEHV,XCVW,HKAN,ERNH,MCRQ,YCHN,PKVC,VLGE,XWTK,TMMH,KRAP,MHXE,' \
     'PBCP,BHAL,EGGM'

# print(len(CT.replace(',', '')))
# classifyCropedImages(dir1, dir2, CT)

# ++++++++++++++

# rename(dir0)




#resp = requests.get(r'https://uam.ct10000.com/ct10000uam/validateImg.jsp', verify=False)