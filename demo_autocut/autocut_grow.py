# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import PIL.Image
from PIL import Image
import cv2
import math
import datetime
import subprocess
import os
import argparse
# import sys

starttime = datetime.datetime.now()

# 输入路径、输出路径参数调用
parser = argparse.ArgumentParser(description='This is an autocut program.')
parser.add_argument('image_path')
# parser.add_argument('output_path')

args = parser.parse_args()

im_path = args.image_path
# out_path = args.output_path

###### 输入图像格式转换
# im_path = '/home/workspace/guoxin/autocut/color_test.eps'
# # im_path = '/home/data/logo_cut/regular1/color/631a61a265b6467991ef2f5cccb67920.eps'
# out_path = '/home/workspace/guoxin/autocut/test/'

# 获取输入图像名称
(filepath,filename)=os.path.split(im_path)
image_name = os.path.splitext(filename)[0]

# 创建以图像名命名的文件夹
result_path = filepath + '/' + image_name + '/'
# png_path = result_path + 'png/'
# eps_path = result_path + 'eps/'

# 判断路径是否存在，如不存在则创建
def mkdir(path):
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path)
        print path + u' 创建成功'
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path + u' 目录已存在'
        return False

mkdir(result_path)
# mkdir(png_path)
# mkdir(eps_path)

im = Image.open(im_path)
# im.save(result_path + 'demo.jpg', 'JPEG')

###### 区域生长
im_g = PIL.Image.open(im_path)
im_g = im_g.convert('L')
im_g = np.array(im_g)

seed = (1, 1)#选取种子
mask = np.zeros(im_g.shape[:2], dtype=np.uint8)
mask[seed] = 1
vect = [seed]
area = [im_g[seed]]

while vect:
    n = len(vect)
    mean = np.sum(np.array(area), axis=0) / len(area)
    for i in range(n):
        seed = vect[i]
        s0 = seed[0]
        s1 = seed[1]
        for p in [
            (s0 - 1, s1 - 1),
            (s0 - 1, s1),
            (s0 - 1, s1 + 1),
            (s0, s1 - 1),
            (s0, s1 + 1),
            (s0 + 1, s1 - 1),
            (s0 + 1, s1),
            (s0 + 1, s1 + 1)
        ]:
            if p[0] < 0 or p[0] >= im_g.shape[0] or p[1] < 0 or p[1] >= im_g.shape[1]:
                continue
            if mask[p] == 1:
                continue
            # 区域生长条件: 灰度值差值小于等于 5
            if abs(mean - im_g[p]) <= 5:
                mask[p] = 1
                vect.append(p)
                area.append(im_g[p])
    vect = vect[n:]

mask = (1 - mask) * 255
im_g = PIL.Image.fromarray(mask)
im_g.save(result_path + 'growing.jpg')


###### canny轮廓检测
img = cv2.imread(result_path + 'growing.jpg',0)
edges = cv2.Canny(img,100,200)
print u'图片大小为' , edges.shape
# print edges.shape
# print edges

# 轮廓图存储
# cv2.imwrite(result_path + 'canny.jpg',edges)

#### 列处理
col_sum = np.sum(edges, axis=0)#axis=0, 按列相加 
# print col_sum

# 找出二值图中不为0的点
col_where_not = np.where(col_sum != 0)
col_where_not_ls = list(col_where_not[0])

#### 行处理
row_sum = np.sum(edges, axis=1)#axis=1, 按行相加 
# print col_sum

# 找出二值图中不为0的点
row_where_not = np.where(row_sum != 0)
row_where_not_ls = list(row_where_not[0])

print (edges.shape[1],edges.shape[0])
print (col_where_not_ls[-1], row_where_not_ls[-1])#第一块颜色区域坐标值

# print math.ceil(edges.shape[1] / col_where_not_ls[-1])
# print range(int(math.ceil(edges.shape[1] / col_where_not_ls[-1])))

###### 图片剪裁
for i in range(int(math.ceil(edges.shape[1] / col_where_not_ls[-1]))):
    for j in range(int(math.ceil(edges.shape[0] / row_where_not_ls[-1]))):
        part = im.crop((i * col_where_not_ls[-1], j * row_where_not_ls[-1], (i + 1) * col_where_not_ls[-1], (j + 1) * row_where_not_ls[-1]))
        file_name = result_path + 'image_' + str(j + 1) + str(i + 1) + '.jpg'
        # file_name_eps = eps_path + 'image_' + str(j + 1) + str(i + 1) + '.eps'
        part.save(file_name)

subprocess.Popen("rm" + " " + result_path + "growing.jpg", shell=True)

endtime = datetime.datetime.now()

print 'the spending time : ',(endtime-starttime).microseconds
