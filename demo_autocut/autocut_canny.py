# -*- coding: utf-8 -*-
# 背景单一、规律排列裁剪，取icon边缘裁剪，裁掉边缘线

from PIL import Image
import cv2
import numpy as np
import subprocess
import os
import sys
import argparse
import datetime

starttime = datetime.datetime.now()

parser = argparse.ArgumentParser(description='This is an autocut program.')
parser.add_argument('image_path')
# parser.add_argument('output_path')

args = parser.parse_args()

im_path = args.image_path
# out_path = args.output_path

# 输入图像格式转换
# im_path = '/home/workspace/guoxin/autocut/icon_test.eps'
# im_path = '/home/data/logo_cut/regular1/6c6d67b8f9b64d4e8a240cecfa1edf16.eps'
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

# canny边缘检测
img = cv2.imread(im_path,0)
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
# print col_where_not
# print col_where_not_ls

# 新建列表，位置相减
col_where_not_ls1 = col_where_not_ls[:]
col_where_not_ls2 = col_where_not_ls[:]

col_where_not_ls1.insert(0, 0)
col_where_not_ls2.append(0)

col_jian = list(map(lambda x: x[0] - x[1], zip(col_where_not_ls2, col_where_not_ls1)))
# print col_jian

col_jian_array = np.array(col_jian)

# 找到列表中不为1的点
col_place1 = np.where(col_jian_array != 1)
# print col_place1
col_place1_ls = list(col_place1[0])
# print col_place1_ls
# print len(col_place1_ls)

# 轮廓左边坐标
cut_left = list()
for i in range(len(col_place1_ls)-1):
    cut_left.append(col_where_not_ls[col_place1_ls[i]])
print cut_left

# 轮廓右边坐标
cut_right = list()
for i in range(1,len(col_place1_ls)):
    cut_right.append(col_where_not_ls[col_place1_ls[i] - 1])
print cut_right

#### 行处理
row_sum = np.sum(edges, axis=1)#axis=1, 按行相加 
# print row_sum

# 找到二值图中不为0的点
row_where_not = np.where(row_sum != 0)
row_where_not_ls = list(row_where_not[0])
# print row_where_not

# 新建列表，位置相减
row_where_not_ls1 = row_where_not_ls[:]
row_where_not_ls2 = row_where_not_ls[:]

row_where_not_ls1.insert(0, 0)
row_where_not_ls2.append(0)

row_jian = list(map(lambda x: x[0] - x[1], zip(row_where_not_ls2, row_where_not_ls1)))
# print row_jian

row_jian_array = np.array(row_jian)
# print row_jian_array

# 找到列表中不为1的点
row_place1 = np.where(row_jian_array != 1)
# print row_place1
row_place1_ls = list(row_place1[0])
# print len(row_place1_ls)

# 轮廓上部坐标
cut_top = list()
for i in range(len(row_place1_ls)-1):
    cut_top.append(row_where_not_ls[row_place1_ls[i]])
print cut_top

# 轮廓下部坐标
cut_low = list()
for i in range(1,len(row_place1_ls)):
    cut_low.append(row_where_not_ls[row_place1_ls[i] - 1])
print cut_low

# 剪裁图片
for i in range(len(cut_left)):
    for j in range(len(cut_top)):
        part = im.crop((cut_left[i], cut_top[j], cut_right[i], cut_low[j]))
        file_name = result_path + 'image_' + str(j+1) + str(i+1) + '.jpg'
        # file_name_eps = eps_path + 'image_' + str(j+1) + str(i+1) + '.eps'
        part.save(file_name)
        # subprocess.Popen("convert" + " " + file_name + " " + file_name_eps, shell=True)

endtime = datetime.datetime.now()

print 'the spending time : ',(endtime-starttime).microseconds
