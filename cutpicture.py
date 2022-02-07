# -*- coding: utf-8 -*-
import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import time

#裁剪子图
#load data
w,h =1024,1024
def read_img(path,save_path):
    cate=[x for x in os.listdir(path) if os.path.isdir(path+x)]
    #x = [x for x in os.listdir(path) if os.path.isdir(path+x)]
    for idx,folder in enumerate(cate):
        file_name = save_path + folder
        os.makedirs(file_name)
        for im in os.listdir(path+folder):
            print('reading the images:%s'%(im))
            allpath = path+folder+'/'+im
            print(allpath)
            img=cv2.imread(allpath)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M_1 = cv2.getRotationMatrix2D(center, -8,1)
            img = cv2.warpAffine(img, M_1, (w, h))
            #cv2.imwrite('G:/cuttest1/' + str(folder) + im, img)
            #img=cv2.resize(img,(w,h))
            z=0
            #print(img.shape)
            for i in range(0,6):#裁剪500*500子图
                for j in range(0,6):
                    if (i+1)*500>img.shape[0]or 500*(j+1)>img.shape[1]:
                        break
                    imgfu = img[500*i:(i+1)*500,500*j:500*(j+1),:]
                    z=z+1
                    #print('G:/cuttest1/'+str(folder)+'/'+im[0:im.find(".")]+str(i)+'.jpg')
                    cv2.imwrite(save_path+str(folder)+'/%s_{}.jpg'.format(z) %im[0:im.find(".")], imgfu)

    #return imgs,np.asarray(labels,np.in
read_img('F:/shushinew2_san/','F:/susi500san/')#源数据，放置新数据位置