# coding:utf-8
#bmp格式转jpg格式
import os
from PIL import Image
# bmp 转换为jpg
def bmpToJpg(file_path):
    cate = [ x for x in os.listdir(file_path) if os.path.isdir(file_path + x)]
    for idx,folder in enumerate(cate):
        newpath = "G:\\ybnew2_san\\"+str(folder)
        os.makedirs(newpath)
        for im in os.listdir(file_path+folder):
            allpath = file_path+folder

            print(allpath)
            #print(im)
            ago =im[0:im.find(".")]
            new = im[0:im.find(".")]+".jpg"
            print(new)
            #newFileName = new+".jpg"
            #print(newFileName)
            img = Image.open(allpath+'\\'+im)
            #print(folder+'\\'+im)
            #print(folder+"\\"+newFileName)
            #print("G:\\newtest\\"+str(folder)+"\\"+new)
            img.save("G:\\ybnew2_san\\"+str(folder)+"\\"+new)#保存地址
# 删除原来的位图
def deleteImages(file_path, imageFormat):
    cate = [file_path + x for x in os.listdir(file_path) if os.path.isdir(file_path + x)]
    for idx, folder in enumerate(cate):
        command = "del "+folder+"\\*."+imageFormat
        print("del " + folder + "\\*." + imageFormat)
        os.system(command)
def main():
    file_path = "G:\\image_2.0_san\\"#图片地址
    bmpToJpg(file_path)
    #deleteImages(file_path, "bmp")
if __name__ == '__main__':
    main()