# coding=gbk
#��ȡͼƬ
import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #ȡͼƬ��ԭʼ·��
        filenumber=len(pathDir)
        rate=0.1    #�Զ����ȡͼƬ�ı������ȷ�˵100�ų�10�ţ��Ǿ���0.1
        #picknumber=int(filenumber*rate) #����rate�������ļ�����ȡһ������ͼƬ
        picknumber = 110
        sample = random.sample(pathDir, picknumber)  #���ѡȡpicknumber����������ͼƬ
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = "G:/xingjiuhunheyabo/3/"    #ԴͼƬ�ļ���·��
	tarDir = 'G:/xingjiuhunheyabo/hunhe/3/'    #�ƶ����µ��ļ���·��
	moveFile(fileDir)
