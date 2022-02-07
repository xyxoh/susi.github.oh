import cv2
import numpy as np
from PIL import  Image
im = 'G:/resize/500.jpg'''
img = Image.open(im)
#img=cv2.resize(img,(224,224))
img = img.resize((224,224))
img = np.array(img)
print(img)
print(img.shape)
#img.save('G:/500resize224.jpg')