# -*- coding:utf-8 -*-
import os
import shutil
import cv2
import glob
import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from resnet18test32 import ResNet18
#np.set_printoptions(threshold=np.inf)

np.set_printoptions(suppress=True)
image_generator = ImageDataGenerator(rescale=1./255,
                                     validation_split = 0.3,)
image_generator1 = ImageDataGenerator(rescale=1./255,
                                    )
image_generator2 = ImageDataGenerator(rescale=1./255,)
data_dir = ('/home/stu1/xyx/dataset/hunhetest4/')
test_dir = ('/home/stu1/xyx/dataset/xingtest/')
no_dir=('/home/stu/xyx/dataset')
batch_size = 32
train_data_gen = image_generator.flow_from_directory(directory=data_dir,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                target_size=(224, 224),

                                                class_mode='categorical',
                                                subset='training')
val_data_gen = image_generator.flow_from_directory(directory=data_dir,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                target_size=(224, 244),

                                                class_mode='categorical',
                                                subset='validation')
#检验集
test_data_gen = image_generator1.flow_from_directory(directory=test_dir,
                                                     target_size=(224,224),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     shuffle=False)

no_data_gen = image_generator2.flow_from_directory(directory=no_dir,
                                                     target_size=(224,224),
                                                     batch_size=batch_size,
                                                     class_mode='categorical',
                                                     shuffle=False)
test_labels = test_data_gen.class_indices
print(test_labels)
classes = 3
test_label = test_data_gen.labels
real_labels = tf.one_hot(test_label,depth=classes)

NUM_TRAIN=train_data_gen.n
NUM_VAL=val_data_gen.n
NUM_No = no_data_gen.n
print(NUM_TRAIN,NUM_VAL)

iter1 = 0
while(iter1<50 or NUM_No>5) :

    model = ResNet18([2, 2, 2, 2])
    #模型训练
    model.compile(  # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(0.0001, decay=0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=['acc'])

    checkpoint_save_path = "./checkpoint/hunhetest4small.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_best_only=True)

    history = model.fit_generator(generator=train_data_gen,
                                  steps_per_epoch=NUM_TRAIN // batch_size,
                                  epochs=30,
                                  validation_data=val_data_gen,
                                  validation_steps=NUM_VAL // batch_size,
                                  verbose=1,
                                  callbacks=[cp_callback])
    # history = model.fit(x=train_data_gen, batch_size=2, epochs=5, validation_data=val_data_gen, validation_freq=1)
    # callbacks=[cp_callback])
    model.summary()
    ###################################semi supervise###########################
    pred = model.predict_generator(no_data_gen, verbose=1)
    pred1 = np.argmax(pred, axis=1)[max(pred) > 0.95]
    no_name = no_data_gen.filenames
    no_name = no_name[max(pred) > 0.95]
    find_data = dict(zip(no_name, pred1))
    for name, label in find_data.items():
        file = name[0:name.find('/')]
        imgname = name[name.find('/'):]
        print(name)
        shutil.move(name, data_dir + '/' + label + '/' + imgname)








    ###################################semi supervise###########################
    pred = model.predict_generator(no_data_gen,verbose=1)
    pred1 = np.argmax(pred,axis=1)[max(pred)>0.95]
    no_name = no_data_gen.filenames
    no_name = no_name[max(pred)>0.95]
    find_data = dict(zip(no_name,pred1))
    for name,label in find_data.items():
        file = name[0:name.find('/')]
        imgname = name[name.find('/'):]
        print(name)
        shutil.move(name,data_dir+'/'+label+'/'+imgname)

    iter1 = iter1+1
    print(NUM_No)

