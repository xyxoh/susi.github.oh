import os
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
#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(suppress=True)
image_generator = ImageDataGenerator(rescale=1./255,
                                     validation_split = 0.3,)
image_generator1 = ImageDataGenerator(rescale=1./255,
                                    )
data_dir = ('/home/stu1/xyx/dataset/hunhetest2/')
test_dir = ('/home/stu1/xyx/dataset/xingtest/')
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
                                                target_size=(224, 224),

                                                class_mode='categorical',
                                                subset='validation')
#检验集
test_data_gen = image_generator1.flow_from_directory(directory=test_dir,
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
print(NUM_TRAIN,NUM_VAL)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#load data


#x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
#y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        self.c1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')  # 卷积层1
        self.b1 = BatchNormalization()  # BN层1
        self.a1 = Activation('relu')  # 激活层1
        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', )
        self.b2 = BatchNormalization()  # BN层1
        self.a2 = Activation('relu')  # 激活层1
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d1 = Dropout(0.2)  # dropout层

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b3 = BatchNormalization()  # BN层1
        self.a3 = Activation('relu')  # 激活层1
        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.b4 = BatchNormalization()  # BN层1
        self.a4 = Activation('relu')  # 激活层1
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d2 = Dropout(0.2)  # dropout层

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b5 = BatchNormalization()  # BN层1
        self.a5 = Activation('relu')  # 激活层1
        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b6 = BatchNormalization()  # BN层1
        self.a6 = Activation('relu')  # 激活层1
        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')
        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b8 = BatchNormalization()  # BN层1
        self.a8 = Activation('relu')  # 激活层1
        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b9 = BatchNormalization()  # BN层1
        self.a9 = Activation('relu')  # 激活层1
        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')
        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b11 = BatchNormalization()  # BN层1
        self.a11 = Activation('relu')  # 激活层1
        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b12 = BatchNormalization()  # BN层1
        self.a12 = Activation('relu')  # 激活层1
        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')
        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(4096, activation='relu')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(4096, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(3, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y


model = VGG16()
model.compile(#optimizer='adam',
             optimizer=tf.keras.optimizers.Adam(0.001,decay=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['acc'])

checkpoint_save_path = "./checkpoint/vgg16tes5.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True)


history = model.fit_generator(generator = train_data_gen,
                           steps_per_epoch=NUM_TRAIN // batch_size,
                           epochs=50,
                           validation_data=val_data_gen,
                              validation_steps=NUM_VAL // batch_size ,
                             verbose=1,
                             callbacks=[cp_callback])
#history = model.fit(x=train_data_gen, batch_size=32, epochs=5, validation_data=val_data_gen, validation_freq=1,
                    #callbacks=[cp_callback])
model.summary()

#################################test################################
pred = model.predict_generator(test_data_gen,verbose=1)
pred1 = np.argmax(pred,axis=1)
print('--------------teal--------------')
#print(test_labels)
print('-------------pred----------------')
#print(pred)
#print(pred1)
# print(model.trainable_variables)
##############################save##################################
file = open('./500500muhu.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(6,4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
#plt.savefig('./picture/total.png')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
#plt.show()
plt.savefig('./picture/total.png')




plt.figure(figsize=(6,4))
plt.subplot(1, 2, 1)
plt.axes(yscale = "log")
#plt.ylim(0.9,1.0)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
#plt.show()
plt.savefig('./picture/acc.png')


plt.subplot(1, 2, 2)
plt.axes(yscale = "log")
#plt.ylim(0.06,0.08)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
#plt.show()
plt.savefig('./picture/loss.png')




