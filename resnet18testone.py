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
from PIL import Image


image_generator = ImageDataGenerator(rescale=1./255,
                                     validation_split = 0.3,)
image_generator1 = ImageDataGenerator(rescale=1./255,
                                    )
data_dir = ('/home/stu1/xyx/dataset/yb3503dic/trainval/')
test_dir = ('/home/stu1/xyx/dataset/yb3503dic/test/')
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
                                                     shuffle=True)
test_labels = test_data_gen.class_indices
print(test_labels)

NUM_TRAIN=train_data_gen.n
NUM_VAL=val_data_gen.n
print(NUM_TRAIN,NUM_VAL)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#load data


#x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
#y_=tf.placeholder(tf.int32,shape=[None,],name='y_')


class ResnetBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet18(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


model = ResNet18([2, 2, 2, 2])

model.compile(#optimizer='adam',
              optimizer=tf.keras.optimizers.Adam(0.0001,decay=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['acc'])

checkpoint_save_path = "./checkpoint/resnet500500testkeshi1.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True)


history = model.fit_generator(generator = train_data_gen,
                           steps_per_epoch=NUM_TRAIN // batch_size,
                           epochs=20,
                           validation_data=val_data_gen,
                              validation_steps=NUM_VAL // batch_size ,
                              verbose=1,
                              callbacks=[cp_callback])
#history = model.fit(x=train_data_gen, batch_size=2, epochs=5, validation_data=val_data_gen, validation_freq=1)
                    #callbacks=[cp_callback])
model.summary()
#pred = model.predict_generator(test_data_gen,verbose=1)
#pred = np.argmax(pred,axis =1)
print('--------------teal--------------')
print(test_labels)
print('-------------pred----------------')
#print(pred)
# print(model.trainable_variables)
file = open('./350350test.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################


im = '/home/stu1/xyx/dataset/ybtest1/1/1.jpg'
img = Image.open(im)
#img=cv2.resize(img,(224,224))
img = img.resize((224,224))
img = np.array(img)
#img = np.reshape(224,224,3)
img = img/255.0
img = np.expand_dims(img,axis=0)
predictdan = model.predict(img)
predictdan = np.argmax(predictdan,axis = 1)
print(predictdan)


#plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


