# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:43:05 2023

@author: megav
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
#import tqdm
import random
from keras.utils import load_img
from keras.optimizers import SGD
from sklearn.model_selection import StratifiedKFold
#form keras_preprocessing import load_img
warnings.filterwarnings('ignore')
input_path=[]
label=[]
input_path1=[]
label1=[]

for class_name in os.listdir("D:/turbinefire/train"):
    for path in os.listdir("D:/turbinefire/train/"+class_name):
        if class_name=='not_fire':
            label.append(0)
        else:
            label.append(1)
        input_path.append(os.path.join( "D:/turbinefire/train",class_name,path))

traindf=pd.DataFrame()
traindf['images']=input_path
traindf['label']=label
traindf=traindf.sample(frac=1).reset_index(drop=True)

traindf['label']=traindf['label'].astype('str')


for class_name in os.listdir("D:/turbinefire/valid"):
    for path in os.listdir("D:/turbinefire/valid/"+class_name):
        if class_name=='not_fire':
            label1.append(0)
        else:
            label1.append(1)
        input_path1.append(os.path.join( "D:/turbinefire/valid",class_name,path))

validdf=pd.DataFrame()
validdf['images']=input_path1
validdf['label']=label1
validdf=validdf.sample(frac=1).reset_index(drop=True)

validdf['label']=validdf['label'].astype('str')


from keras.preprocessing.image import ImageDataGenerator
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

val_generator=ImageDataGenerator(rescale=1./255)


skf= StratifiedKFold(n_splits=5,random_state=7,shuffle=True)

from keras import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense

model=Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPool2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPool2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPool2D(2,2),
    Conv2D(128,(3,3),activation='relu'),
    MaxPool2D(2,2),
    Flatten(),
    Dense(512,activation='relu'),
    Dense(128,activation='relu'),
    Dense(1,activation='sigmoid')
    ])


epochs=25
learning_rate=0.01
decay_rate=learning_rate/epochs
momentum=0.8

sgd= SGD(lr=learning_rate,momentum=momentum,decay=decay_rate,nesterov=False)

model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])

scores=list()
FoldSetno = 0
for train_index, test_index in skf.split(input_path,label):
    #x_train,y_train,x_test,y_test=input_path[train_index],label[train_index],input_path[test_index],label[test_index]
    
    train_iterator=train_generator.flow_from_dataframe(
        traindf,
        x_col='images',
        y_col='label',
        target_size=(128,128),
        batch_size=512,
        class_mode='binary'
        )
    val_iterator=train_generator.flow_from_dataframe(
        validdf,
        x_col='images',
        y_col='label',
        target_size=(128,128),
        batch_size=512,
        class_mode='binary'
        )
  
    
    history=model.fit(train_iterator, epochs=25 ,validation_data=val_iterator) 
    scores.append({'acc':np.average(history.history['accuracy']),'val_acc':np.average(history.history['val_accuracy'])})
    FoldSetno+=1
    

#print(model.summary())
#history=model.fit(train_iterator, epochs=25,validation_data=val_iterator)
##
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(len(acc))


train_d=[]
validation_d=[]
plt.subplot(1,1,1)
for s in scores:
    train_d.append(s['acc'])
    validation_d.append(s['val_acc'])
print(train_d)
print(validation_d)

plt.plot(train_d, color='blue',label='train')
plt.plot(validation_d,color='red',label='validation')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()


image_path="D:/turbinefire/test/fire/fire.71.png"
img=load_img(image_path,target_size=(128,128))
img=np.array(img)
img=img/255.0
img=img.reshape(1,128,128,3)
pred=model.predict(img)
if(pred[0]>0.5):
    label='fire'
else:
    label='not_fire'
print (label)

filename="firemodelnew.h5"
model.save(filename)