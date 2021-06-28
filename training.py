# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:15:01 2021

@author: PC
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
from preprocessing import set_train_type
DATASET_PATH = "dataset"
label_csv = 'labels.csv'
label_df = pd.read_csv(label_csv)

label_df['type'] = label_df.apply(set_train_type, axis=1)

print('train : val : test = 8 : 1 : 1')

#%%
CLASSES = ['bee', 'wasp', 'insect']
n_classes = len(CLASSES)

data = label_df[label_df['label'].isin(CLASSES)]
data['path'] = data['path'].str.replace('\\', os.sep)

## train : valid : test = 8 : 1 : 1
train = data[data['type'] == 'train']
validation = data[data['type'] == 'validation']
test = data[data['type'] == 'test']


#%%
from preprocessing import train_image_generator, test_image_generator
TARGET_SIZE = (320, 240)

train_datagen = train_image_generator(train, DATASET_PATH, TARGET_SIZE)
validation_datagen = test_image_generator(validation, DATASET_PATH, TARGET_SIZE)
test_datagen = test_image_generator(test, DATASET_PATH, TARGET_SIZE)



#%%
from models import CustomCNN
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

# model = CustomCNN(TARGET_SIZE[0], TARGET_SIZE[1], 3, n_classes, 0.0001)
model = DenseNet121(weights=None, classes=3)

opt = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#%%
import tensorflow as tf
import datetime
import tensorflow.keras.callbacks as callbacks

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = callbacks.TensorBoard(logdir, histogram_freq=1)

early_stopping = callbacks.EarlyStopping(patience=7,
                                         monitor='val_loss',
                                         restore_best_weights=True)

reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)


# model_name = 'model{}.h5'

# checkpoint_path = "ckpt"
# model_checkpoint_callback = callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     save_weights_only=True,
#     monitor='val_loss',
#     save_best_only=True)

#%%
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 5, shuffle = True) 

X = pd.concat([train, validation], axis=0)
y = X['label']

#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagenerator = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=45,
                                   zoom_range=0.3,
                                   shear_range=0.1,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagenerator = ImageDataGenerator(rescale=1./255.)

#%%
BATCH_SIZE = 32
TARGET_SIZE = (320, 240)

cvscores = []
fold = 1
model_name=''
is_reduce_lr = False

for _ in range(3):
    if (_ == 2):
        is_reduce_lr = True
    for train_index, val_index in skf.split(X, y):
        train_data = X.iloc[train_index]
        val_data = X.iloc[val_index]
        print ('Fold: ',fold)
        
        
        train_datagen = train_datagenerator.flow_from_dataframe(train_data,
                                                          directory=DATASET_PATH, 
                                                          x_col='path', 
                                                          y_col='label',
                                                          class_mode="categorical",
                                                          target_size=TARGET_SIZE,
                                                          batch_size=BATCH_SIZE,
                                                          seed=42
                                                          ) 

        val_datagen = test_datagenerator.flow_from_dataframe(val_data,
                                                       directory=DATASET_PATH, 
                                                       x_col='path', 
                                                       y_col='label',
                                                       target_size=TARGET_SIZE,
                                                       class_mode="categorical",
                                                       batch_size=8,
                                                       seed=42
                                                       ) 

        if (os.path.isfile(model_name+'.h5')):
            model.load_weights(model_name)
        
        
        if is_reduce_lr:
            history = model.fit(train_datagen,
                                epochs=30,
                                callbacks=[early_stopping, reduce_lr, tensorboard_callback],
                                validation_data=val_datagen)
            
        else:
            history = model.fit(train_datagen,
                                epochs=30,
                                callbacks=[early_stopping, tensorboard_callback],
                                validation_data=val_datagen)
        
        
        
        # Save each fold model
        model_name = 'trained\\model_DenseNet3_fold_'+str(fold)+'.h5'
        model.save(model_name)
        
        # model.load_weights(model_name)
        # results = model.evaluate(validation_datagen)
        # results = dict(zip(model.metrics_names, results))
    	
        # cvscores.append(results['accuracy'])
        # cvscores.append(results['loss'])
    	
        tf.keras.backend.clear_session()
        
        # evaluate the model
        scores = model.evaluate(val_datagen, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        
        ## save the probability prediction of each fold in separate csv file
        # proba = model.predict(test_datagen, batch_size=None, steps=1)
        # labels=[np.argmax(pred) for pred in proba]
        # csv_name= 'submission_CNN_keras_aug_Fold'+str(fold)+'.csv'
        # create_submission(predictions=labels,path=csv_name)
        
        fold += 1

#%%
test_datagen = test_datagenerator.flow_from_dataframe(test,
                                                      directory=DATASET_PATH,
                                                      x_col='path',
                                                      y_col='label',
                                                      # don't shuffle
                                                      shuffle=False,
                                                      # use same size as in training
                                                      target_size=TARGET_SIZE)

test_loss, test_acc = model.evaluate(test_datagen, verbose=2)

#%%
plt.plot(cvscores)
plt.ylabel('Validation Accuracy')
plt.xlabel('epoch')

#%%
import cv2

def get_class_string_from_index(index):
   for class_string, class_index in test_datagen.class_indices.items():
      if class_index == index:
         return class_string

x, y = next(test_datagen)
image = x[:, :, :, :]
true_index = np.argmax(y, axis=1)
plt.figure(figsize=(15,10))

for i in range(len(x)):
    image = x[i, :, :, :]
    plt.subplot(4, 8, i+1)
    image = cv2.resize(image, (224, 224))
    plt.imshow(image)
    
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)

    plt.title("Label:" + get_class_string_from_index(true_index[i])+"\n"+"Predicted:"+
              get_class_string_from_index(predicted_index))
    plt.xlabel(prediction_scores.reshape(-1,)[predicted_index])
    plt.xticks([])
    plt.yticks([])



#%%
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(2, 1, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.ylabel('Accuracy')
# plt.ylim([min(plt.ylim()),1])
# plt.title('Training and Validation Accuracy')

# plt.subplot(2, 1, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.ylabel('Cross Entropy')
# plt.ylim([0,1.0])
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()


#%%
from tensorflow.keras.models import load_model

model_densenet = load_model()

train_loss, train_acc = model.evaluate(train_datagen, verbose=2)
test_loss, test_acc = model.evaluate(test_datagen, verbose=2)

### Bee vs Insect model results
# model_DenseNet2_fold_15.h5 : loss=0.2544 - accuracy=0.9124