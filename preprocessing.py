# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:51:19 2021

@author: Hongmin
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
## this data has been splited.

# DATASET_PATH = "./kaggle_bee_vs_wasp"
# label_csv = os.path.join(DATASET_PATH, 'labels.csv')
# label_df = pd.read_csv(label_csv)
# label_df.head(10)

# label_df.set_index('id')

def set_train_type(row):
    if row['is_validation'] == 0 and row['is_final_validation'] == 0:
        return 'train'
    if row['is_validation'] ==1:
        return 'validation'
    else: return 'test'


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]        
 
# label_df['type'] = label_df.apply(set_train_type, axis=1)
# print('Number values of each type')
# label_df['type'].value_counts()

## print('train : val : test = 8 : 1 : 1')


def train_image_generator(data, DATASET_PATH, TARGET_SIZE):
    generator = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=45,
                                   zoom_range=0.3,
                                   shear_range=0.1,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)
    
    datagen = generator.flow_from_dataframe(data,
                                            directory=DATASET_PATH, 
                                            x_col='path',
                                            y_col='label',
                                            target_size=TARGET_SIZE,
                                            seed=42)
    return datagen

def test_image_generator(data, DATASET_PATH, TARGET_SIZE):
    generator = ImageDataGenerator(rescale=1./255.)
    
    datagen = generator.flow_from_dataframe(data,
                                            directory=DATASET_PATH, 
                                            x_col='path',
                                            y_col='label',
                                            target_size=TARGET_SIZE,
                                            seed=42)
    return datagen