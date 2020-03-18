
import os
import cv2 
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from models.networks import get_models_I3D_LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.pre_process import pre_process_sequence
from utils.params import Params
from utils.videoto3D import Videoto3D
from utils.sequence import UCF101_Sequence

model = get_models_I3D_LSTM()
prs = Params()


X_train, X_test, y_train, y_test =  pre_process_sequence()

loader = Videoto3D(n_frames=40, mode = 'rgb')

sequenceTrain = UCF101_Sequence(X_train, y_train, loader= loader, 
    folder_image = prs.path_img, batch_size= prs.batch_size, image_size=prs.image_size)
sequenceTest = UCF101_Sequence(X_test, y_test, loader= loader, 
    folder_image = prs.path_img, batch_size= prs.batch_size, image_size=prs.image_size)

# Trainning model 
val_acc   = 0
n_epouch  = prs.n_epouchs
num_step  = len(X_train) // 8

for i in range(n_epouch):

    if i <= 10:
        sgd = SGD(lr=prs.lr_1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
    elif i > 10 and i <= 15:
        sgd = SGD(lr=prs.lr_2, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
    else: 
        sgd = SGD(lr=prs.lr_3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    print("Epouch :" , i , "/20")
    for item in range(num_step):

        (images, labels) = sequenceTrain.__getitem__(item)
        his = model.train_on_batch(images, labels)
        
    res = model.evaluate_generator(sequenceTest, steps=len(X_test) // 8)
    print("Evaluate test data: ",res)
    if res[1] > val_acc:
        val_acc =  res[1]
        name_file = 'Block_I3D_LSTM-v1-best-weight_' + str(res[1]) + '.h5'
        model.save(name_file)

    sequenceTrain.on_epoch_end()

