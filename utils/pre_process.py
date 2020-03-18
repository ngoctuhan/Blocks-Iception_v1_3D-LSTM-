import os 
import cv2 
import numpy as np 
from sklearn.preprocessing import LabelBinarizer
from utils.params import Params

prs = Params()

def pre_process_sequence(folder_name = ''):
    
    train_file = prs.trainlist
    test_file  = prs.testlist

    X_train ,X_test , y_train, y_test  = [], [], [], []
    f = open(os.path.join(folder_name, train_file), 'r')

    for line in f:
        
        X_train.append(line.split(' ')[0])

        label = line.split('/')[0]
        y_train.append(label)
    
    f.close()

    f = open(os.path.join(folder_name, test_file), 'r')
    for line in f:
        
        X_test.append(line.split('\n')[0])

        label = line.split('/')[0]
        y_test.append(label)
    
    f.close()

    # onhot label 
    label_binary = LabelBinarizer()
    y_train = label_binary.fit_transform(y_train)
    y_train = np.asanyarray(y_train)

    for i in range(len(label_binary.classes_)):
        print(i, " : " , label_binary.classes_[i])
    y_test = label_binary.transform(y_test)
    y_test = np.asanyarray(y_test)

    return X_train , X_test , y_train, y_test








