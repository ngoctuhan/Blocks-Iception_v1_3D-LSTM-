import os 
import cv2 
import numpy as np
from tensorflow import keras 
from utils.utils import div_blocks 

class UCF101_Sequence(keras.utils.Sequence):


    '''
    Class use sequence dataset for trainning 
    '''
    def __init__ (self, list_IDs, list_lable, loader, folder_image ,batch_size, image_size,n_channels=3, shuffle=True, np_file = False):
        
        self.list_IDs = list_IDs          # list name file images
        self.folder_image = folder_image  # folder images  
        self.loader = loader              # object help cover video to array 
        self.batch_size = batch_size      # batch_size
        self.image_size =  image_size     # size of each image
        self.n_channels = n_channels    
        self.list_label = list_lable      # list label of all images
        self.shuffle = shuffle            # shuffle data after epouch true or no
        self.on_epoch_end()               # init shuffle
        
    def __load__ (self, ids_name):

        if self.loader.mode == 'opt':
            arr_frames = self.loader.covert2array(os.path.join(self.folder_image, ids_name.split('\n')[0]))
            return arr_frames

        if self.loader.mode == 'rgb':
            
            arr_frames = self.loader.covert2array(os.path.join(self.folder_image, ids_name.split('\n')[0]))
            # return [arr_frames[:8,:,:,:], arr_frames[8:16,:,:,:], arr_frames[16:24,:,:,:], arr_frames[24:32,:,:,:], arr_frames[32:,:,:,:]]
            return div_blocks(5, arr_frames)

    def __getitem__(self, index):
        
        if (index + 1) * self.batch_size > len(self.list_IDs):

            batch_index = self.indexes[index*self.batch_size:len(self.list_IDs)] 
        else:
            
            batch_index = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        image = []
        label  = []
        batch_file = [self.list_IDs[k] for k in batch_index]
        label = [self.list_label[k] for k in batch_index]

        for name_file in batch_file:
            
            arr_frames = self.__load__(name_file)
            image.append(arr_frames)

        # image = np.asanyarray(image)
        label = np.asarray(label)
        
        return np.array(image), label

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(   len(self.list_IDs) / float(self.batch_size)   ))

if __name__ == '__main__':

    # from pre_process import pre_process_sequence
    # from videoto3D import Videoto3D

    # loader =  Videoto3D(mode = 'rgb', n_frames =  32)
    # X_train, X_test, y_train, y_test = pre_process_sequence('F:/Paper Human Activity/UCF101_Train_List/ucfTrainTestlist', pair = 3)
    
    # sequenceTrain = UCF101_Sequence(X_train, y_train, loader= loader, 
    # folder_image = 'F:/Paper Human Activity/UCF101/UCF-101', batch_size= 16, image_size=(224,224))

    # sequenceTest =  UCF101_Sequence(X_test, y_test, loader= loader, 
    # folder_image = 'F:/Paper Human Activity/UCF101/UCF-101', batch_size=32, image_size=(224,224))
    # image, label = sequenceTrain.__getitem__(299)
    # image2, label2 = sequenceTest.__getitem__(100)
    # x,y = sequenceTrain.__getitem__(596)
    # print(x.shape)
    for i in range(0, 300):

        print("Run index : ", i)
        x,y = sequenceTrain.__getitem__(i)

        if x.shape != (32,32,224,224,3):
            print(i)
            break
    
    # print(len(X_train) // 32)

    # print(image2.shape)
    # print(label2.shape)
