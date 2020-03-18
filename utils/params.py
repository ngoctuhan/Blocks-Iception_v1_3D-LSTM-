import numpy as np 
class Params:

    # parameter of dataset
    path_img  = 'dataset/UCF-101'
    trainlist = 'dataset/trainlist.txt'
    testlist  = 'dataset/testlist.txt'
    classlnd  = 'dataset/classInd.txt'

    #parameter of model 
    image_shape = (224, 224)
    nframes     = 8 
    n_outputI3D = 1024
    nclasses    = 101
    nblocks     = 5

    #parameter of trainning
    n_epouchs   = 30 
    n_batchsize = 8
    lr_1        = 1e-2 # epouch 0  - 10
    lr_2        = 1e-3 # epouch 10 - 15 
    lr_3        = 1e-4 # epouch > 15
    moment      = 0.9
    decay       = 1e-6

    #parameter save model
    min_acc     = 0.8
    
    def __init__(self):
        super().__init__()