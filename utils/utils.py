import numpy as np 

def div_blocks(n_blocks , list_frames):

    blocks = []
    size_block = list_frames.shape[0] // n_blocks
    for i in range(n_blocks):
        st = i * size_block
        en = (i + 1) * size_block
        blocks.append(list_frames[st:en])
    return blocks