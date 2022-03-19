import numpy as np
import torch

def SamplingPatches(location, frame_location, source_image, sample_space, low_img_level, high_img_level, patch_size):
    '''

    :param location: (x,y)
    :param source_image:
    :param sample_space: shape of attention, like [100, 100]
    :return: shape [bs, C, H, W]
    '''
    # sampled location transform horizon expansion
    if len(location.shape) == 2:
        location = location[0]
    row = np.floor(location // sample_space[1])
    col = location % sample_space[1]
    # location - axis switch！ and linear map back
    x = col * (np.around(2**low_img_level)) + frame_location[0] -int(np.around(2**high_img_level)* patch_size[1]//2)
    y = row * (np.around(2**low_img_level)) + frame_location[1] -int(np.around(2**high_img_level)* patch_size[0]//2)
    patch_list = []
    for idx in range(x.size):
        patch = source_image.extract_patches(x[idx], y[idx], level=high_img_level, size=patch_size, show_mode='channel_first')
        patch_list.append(patch)
    patch_list = torch.stack(patch_list)
    return patch_list

