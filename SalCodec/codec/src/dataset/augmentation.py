import random
from numpy import pad
import torch
import torch.nn.functional as F

def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return (combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :])

def random_flip(images, labels):
    
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1
    if transforms and vertical_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])
    return images, labels

def random_crop_pad_filp(images, size):
    '''
    Arguments:
        images: [seq_len, chn, h, w]
        size: [max_seq_len, chn, h, w]
    '''
    image_shape = images.size()
    images = F.pad(images, (0, max(size[1], image_shape[3]) - image_shape[3], 0, max(size[0], image_shape[2]) - image_shape[2]))
    if random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
    if random.randint(0, 1) == 1:
        images = torch.flip(images, [3])
    freesize0 = random.randint(0, max(size[0], image_shape[2]) - size[0])
    freesize1 = random.randint(0,  max(size[1], image_shape[3]) - size[1])
    images = images[:, :, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return images
