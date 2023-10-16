from logging import root
import os
import torch
from PIL import Image, ImageFile
import sys
from .fast_glcm import gen_features
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imageio.v2 as imageio
import numpy as np
import torch.utils.data as data
import glob
from ..utils.ms_ssim_torch import ms_ssim
from .augmentation import random_flip, random_crop_and_pad_image_and_labels, random_crop_pad_filp


class BaseDataSet(data.Dataset):
    def __init__(self, root="../data/UVG/CropVideos/", filelist="../data/UVG/originalv.txt", refdir='L12000', testfull=False):
        super(BaseDataSet, self).__init__()
        self.dataset_name = self.__class__.__name__
        with open(filelist) as f:
            folders = f.readlines()
        self.ref = []
        self.refbpp = []
        self.input = []
        self.hevcclass = []
        AllIbpp = self.getbpp(refdir)
        ii = 0
        for folder in folders:
            seq = folder.rstrip()
            seqIbpp = AllIbpp[ii]
            imlist = os.listdir(os.path.join(root, seq, seq))
            cnt = 0
            for im in imlist:
                if im[-4:] == '.png':
                    cnt += 1
            if testfull:
                framerange = cnt // 12
            else:
                framerange = 1
            for i in range(framerange):
                refpath = os.path.join(root, seq, refdir, 'im'+str(i * 12 + 1).zfill(4)+'.png')
                inputpath = []
                for j in range(12):
                    inputpath.append(os.path.join(root, seq, seq, 'im' + str(i * 12 + j + 1).zfill(3)+'.png'))
                self.ref.append(refpath)
                self.refbpp.append(seqIbpp)
                self.input.append(inputpath)
            ii += 1
        self.ref = self.ref


    def getbpp(self, ref_i_folder):
        raise NotImplementedError

    
    def __len__(self):
        return len(self.ref)

    def __getitem__(self, index):
        ref_image = imageio.imread(self.ref[index]).transpose(2, 0, 1).astype(np.float32) / 255.0
        h = (ref_image.shape[1] // 64) * 64
        w = (ref_image.shape[2] // 64) * 64
        ref_image = np.array(ref_image[:, :h, :w])
        input_images = []
        refpsnr = None
        refmsssim = None
        for filename in self.input[index]:
            input_image = (imageio.imread(filename).transpose(2, 0, 1)[:, :h, :w]).astype(np.float32) / 255.0
            if refpsnr is None:
                refpsnr = CalcuPSNR(input_image, ref_image)
                refmsssim = ms_ssim(torch.from_numpy(input_image[np.newaxis, :]), torch.from_numpy(ref_image[np.newaxis, :]), data_range=1.0).numpy()
            else:
                input_images.append(input_image[:, :h, :w])

        input_images = np.array(input_images)
        return input_images, ref_image, self.refbpp[index], refpsnr, refmsssim


class UVGDataset(BaseDataSet):
    def __init__(self, root="../data/UVG/CropVideos/", filelist="../data/UVG/originalv.txt", refdir='L12000', testfull=False):
        super(UVGDataset, self).__init__(root=root, filelist=filelist, refdir=refdir, testfull=testfull)

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [0.1504, 0.01745, 0.31756, 0.11921, 0.02103, 0.13817, 0.1392]# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.04171, 0.01282, 0.01033, 0.08448, 0.23296, 0.09118, 0.07624]# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = [0.0179, 0.00834, 0.00721, 0.05789, 0.1724, 0.05875, 0.04626]# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = [0.01067, 0.00562, 0.00536, 0.04143, 0.12788, 0.03649, 0.0268]# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp 


class JCT1080pDataset(BaseDataSet):
    def __init__(self, root="../data/JCT1080p/CropVideos/", filelist="../data/JCT1080p/originalv.txt", refdir='L12000', testfull=False):
        super(JCT1080pDataset, self).__init__(root=root, filelist=filelist, refdir=refdir, testfull=testfull)

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [0.12167881556919646, 0.11956796797495042, 0.07645100911458333, 0.1139898681640625]# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.06145252046130952, 0.05361531575520834, 0.04680989583333334, 0.07200480143229167]# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = [0.03605811709449405, 0.03225678943452381, 0.03035196940104166, 0.047237548828125]# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = [0.023238796657986113, 0.02102632068452381, 0.02004313151041667, 0.031479288736979166]# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp 


class JCT720pDataset(BaseDataSet):
    def __init__(self, root="../data/JCT720p/CropVideos/", filelist="../data/JCT720p/originalv.txt", refdir='L12000', testfull=False):
        super(JCT720pDataset, self).__init__(root=root, filelist=filelist, refdir=refdir, testfull=testfull)

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [0.028552201704545456, 0.024056818181818183, 0.024056818181818183, 0.024326171875]# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.0173046875, 0.023172052556818184, 0.013582208806818182, 0.013746271306818181]# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = [0.012318004261363636, 0.008398259943181819, 0.008806285511363638, 0.008910866477272728]# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = [0.009122514204545453, 0.0061439985795454545, 0.0061899857954545445, 0.006292258522727273]# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp 


class MCL1080pDataset(BaseDataSet):
    def __init__(self, root="../data/MCL1080p/CropVideos/", filelist="../data/MCL1080p/originalv.txt", refdir='L12000', testfull=False):
        super(MCL1080pDataset, self).__init__(root=root, filelist=filelist, refdir=refdir, testfull=testfull)

    def getbpp(self, ref_i_folder):
        Ibpp = None
        if ref_i_folder == 'H265L20':
            print('use H265L20')
            Ibpp = [0.035380045572916664, 0.49012325971554493, 0.8155591560132575, 0.31100704308712124, 0.27340047200520834, 0.04455910707131412]# you need to fill bpps after generating crf=20
        elif ref_i_folder == 'H265L23':
            print('use H265L23')
            Ibpp = [0.015949894831730767, 0.34788286258012824, 0.14272054036458331, 0.030692545572916667]# you need to fill bpps after generating crf=23
        elif ref_i_folder == 'H265L26':
            print('use H265L26')
            Ibpp = []# you need to fill bpps after generating crf=26
        elif ref_i_folder == 'H265L29':
            print('use H265L29')
            Ibpp = []# you need to fill bpps after generating crf=29
        else:
            print('cannot find ref : ', ref_i_folder)
            exit()
        if len(Ibpp) == 0:
            print('You need to generate I frames and fill the bpps above!')
            exit()
        return Ibpp 


class TrainDataset(data.Dataset):
    def __init__(self, path="../data/vimeo_septuplet", im_height=256, im_width=256):
        super().__init__()
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir=path)
        self.im_height = im_height
        self.im_width = im_width
        self.image_input_list = self.image_input_list
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="../data/vimeo_septuplet/sequences/"):
        with open(os.path.join(rootdir,"sep_trainlist.txt")) as f:
            seqs = f.readlines()
            seqs = [seq.strip() for seq in seqs]
        fns_train_input = []
        fns_train_ref = []
        for seq in seqs:
            imgs = glob.glob(os.path.join(rootdir, "sequences", seq, "*.png"))
            imgs.sort()
            for i, img in enumerate(imgs[2:]):
                refnumber = int(img[-5:-4]) - 2
                refname = img[0:-5] + str(refnumber) + '.png'
                if os.path.exists(img) and os.path.exists(refname):
                    fns_train_input.append(img)
                    fns_train_ref.append(refname)

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image, [self.im_height, self.im_width])
        input_image, ref_image = random_flip(input_image, ref_image)
        return input_image, ref_image


class MultiFrameTrainDataset(data.Dataset):
    def __init__(self, path="../data/vimeo_septuplet", im_height=256, im_width=256, seq_len=3):
        super().__init__()
        self.im_height = im_height
        self.im_width = im_width
        self.seq_len = seq_len
        self.seq_list = self.get_vimeo(rootdir=path)
        print("dataset find sequences: ", len(self.seq_list))

    def get_vimeo(self, rootdir="../data/vimeo_septuplet/sequences/"):
        seqs = []
        with open(os.path.join(rootdir, "sep_trainlist.txt"), 'r') as f:
            seqs = f.readlines()
            seqs = [seq.strip() for seq in seqs]
        # with open('hard_case.txt', 'r') as f:
        #     hard_cases = f.readlines()
        #     hard_cases = [case.strip() for case in hard_cases]
        # seqs.extend(hard_cases)
        seq_list = []
        print('scaning %s'%rootdir)
        for seq in seqs:
            imgs = glob.glob(os.path.join(rootdir, "sequences", seq, "*.png"))
            imgs.sort()
            seq_list.append(imgs)
        return seq_list

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):
        res = []
        seq = self.seq_list[index][::np.random.randint(1, 3)]
        # seq = self.seq_list[index]
        for i in range(self.seq_len):
            if i < len(seq):
                img_path = seq[i]
                img = imageio.imread(img_path).transpose(2, 0, 1)
                # img_feature = gen_features(img)
                img = img.astype(np.float32) / 255.0
                # img_feature = np.concatenate([img, img_feature], axis=0)
            # else:
                # img = np.zeros(img.shape)
            res.append(img)
        res = np.array(res)
        imgs = torch.from_numpy(res).float()
        imgs = random_crop_pad_filp(imgs, [self.im_height, self.im_width])
        return seq[:self.seq_len], imgs


class TestDataset(data.Dataset):
    def __init__(self, path="../data/vimeo_septuplet"):
        super().__init__()
        self.image_input_list, self.image_ref_list = self.get_vimeo(rootdir=path)
        self.image_input_list = self.image_input_list
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="../vimeo_septuplet/sequences/"):
        with open(os.path.join(rootdir,"test.txt")) as f:
            imgs = f.readlines()[:2000]
            fns_train_input = [os.path.join(rootdir, 'sequences', seq.strip()) for seq in imgs]
        def replace_fun(img):
            refnumber = int(img[-5:-4]) - 2
            refname = img[0:-5] + str(refnumber) + '.png'
            return refname
        fns_train_ref = list(map(replace_fun, fns_train_input))
        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)
        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()
        return input_image, ref_image     

if __name__ == "__main__":
    path = '/data/lqh/deepvideo/DVC/DVC/data/vimeo_septuplet'
    dataset = MultiFrameTrainDataset(path)
    for idx, imgs in enumerate(dataset):
        print(idx, imgs.shape)