
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.dataset.dataset import TestDataset
from tensorboardX import SummaryWriter
import datetime
from src.models.video_model import VideoCompressor
from prettytable import PrettyTable
import time
from tqdm import tqdm
torch.backends.cudnn.enabled = True
# gpu_num = 4
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_num = torch.cuda.device_count()


parser = argparse.ArgumentParser(description='DVC reimplement')

parser.add_argument('-l', '--log', default='loguvg.txt',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = 'output/codec-09-28_09-05-49/epoch_0003.pt',
        help='load pretrain model')
parser.add_argument('-b', '--batch_size', default = 1, type=int)
parser.add_argument('-rate_idx', default = 3, type=int)
parser.add_argument('-w', '--num_workers', default = 2, type=int)
parser.add_argument('--output', default='./output', type=str)


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()

if __name__ == "__main__":
    args = parser.parse_args()
    path_output = os.path.join(args.output, "codec-"+time.strftime("%m-%d_%H-%M-%S"))
    p_frame_y_q_scales = [1.2383, 0.9623, 0.7134, 0.5319]
    p_frame_mv_y_q_scales = [1.1844, 1.1044, 1.0107, 0.9189]
    model = VideoCompressor()
    if args.pretrain != '' and os.path.exists(args.pretrain):
        print("loading pretrain : ", args.pretrain)
        state_dict = torch.load(args.pretrain, map_location='cpu')
        new_state_dict = {}
        for name, param in state_dict.items():
            name = name.replace("module.", "")
            new_state_dict[name] = param
        model.load_state_dict(new_state_dict, strict=True)
    model = model.cuda()
    model = torch.nn.DataParallel(model, list(range(gpu_num)))

    tb_logger = SummaryWriter('./events')
    train_dataset = TestDataset("/home/ziming_wang/code/vimeo_septuplet")
    stepoch = 0
   
    dataloader = DataLoader(dataset = train_dataset, 
        shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True)
    model.eval()
    p_frame_y_q_scales = [1.2383, 0.9623, 0.7134, 0.5319]
    p_frame_mv_y_q_scales = [1.1844, 1.1044, 1.0107, 0.9189]
    tb = PrettyTable()
    tb.field_names = ["BPP", 'PSNR']
    for rate_idx in range(4):
        psnr_list = []
        bpp_list = []
        for input in tqdm(dataloader, desc="rate_idx:{}".format(rate_idx)):
            input = input
            input_image, ref_image = input[0].cuda(), input[1].cuda()
            dpb = {
                    "ref_frame": ref_image,
                    "ref_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
            result = model(input_image, dpb, p_frame_y_q_scales[rate_idx], p_frame_mv_y_q_scales[rate_idx])
            psnr = PSNR(input_image, result['x_hat'])
            # print('PSNR {:.3f}'.format(psnr))
            psnr_list.append(psnr)
            bpp_list.append(result['bpp'].item())
        psnr = np.mean(psnr_list)
        bpp = np.mean(bpp_list)
        tb.add_row([bpp, psnr])
        print("Average PSNR {:.3f}".format(psnr))
        print("Average BPP {:.3f}".format(bpp))
        print("Slope: {:.3f}".format(psnr / bpp))
    print(tb)

