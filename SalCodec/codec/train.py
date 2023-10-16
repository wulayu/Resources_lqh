import os
import argparse
import torch
import logging
import numpy as np
import datetime
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.nn import SyncBatchNorm
import torch.distributed as dist

from src.models.video_model import VideoCompressor
from src.models.image_model import IntraNoAR
from src.utils.misc import get_temperature
from src.dataset.dataset import MultiFrameTrainDataset
from src.models.focal_frequency_loss import FocalFrequencyLoss as FFL
from src.utils.common import dump_json

torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser(description='DVC reimplement')

parser.add_argument('--i_frame_model_path', default="checkpoints/acmmm2022_image_psnr.pth.tar", type=str)
parser.add_argument('-l', '--log', default='loguvg.txt',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = 'output/codec-10-10_10-56-47/epoch_0000_step009000.pt',
        help='load pretrain model')
parser.add_argument('-s', '--strict', default=True)
parser.add_argument('-tot_epoch', default=20, type=int)
parser.add_argument('-lr', default=0.0001, type=float)
parser.add_argument('--step_size', default=5, type=int)
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('-b', '--batch_size', default = 2, type=int)
parser.add_argument('-w', '--num_workers', default = 2, type=int)
parser.add_argument('--freeze', default = False, action="store_true")
parser.add_argument('--test_dataset', choices=["uvg", "jct1080p", "jct720p", "mcl1080p"], default="")
parser.add_argument('--output', default='./output', type=str)
parser.add_argument('--save_latest', default=True, type=bool)
parser.add_argument('--load_latest', default=False, type=bool)
parser.add_argument('--data_path', default="/home/ziming_wang/code/vimeo_septuplet", type=str)
parser.add_argument('--seq_len', default=2, type=int)

parser.add_argument('--temp_epoch', type=int, default=2, help='number of epochs for temperature annealing')
parser.add_argument('--temp_init', type=float, default=30.0, help='initial value of temperature')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

# DEVICE option
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--gpu_ids", default=-1, type=int)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (args.decay ** (epoch // args.decay_interval))
    # lr = args.lr ** epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_latest(model, optimizer, epoch, step, path):
    data = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer,
        'epoch': epoch,
        'step': step,
    }
    torch.save(data, os.path.join(path, "checkpoint_latest.pth.tar"))

def load_latest(path):
    data = torch.load(path, map_location='cpu')
    return data

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def freeze_encoder(net: VideoCompressor, args):
    # if freeze:
    #     net.round = True
    params = []
    new_dict = [
                'optic_flow',
                'mv_condition_prior_encoder',
                'mv_condition_prior_decoder',
                'mv_y_prior_fusion',
                'contextual_condition_prior_encoder',
                'contextual_condition_prior_decoder',
                'y_prior_fusion',
                ]
    train_dict = [
        'mv_encoder',
        'mv_decoder',
        'mv_hyper_prior_encoder',
        'mv_hyper_prior_decoder',
        "mv_y_prior_fusion",
        'mv_y_spatial_prior',
        'mv_y_mask_gen',
        'mv_y_q_basic',
        'mv_y_q_scale'
    ]
    for name, param in net.named_parameters():
        name = name.split('.')[0]
        if args.freeze:
            name = name.split('.')[0]
            if name in train_dict:
                params.append(param)
                print('rank', args.local_rank, 'required_grad:', name)
            else:
                param.requires_grad = False
        elif name in ['optic_flow']:
                param.requires_grad = False
        else:
            params.append(param)
            print('rank', args.local_rank, 'required_grad:', name)
    optimizer = optim.Adam(params, lr=args.lr)
    return net, optimizer    


if __name__ == "__main__":
    args = parser.parse_args()
    logger = logging.getLogger("VideoCompression")
    path_output = os.path.join(args.output, "codec-"+time.strftime("%m-%d_%H-%M-%S"))
    if args.local_rank <= 0 and not os.path.exists(path_output):
        logger.info("{} output dir {}".format(args.local_rank, path_output))
        os.makedirs(path_output)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.local_rank <= 0 and args.log != '':
        filehandler = logging.FileHandler(os.path.join(path_output, args.log))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("Codec")
    logger.info(args)
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device ='cuda:6'
    logger.info("DEVICE: " + device)

    i_model = IntraNoAR().to(device)
    state_dict = torch.load(args.i_frame_model_path, map_location='cpu')
    state_dict_ = {}
    for name, param in state_dict.items():
        name = name.replace("module.", "")
        state_dict_[name] = param
    i_model.load_state_dict(state_dict_)
    
    model = VideoCompressor()
    if args.pretrain != '' and os.path.exists(args.pretrain):
        print("loading pretrain : ", args.pretrain)
        state_dict = torch.load(args.pretrain, map_location='cpu')
        state_dict_ = {}
        for name, param in state_dict.items():
            name = name.replace("module.", "")
            # name = "module." + name
            state_dict_[name] = param
        model.load_state_dict(state_dict_, strict=args.strict)
    model = model.to(device)
    model, optimizer = freeze_encoder(model, args)
    dataset = MultiFrameTrainDataset(args.data_path, seq_len=args.seq_len)
    if args.local_rank >= 0:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        i_model = SyncBatchNorm.convert_sync_batchnorm(i_model)
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        i_model = DistributedDataParallel(i_model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        train_loader = DataLoader(dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampler=train_sampler,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False)
    else:
        model = model.to(device)
        i_model = i_model.to(device)
        train_loader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)

    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.local_rank <= 0:
        tb_logger = SummaryWriter('./events')
    print("train\tlr: {}".format(args.lr))

    cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
    train_loader_len = len(train_loader)
    start_epoch = args.start_epoch
    temp = get_temperature(0, start_epoch, train_loader_len,
                                temp_epoch=args.temp_epoch, temp_init=args.temp_init)
    if hasattr(model, 'module') and hasattr(model.module, "net_update_temperature"):
        model.module.net_update_temperature(temp)
    else:
        model.net_update_temperature(temp)
    i_model.eval()
    model.train()
    lamda = [85, 170, 380, 840]
    # lamda = [75.0, 94.4941, 119.0551, 150.0]
    i_frame_q_scales = [1.541, 1.083, 0.729, 0.500]
    # p_frame_y_q_scales = [1.2383, 0.9623, 0.7134, 0.5319]
    # p_frame_y_q_scales = [1.2083, 1.1029, 1.0067, 0.9189]
    # p_frame_mv_y_q_scales = [1.1844, 1.1044, 1.0107, 0.9189]
    # scale_wieght = [0.8 , 0.83, 0.86, 0.89, 0.93, 0.96, 1.  ]
    # frame_wieght = [2.2, 1.9291, 1.0123, 1.7832, 1.0006, 1.7404, 1.0]
    frame_wieght = [1.0] * 7
    global_step = 0
    if hasattr(model, 'module'):
        model.module.update_tau(1.0)
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.tot_epoch):
            t0 = datetime.datetime.now()
            res = defaultdict()
            for batch_idx, data in enumerate(train_loader):
                seqs, imgs = data
                if epoch < args.temp_epoch and hasattr(model, 'module') and hasattr(model.module, 'net_update_temperature'):
                    temp = get_temperature(batch_idx + 1, epoch, train_loader_len,
                                        temp_epoch=args.temp_epoch, temp_init=args.temp_init)
                    model.module.net_update_temperature(temp)
                for rate_idx in range(4):
                    rd_loss = 0
                    dpb = {
                        "ref_frame": None,
                        "ref_feature": None,
                        "ref_y": None,
                        "ref_mv_y": None,
                        }
                    for frame_idx in range(imgs.shape[1]):
                        global_step += 1
                        img = imgs[:, frame_idx].to(device)
                        if frame_idx == 0:
                            result = i_model(img, i_frame_q_scales[rate_idx])
                            # dpb['ref_frame'] = result['x_hat']
                            dpb['ref_frame'] = img
                            # I frame Code here
                            continue
                        result = model(img, dpb, rate_idx=rate_idx)
                        dpb = result['dpb']
                        rd_loss += lamda[rate_idx] * frame_wieght[frame_idx] * result['mse']+ result['bpp']
                        psnr = -10 * torch.log10(result['mse'].mean())
                        bpp = result['bpp'].mean()
                        bpp_mv = (result['bpp_mv_y'] + result['bpp_mv_z']).mean()
                        bpp_ctx = (result['bpp_y'] + result['bpp_z']).mean()
                        if args.local_rank <= 0:
                            tb_logger.add_scalar('lr', cur_lr, global_step)
                            tb_logger.add_scalar('rd_loss', rd_loss.mean() / frame_idx, global_step)
                            tb_logger.add_scalar('psnr', psnr, global_step)
                            tb_logger.add_scalar('bpp', bpp, global_step)
                            tb_logger.add_scalar('bpp_mv', bpp_mv, global_step)
                            tb_logger.add_scalar('bpp_ctx', bpp_ctx, global_step)
                            tb_logger.add_scalar('bpp_mv_y', result['bpp_mv_y'].mean(), global_step)
                            tb_logger.add_scalar('bpp_mv_z', result['bpp_mv_z'].mean(), global_step)
                            tb_logger.add_scalar('bpp_y', result['bpp_y'].mean(), global_step)
                            tb_logger.add_scalar('bpp_z', result['bpp_z'].mean(), global_step)
                        if (args.local_rank <= 0 and batch_idx % 100 == 0) or args.local_rank < 0:
                            log = 'Epoch: {:02}/{:02} Step:{:4}/{:4}\trate_idx: {}\tlr:{}\t'.format(epoch, args.tot_epoch, batch_idx, len(train_loader), rate_idx, cur_lr) + \
                                    'frame: {:2d}\t'.format(frame_idx) +\
                                    'Loss: {:.6f}\t'.format(rd_loss.mean() / frame_idx) +\
                                    'psnr:{:.3f}\t'.format(psnr.item()) +\
                                    'bpp:{:.4f}\t'.format(bpp.item()) +\
                                    'bppctx:{:.4f}\t'.format(bpp_ctx.item())+\
                                    'bppmv:{:.4f}\t'.format(bpp_mv.item())+\
                                    'bpp_mv_y:{:.4f}\t'.format(result['bpp_mv_y'].mean().item()) +\
                                    'bpp_mv_z:{:.4f}\t'.format(result['bpp_mv_z'].mean().item()) +\
                                    'bpp_y:{:.4f}\t'.format(result['bpp_y'].mean().item()) +\
                                    'bpp_z:{:.4f}\t'.format(result['bpp_z'].mean().item())
                            logger.info('{} '.format(args.local_rank) + log)
                    for i in range(len(seqs[0])):
                        seq_name = seqs[0][i][:-7]
                        loss = rd_loss.detach()[i].item()
                        res.setdefault(rate_idx, {}).update({seq_name: loss})
                    rd_loss = rd_loss.mean()
                    optimizer.zero_grad()
                    rd_loss.backward()
                    clip_gradient(optimizer, 0.5)
                    optimizer.step()
                if args.local_rank <= 0  and batch_idx % 1000 == 0 and batch_idx > 0:
                    torch.save(model.state_dict(), os.path.join(path_output, 'epoch_%04d_step%06d.pt' % (epoch, batch_idx)))
                # if batch_idx > 0 and batch_idx % 2000 == 0:
            lr_scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']
            if hasattr(model, 'module'):
                model.module.update_tau(model.module.tau * 0.5)
            if args.local_rank <= 0:
                torch.save(model.state_dict(), os.path.join(path_output, 'epoch_%04d.pt' % epoch))
                with open(os.path.join(path_output, 'epoch_%04d.json' % epoch), 'w') as fp:
                    dump_json(res, fp, float_digits=6, indent=2)
