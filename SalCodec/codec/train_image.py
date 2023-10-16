import torch
from torch.optim import Adam
import os
import time
import logging
from torch.utils.data import DataLoader
from src.models.image_model import IntraNoAR
from src.dataset.image_dataset import VimeoDataset
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.nn import SyncBatchNorm
import torch.distributed as dist

import time
import argparse
from src.utils.misc import get_temperature


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def main(args):
    logger = logging.getLogger("ImageCodec")
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
    model_save_path = os.path.join(args.output, 'image' + time.strftime("%m-%d_%H-%M-%S"))
    if args.local_rank <= 0 and not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if args.local_rank <= 0:
        tb_logger = SummaryWriter('./events')
    cur_lr = args.lr
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if args.local_rank <=0 and args.log != '':
        filehandler = logging.FileHandler(os.path.join(model_save_path, args.log))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.addHandler(stdhandler)
    logger.setLevel(logging.INFO)
    logger.info("Codec")
    logger.info(args)

    dataset = VimeoDataset(args.data_path)
    logger.info("length of dataset {}".format(len(dataset)))
    model = IntraNoAR()
    if args.pretrain != '' and os.path.exists(args.pretrain):
        logger.info('loading checkpoint from {}'.format(args.pretrain))
        state_dict = torch.load(args.pretrain, map_location='cpu')
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state, strict=True)
    if args.local_rank >= 0:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        find_unused_parameters=True)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            sampler=train_sampler,
                            pin_memory=True,
                            drop_last=False)
    else:
        model = DataParallel(model.cuda(), list(range(torch.cuda.device_count())))
        data_loader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=True)

    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    train_loader_len = len(data_loader)
    start_epoch = args.start_epoch
    if hasattr(model.module, "net_update_temperature"):
        temp = get_temperature(0, start_epoch, train_loader_len,
                                temp_epoch=args.temp_epoch, temp_init=args.temp_init)
        model.module.net_update_temperature(temp)

    model.train()
    optimizer = Adam(model.parameters(), lr = cur_lr)
    lrlambda = lambda epoch: 0.5 ** (epoch // 20)
    lrscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrlambda)
    lamda = [85, 170, 380, 840]
    q_scales = [1.541, 1.083, 0.729, 0.500]
    global_step = 0

    for epoch in range(args.tot_epoch):
        for batch_idx, img in enumerate(data_loader):
            if epoch < args.temp_epoch and hasattr(model.module, 'net_update_temperature'):
                temp = get_temperature(batch_idx + 1, epoch, train_loader_len,
                                    temp_epoch=args.temp_epoch, temp_init=args.temp_init)
                model.module.net_update_temperature(temp)
            for rate_idx in range(len(q_scales)):
                global_step =  global_step + 1
                result = model(img.cuda(), q_scales[rate_idx])
                rd_loss = lamda[rate_idx] * result['mse'].mean() + result['bpp'].mean()
                optimizer.zero_grad()
                rd_loss.backward()
                clip_gradient(optimizer, 0.5)
                optimizer.step()
                mse = result['mse'].mean()
                psnr = -10 * torch.log10(result['mse'].mean())
                bpp = result['bpp'].mean()
                bpp_y = result['bpp_y'].mean()
                bpp_z = result['bpp_z'].mean()
                if args.local_rank == 0:
                    tb_logger.add_scalar('lr', cur_lr, global_step)
                    tb_logger.add_scalar('loss', rd_loss, global_step)
                    tb_logger.add_scalar('psnr', psnr, global_step)
                    tb_logger.add_scalar('bpp', bpp, global_step)
                    tb_logger.add_scalar('bpp_y', bpp_y, global_step)
                    tb_logger.add_scalar('bpp_z', bpp_z, global_step)
                if batch_idx % 100 == 0 and args.local_rank <= 0:
                    log = 'Epoch: {:02}/{:02} Step:{:4}/{:4}\trate_idx: {}\tlr:{}\t'.format(epoch, args.tot_epoch, batch_idx, len(data_loader), rate_idx, cur_lr) + \
                                    'Loss: {:.6f}\t'.format(rd_loss.item()) +\
                                    'mse: {:6f}\t'.format(mse.item())+\
                                    'psnr:{:.3f}\t'.format(psnr.item()) +\
                                    'bpp_y:{:.4f}\t'.format(bpp_y.item()) +\
                                    'bpp_z:{:.4f}\t'.format(bpp_z.item())
                    logger.info(log)
            if batch_idx % 1000 == 0 and batch_idx > 0 and args.local_rank <= 0:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_%04d_step%06d.pt' % (epoch, batch_idx)))
                tb_logger.add_image('x_hat', result['x_hat'][0], global_step)
                tb_logger.add_image('img', img[0], global_step)
        lrscheduler.step()
        cur_lr = optimizer.param_groups[0]['lr']
        if args.local_rank <= 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'final_epoch_%04d.pt' % epoch))


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='ImageCodec')
    parser.add_argument('-l', '--log', default='loguvg.txt',
            help='output training details')
    parser.add_argument('-p', '--pretrain', default = 'checkpoints/epoch_0000_step031000.pt',
            help='load pretrain model')
    parser.add_argument('-tot_epoch', default=10000, type=int)
    parser.add_argument('-lr', default=0.0001, type=float)
    parser.add_argument('-decay', default=0.1, type=float)
    parser.add_argument('-decay_interval', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default = 6, type=int)
    parser.add_argument('-w', '--num_workers', default = 2, type=int)
    parser.add_argument('--output', default='./output', type=str)
    parser.add_argument('--data_path', default='/home/bjy/codes/lqh/vimeo_septuplet/sequences', type=str)
    parser.add_argument('--temp_epoch', type=int, default=10, help='number of epochs for temperature annealing')
    parser.add_argument('--temp_init', type=float, default=30.0, help='initial value of temperature')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    # DEVICE option
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--gpu_ids", default=-1, type=int)
    parser.add_argument("--world_size", default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    args = parser.parse_args()
    main(args)