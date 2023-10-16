import os
import time
from datetime import timedelta
import torch
from model import MobileSal
from KLDLoss import KLDLoss
from dataset import DHF1KDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import lr_scheduler
import torchvision


def main():
    ''' concise script for training '''
    data_path = '/data/lqh/Saliency/DHF1K/dataset/train'
    checkpoint = "/data/lqh/deepvideo/SalCodec/output/sal-09-03_10-42-14/epoch_0007.pt"
    path_output = './output'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    num_gpu = 2
    batch_size = 64  #海哥源代码为128
    num_epochs = 10
    path_output = os.path.join(path_output, "sal-"+time.strftime("%m-%d_%H-%M-%S"))
    if not os.path.isdir(path_output):
        os.makedirs(path_output)
    state_dict = torch.load(checkpoint, map_location='cpu')
    new_state_dict = {}
    for name, param in state_dict.items():
        new_state_dict[name.replace('module.', '')] = param
    model = MobileSal()
    model.load_state_dict(new_state_dict)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=2e-7)
    criterion = KLDLoss()
    train_loader = DataLoader(DHF1KDataset(data_path), batch_size=batch_size, shuffle=True, num_workers=24)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma = 0.8)
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model = model.cuda()
    model.train()

    start_time = time.time()
    for epoch in range(num_epochs):
        for step, data in enumerate(train_loader):
            img = data[0].cuda()
            sal = data[1].cuda()
            output = model(img)
            loss = criterion(output, sal)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print ('epoch: [%2d/%2d], step : [%.3f] loss: %.4f, %s' % \
                    (epoch, num_epochs, step / len(train_loader), loss.item(), timedelta(seconds=int(time.time()-start_time))), flush=True)
            if step % 500 == 0:
                torchvision.utils.save_image(img[0], "%s/epoch_%d_step_%d_img.png" % (path_output, epoch, step))
                torchvision.utils.save_image(sal[0], "%s/epoch_%d_step_%d_sal.png" % (path_output, epoch, step))
                torchvision.utils.save_image(output[0].clip(0, 1), "%s/epoch_%d_step_%d_pred.png" % (path_output, epoch, step))
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(path_output, 'epoch_%04d.pt' % epoch))


if __name__ == '__main__':
    main()

