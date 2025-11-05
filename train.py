import os
import argparse

import numpy as np
import torch
from torch.autograd import Variable

from Datas import Vimeo_90K_dataset_generation
from Datas import RandomBalancedSampler
from Models import SmiffNet
from Utils import charbonier_loss
from Utils import ReduceLROnPlateau, AverageMeter

import warnings
warnings.filterwarnings("ignore")

def gen_dataset(dataset_path, train_list_txt, test_list_txt, batch_size):
    print("***Dataset setting***")
    train_set, test_set = Vimeo_90K_dataset_generation(dataset_path, train_list_txt, test_list_txt)
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size = batch_size,
        sampler = RandomBalancedSampler(train_set, int(len(train_set) // batch_size)),
        num_workers = 0, 
        pin_memory = True
    )

    val_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size = batch_size,
        num_workers = 8, 
        pin_memory = True
    )
    print('{} samples found, {} train samples and {} test samples '.format(
            len(test_set)+len(train_set), len(train_set), len(test_set)
          )
    )
    print("EPOCH is: "+ str(int(len(train_set) / batch_size)))
    return train_loader, val_loader, len(train_set), len(test_set)

def load_model(pretrained_weight_path = None):
    print("***Model setting***")
    model = SmiffNet(training = True)
    model = model.cuda()
    if torch.cuda.device_count() > 1 and isinstance(model,torch.nn.DataParallel):
        model = model.module
        model = torch.nn.DataParallel(model)
    if pretrained_weight_path is not None:
        pretrained_weight_name = pretrained_weight_path.split('/')[-1]
        print("Fine tuning on " + pretrained_weight_name)
        pretrained_dict = torch.load(pretrained_weight_path)    
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict) 
        
    model = model.train()
    for mm in model.modules():
        if isinstance(mm, torch.nn.BatchNorm2d):
            mm.eval()       
    return model

def set_train_schedule(model, factor, patience, base_lr, optical_flow_alpha, mask_alpha, structure_alpha):
    print("***Optimizer setting***")
    optimizer = torch.optim.Adamax(
        [
            # base module
            {'params': model.initScaleNets_filter.parameters(), 'lr': base_lr},
            {'params': model.initScaleNets_filter1.parameters(), 'lr': base_lr},
            {'params': model.initScaleNets_filter2.parameters(), 'lr': base_lr},
            {'params': model.ctxNet.parameters(), 'lr': base_lr},
            {'params': model.rectifyNet.parameters(), 'lr': base_lr},
            {'params': model.rectifyNet_it.parameters(), 'lr': base_lr},
            # mask module
            {'params': model.occmask.parameters(), 'lr': mask_alpha * base_lr},
            {'params': model.occmask_it1.parameters(), 'lr': mask_alpha * base_lr},
            {'params': model.occmask_it2.parameters(), 'lr': mask_alpha * base_lr},
            {'params': model.occdain.parameters(), 'lr': mask_alpha * base_lr},
            # optical-flow module (motion)
            {'params': model.flownets.parameters(), 'lr': optical_flow_alpha * base_lr},
            # structure module (structure)
            {'params': model.bdcn.parameters(), 'lr': structure_alpha * base_lr},
            {'params': model.structure_gen.parameters(), 'lr': structure_alpha * base_lr},
            {'params': model.detail_enhance.parameters(), 'lr': structure_alpha * base_lr},
            {'params': model.detail_enhance_it.parameters(), 'lr': structure_alpha * base_lr},
            # saliency mask for evaluation
            {'params': model.salNet.parameters(), 'lr': 0},
        ],
        lr = base_lr, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0
    )

    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = 'min',
        factor = factor, 
        patience = patience,
        verbose = True
    )
    print("LR is: "+ str(float(optimizer.param_groups[0]['lr'])))
    return optimizer, scheduler
    
def train(args):
    # load trained model
    model = load_model(args.pretrained_weight_path)
    # load images
    train_loader, val_loader, train_list_len, test_list_len = gen_dataset(
        args.dataset_path, args.train_list_txt, args.test_list_txt, args.batch_size
    )
    optimizer, scheduler = set_train_schedule(
        model, args.factor, args.patience, 
        args.base_lr, args.optical_flow_alpha, args.mask_alpha, args.structure_alpha
    )
    print("Num of EPOCH is: "+ str(args.numEpoch))
    
    log_data = list()
    training_losses = AverageMeter()
    for epoch in range(args.numEpoch):
        model = model.train()
        for mm in model.modules():
            if isinstance(mm, torch.nn.BatchNorm2d):
                mm.eval()

        # training process
        for i, (image_t0, image_t2, image_gt) in enumerate(train_loader):
            if i > train_list_len:
                break
            image_t0, image_t2, image_gt = image_t0.cuda(), image_t2.cuda(), image_gt.cuda()
            image_t0 = Variable(image_t0, requires_grad = False)
            image_t2 = Variable(image_t2, requires_grad = False)
            image_gt = Variable(image_gt, requires_grad = False)
            diff = model(torch.stack((image_t0, image_t2, image_gt), dim = 0))
            total_loss = charbonier_loss(diff, epsilon = 1e-6)
            training_losses.update(total_loss.item(), args.batch_size)
            
            if i % 10 == 0:
                print(
                    "Ep [" + str(epoch) + "/" + str(i) + "]" +
                    "\tTotal: " + str([round(x.item(),5) for x in [total_loss]]) +
                    "\tAvg. Loss: " + str([round(training_losses.avg, 5)])
                )
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if (epoch % args.model_save_interval == 0) or (epoch ==  args.numEpoch - 1):
            os.makedirs(args.model_save_path, exist_ok = True)
            torch.save(model.state_dict(), os.path.join(args.model_save_path, str(epoch) + ".pth"))        
        
        # validation process
        val_total_losses = AverageMeter()
        val_total_PSNR = AverageMeter()
        
        for i, (image_t0, image_t2, image_gt) in enumerate(val_loader):
            if i > test_list_len:
                break
            with torch.no_grad():
                image_t0, image_t2, image_gt = image_t0.cuda(), image_t2.cuda(), image_gt.cuda()
                diff = model(torch.stack((image_t0, image_t2, image_gt),dim = 0))
                total_loss = charbonier_loss(diff, epsilon = 1e-6)
                per_sample_pix_error = torch.mean(diff ** 2, dim = [1,2,3]).data
                PSNR_score = torch.mean(
                    20 * torch.log(1.0 / torch.sqrt(per_sample_pix_error))
                ) / torch.log(torch.Tensor([10]))
                val_total_losses.update(total_loss.item(), args.batch_size)
                val_total_PSNR.update(PSNR_score[0], args.batch_size) 
                       
        print(
            "\nEpoch " + str(int(epoch)) +
            "\tValidate Loss: " + str([round(float(val_total_losses.avg), 5)]) +
            "\tValidate PSNR: " + str([round(float(val_total_PSNR.avg), 5)])
        )
        
        log_data.append([epoch, training_losses.avg, val_total_losses.avg, val_total_PSNR.avg])
        os.makedirs(args.log_save_path, exist_ok = True)
        np.savetxt(
            os.path.join(args.log_save_path, 'log.txt'), np.array(log_data), fmt='%.8f', delimiter=','
        )
        training_losses.reset()
        print("\tFinished an epoch training")
        scheduler.step(val_total_losses.avg)
    
    print("*********Finish Training********")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMIFF')
    # dataset settings
    parser.add_argument('--dataset_path', type = str, help = 'dataset path')
    parser.add_argument('--train_list_txt', type = str)
    parser.add_argument('--test_list_txt', type = str)
    parser.add_argument('--batch_size', type = int, default = 1)
    # model settings
    parser.add_argument('--pretrained_weight_path',type = str, help = 'pretrained weight')
    # training settings:
    parser.add_argument('--base_lr', type = float, default = 0.0005)
    parser.add_argument('--optical_flow_alpha', type = float, default = 0.01)
    parser.add_argument('--structure_alpha', type = float, default = 0.01)
    parser.add_argument('--mask_alpha', type = float, default = 1.0)
    parser.add_argument('--patience', type = int, default = 4)
    parser.add_argument('--factor', type = float, default = 0.2)
    parser.add_argument('--numEpoch', type = int, default = 50)
    # model saving settings:
    parser.add_argument('--model_save_interval', type = int, default = 10)
    parser.add_argument('--model_save_path', type = str)
    parser.add_argument('--log_save_path', type = str)
    args = parser.parse_args()
    
    train(args)