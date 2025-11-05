import os
import argparse
from glob import glob

import numpy as np
import torch
from torch.autograd import Variable
from scipy.misc import imread, imsave

from Models import SmiffNet

import warnings
warnings.filterwarnings("ignore")
import time

def get_inputs(bi_image_folder_path, frame_name):
    I = imread(os.path.join(bi_image_folder_path, frame_name))
    if len(I.shape) == 2:
        I = np.stack([I, I, I], axis = 2)
        image_type = 'gray'
    else:
        image_type = 'color'
    I = np.transpose(I, (2, 0, 1)).astype("float32") / 255.0
    input_frame = torch.from_numpy(I).type(torch.cuda.FloatTensor)
    input_frame = Variable(torch.unsqueeze(input_frame, 0))
    input_frame = input_frame.cuda()
    return input_frame, image_type

def load_model(weight_path):
    torch.manual_seed(1)
    model = SmiffNet(training = False)
    if os.path.exists(weight_path):
        print("The testing model weight is: " + weight_path)
        model = model.cuda()
        pretrained_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.eval()
    else:
        print("**************************************")
        print("**** Can not find trained weight! ****")
        print("**************************************")
        raise NotImplementedError
    return model

def inference(args):
    # load trained model
    model = load_model(args.weight_path)
    torch.set_grad_enabled(False)
    # load images
    bi_image_folder_list = glob(os.path.join(args.dataset_path, '*'))
    os.makedirs(args.result_path, exist_ok = True)
    
    for bi_image_folder_path in bi_image_folder_list:
        # VFI inference
        start_time = time.time()
        input_t0, image_type = get_inputs(bi_image_folder_path, args.frame_t0_name)
        input_t2, _ = get_inputs(bi_image_folder_path, args.frame_t2_name)
        y_t1 = model(torch.stack((input_t0, input_t2), dim = 0))
        end_time = time.time()
        print("current image process time \t " + str(end_time - start_time)+"s" )
   
        # result saving
        y_t1 = y_t1.data.cpu().numpy()
        y_t1 = np.transpose(255.0 * y_t1.clip(0, 1.0)[0,:,:,:], (1, 2, 0))
        if image_type == 'gray':
            y_t1 = np.mean(y_t1, 2)
        folder_name = bi_image_folder_path.split('/')[-1]
        save_path = os.path.join(args.result_path, folder_name)
        os.makedirs(save_path, exist_ok = True)
        imsave(
            os.path.join(save_path, args.output_frame_t1_name), 
            np.round(y_t1).astype(np.uint8)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMIFF')
    # dataset settings
    parser.add_argument('--dataset_path', type = str, help = 'dataset path')
    parser.add_argument('--result_path', type = str, help = 'save path')
    parser.add_argument('--frame_t0_name', type = str, default = 'frame_00.png')
    parser.add_argument('--frame_t2_name', type = str, default = 'frame_02.png')
    parser.add_argument('--output_frame_t1_name', type = str, default = 'frame_01.png')
    # model settings
    parser.add_argument('--weight_path',type = str, help = 'trained weight for interence')
    args = parser.parse_args()
    
    inference(args)