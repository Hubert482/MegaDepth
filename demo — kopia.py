import torch
import sys
from torch.autograd import Variable
import numpy as np
import cv2
from options.train_options import TrainOptions
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from skimage.transform import resize
import time
import os
img_path = 'demo.jpg'
import threading
model = create_model(opt)

#torch.set_default_tensor_type(torch.cuda.CharTensor)



def test_simple(model):
    total_loss =0 
    toal_count = 0

    print("============================= TEST ============================")
    model.switch_to_eval()
    dep=True
    def save():
        io.imsave(f'{dir2}_depth.png', pred_inv_depth)
    #
    for file in os.listdir("frames/"):
        #print(os.path.join("frames/", file))
        dir2=os.path.join("frames/", file)
        img = np.float32(cv2.imread(f"{dir2}"))/255
        #print(img)
        dsize = (480, 256)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
        seconds = time.time()

        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        #print(f"imrad {img_path}")
        input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
        input_img = input_img.unsqueeze(0)

        input_images = Variable(input_img.cuda())
        pred_log_depth = model.netG.forward(input_images)
        pred_log_depth = torch.squeeze(pred_log_depth)

        pred_depth = torch.exp(pred_log_depth)

        # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
        # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
        pred_inv_depth = 0.5/pred_depth
        pred_inv_depth = pred_inv_depth.data.cpu().numpy()
        # you might also use percentile for better visualization

        pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
        seconds2 = time.time()
        print(f"frame time: {seconds2-seconds}")
        #print(pred_inv_depth)
        tsave = threading.Thread(target=save)
        tsave.start()
        dep=False

test_simple(model)
print("We are done")
