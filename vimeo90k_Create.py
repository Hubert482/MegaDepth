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

import warnings
warnings.simplefilter('ignore')

def test_simple(model):
    total_loss =0 
    toal_count = 0

    print("============================= TEST ============================")
    model.switch_to_eval()
    def save():
        io.imsave(f'{dir}/im{leftframes}_depth.png', pred_inv_depth)
    for file in os.listdir("vimeo_triplet\sequences"):
        #print(os.path.join("vimeo_triplet\sequences", file))
        dir2=os.path.join("vimeo_triplet\sequences", file)
        for file in os.listdir(os.path.join("vimeo_triplet\sequences", file)):
            #print(os.path.join(dir2, file))
            dir=os.path.join(dir2, file)
            leftframes=1
            while leftframes<4:
                try:
                    print(f"{dir}\im{leftframes}.png")
                    img = np.float32(cv2.imread(f"{dir}\im{leftframes}.png"))/256
                    #print(img)
                    #dsize = (448, 256)
                    #img = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
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
                    pred_inv_depth = 0.75/pred_depth
                    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
                    # you might also use percentile for better visualization

                    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
                    seconds2 = time.time()
                    print(f"frame time: {seconds2-seconds}")
                    #print(pred_inv_depth)
                    tsave = threading.Thread(target=save)
                    tsave.start()
                    dep=False
                    leftframes+=1
                except:
                    print("error")

    # print(pred_inv_depth.shape)
    sys.exit()



test_simple(model)
print("We are done")
