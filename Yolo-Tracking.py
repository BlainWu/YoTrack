from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from YoloV3.util import *
import argparse
from YoloV3.darknet import Darknet
import pickle as pkl
import random

def arg_parse():
    """
    Parse arguements to the detect module
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "YoloV3/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "models/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on",
                        default = "jet.mp4", type = str)
    
    return parser.parse_args()
def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()
classes = load_classes("models/coco.names")
colors = pkl.load(open("YoloV3/pallete", "rb"))
num_classes = 80


#Set up the neural network
print("加载网络中.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("网络加载完成")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()


#Detection phase

videofile = args.videofile #or path to the video file.
cap = cv2.VideoCapture(videofile)
#cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), '无法打开视频文件或者摄像头'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes,
                               nms_conf = nms_thesh)
#, volatile = True
        if type(output) == int:
            frames += 1
            print("FPS:{:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        list(map(lambda x: write(x, frame), output))
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        fps = frames / (time.time() - start)
        t_size = cv2.getTextSize("FPS:{:5.2f}".format(fps),
                                 cv2.FONT_HERSHEY_DUPLEX,1,9)[0]
        cv2.putText(frame,"FPS:{:5.2f}".format(fps),(t_size[0],t_size[1]*3),
                    cv2.FONT_HERSHEY_DUPLEX,1,[0,255,0],1,9,False)

        print("Time:{:.6f}".format(time.time() - start))
        print("FPS:{:5.2f}".format(fps))
        cv2.imshow("frame", frame)
    else:
        break     






