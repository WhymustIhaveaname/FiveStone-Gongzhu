#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch,torchvision
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image,ImageDraw,ImageFont

def vis_kern(pattern,save_pattern,epoch_num):
    netdict=torch.load(pattern%(epoch_num),map_location="cpu")
    conv1=netdict["conv1.weight"]
    img1=torchvision.utils.make_grid(conv1[:,(2,0,1),:,:],nrow=8,padding=2,normalize=True,scale_each=True,pad_value=1.0)
    #torchvision.utils.save_image(img1,save_pattern%(epoch_num))
    expand=4
    img2=(img1.numpy()*255).astype(np.uint8).transpose([1,2,0])
    img2=Image.fromarray(img2)
    img2=img2.resize((66*expand,66*expand),resample=Image.NEAREST)
    img3=Image.new("RGB",(66*expand,66*expand+24),color=(255,255,255))
    img3.paste(img2,(0,0))
    d=ImageDraw.Draw(img3)
    # font is download from https://fonts.google.com/specimen/Ubuntu+Mono?preview.text_type=custom
    font=ImageFont.truetype("./Ubuntu_Mono/UbuntuMono-Regular.ttf",18)
    d.text((2*expand,66*expand),"Epoch = %03d"%(epoch_num),fill=(0,0,0),font=font)
    img3.save(save_pattern%(epoch_num))

def to_gif(pattern,iter,end_dup,dur,save_name):
    frames=[]
    for i in iter:
        frames.append(imageio.imread(pattern%(i)))
    for i in range(end_dup):
        frames.append(frames[-1])
        frames.insert(0,frames[0])
    imageio.mimsave(save_name,frames,'GIF',duration=dur)

if __name__=="__main__":
    #vis_evolution()
    #vis_kern("./logs/17/PV_resnet-16-15859346-%d.pkl","17th-%d.png",40,save_flag=False)
    num=26
    it=range(0,361,20)
    for i in it:
        vis_kern("./logs/%d/PV_resnet_wide-16-48902546-%%d.pkl"%(num),"%dth-%%d.png"%(num),i)
        #vis_kern("./logs/%d/PV_resnet-16-15859346-%%d.pkl"%(num),"%dth-%%d.png"%(num),i)
    to_gif("./%dth-%%d.png"%(num),it,1,0.5,"%dth-halfs.gif"%(num))
