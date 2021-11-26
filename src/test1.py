#only use original frames to test PSNR
from dataset import SlidingWindowDataset
from e3d_lstm import E3DLSTM
from functools import lru_cache
from torch.utils.data import DataLoader
from utils import h5_virtual_file, window, weights_init
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from math import log10
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    
   
    parser.add_argument('--datapath','-p',type=str,default="/home/jinyang.liu/lossycompression/NSTX-GPI/nstx_gpi_float_tenth.dat")
    parser.add_argument('--hidden_size','-hs',type=int,default=64)
    parser.add_argument('--batchsize','-b',type=int,default=32)
    parser.add_argument('--window','-w',type=int,default=4)
    parser.add_argument('--horizon','-ho',type=int,default=1)
    parser.add_argument('--start_idx','-si',type=int,default=21000)
    parser.add_argument('--end_idx','-ei',type=int,default=30000)    
    parser.add_argument('--t_stride','-ts',type=int,default=1)
    parser.add_argument('--t_frames','-tf',type=int,default=2)
    parser.add_argument('--input_size','-is',type=int,nargs="+",default=[80,64])
    parser.add_argument('--layernum','-n',type=int,default=4)
    parser.add_argument('--norm_tanh','-t',type=bool,default=False)
    #parser.add_argument('--double','-d',type=int,default=0)
    parser.add_argument('--save','-s',type=str,default="ckpts_nstxgpi_tenthdefault/30.pt")
    





    args = parser.parse_args()
    dmax=4070
    dmin=0

    trainer = Trainer(batch_size=args.batchsize,window=args.window,horizon=args.horizon,t_stride=args.t_stride
        ,t_frames=args.t_frames,i_channel=1,i_size=args.input_size,tau=2,hidden_size=args.hidden_size,layernum=args.layernum)
    checkpoint = torch.load(args.save)
    epoch = checkpoint["epoch"]
    trainer.load_state_dict(checkpoint["state_dict"])
    trainer.batch_size=1
    data=np.fromfile(data_path,dtype=np.float32).reshape((-1,args.input_size[0],args.input_size[1]))[start_idx:end_idx]

    data=(data-dmin)/(dmax-dmin)
    if args.norm_tanh:
        data=data*2-1

    val_dataloader = trainer.get_trainloader(data, shuffle=False)

    trainer.eval()
    psnrs=[]
    print(trainer.device)
    with torch.no_grad():
        for i, (input, target) in enumerate(val_dataloader):
            frames_seq = []

            for indices in window(
                range(trainer.input_time_window),
                trainer.temporal_frames,
                trainer.temporal_stride,
            ):
                    # batch x channels x time x window x height
                frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])
            input_seq = torch.stack(frames_seq, dim=0).to(trainer.device)
            target = target.to(trainer.device)
            output = trainer(input_seq)
             
            if args.norm_tanh:
                target=(target+1)/2
                output=(output+1)/2
            target=target*(dmax-dmin)+dmin
            output=output*(dmax-dmin)+dmin

            mse = F.mse_loss(output , target )

            psnr=20*log10(torch.max(target)-torch.min(target)-10*log10(mse)).cpu().numpy()
            print(psnr)
            psnrs.append(psnr)

    psnrs=np.array(psnr)
    print(np.mean(psnrs))

