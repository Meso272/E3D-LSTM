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
import math
from math import log10
class Trainer(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.cpu) else "cpu")
        dtype = torch.float
        self.dtype=dtype
        # TODO make all configurable:Done
        self.num_epoch =args.epoch
        self.batch_size = args.batch_size
        self.lr=args.lr
        self.input_time_window = args.window
        self.output_time_horizon = args.horizon
        self.temporal_stride = args.t_stride
        self.temporal_frames = args.t_frames
        self.time_steps = (
            self.input_time_window - self.temporal_frames + 1
        ) // self.temporal_stride
        self.hidden_size=args.hidden_size
        self.layernum=args.layernum
        # Initiate the network
        # CxT×H×W
        input_shape = (args.i_channel, self.temporal_frames, args.input_size[0], args.input_size[1])
        output_shape = (args.i_channel, self.output_time_horizon, args.input_size[0], args.input_size[1])

        self.tau = args.tau
        #hidden_size = 64
        kernel = (2, 5, 5) #Todo: Different kernel sizes
        lstm_layers = args.layernum

        self.encoder = E3DLSTM(
            input_shape, args.hidden_size, lstm_layers, kernel, self.tau
        ).type(dtype)
        self.decoder = nn.Conv3d(
            args.hidden_size * self.time_steps, output_shape[0], kernel, padding=(0, 2, 2)
        ).type(dtype)
        #what about adding an actv?
        # self.decoder = nn.Sequential(
        #   nn.Conv3d(hidden_size * self.time_steps, output_shape[0]),
        #  nn.ConvTranspose3d(output_shape[0], output_shape[0], kernel)
        # ).type(dtype)

        self.to(self.device)

        # Setup optimizer
        params = self.parameters(recurse=True)
        # TODO learning rate scheduler:Done
        # Weight decay stands for L2 regularization
        self.optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=0)
        #ADDed:LR-scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                             gamma = args.lr_gamma)
        self.apply(weights_init())

    def forward(self, input_seq):
        return self.decoder(self.encoder(input_seq))

    def loss(self, input_seq, target):
        output = self(input_seq)

        l2_loss = F.mse_loss(output , target )
        l1_loss = F.l1_loss(output, target )

        return l1_loss, l2_loss,output

    #All data part undone
    '''
    @property
    @lru_cache(maxsize=1)
    def data(self):
        taxibj_dir = "./data/TaxiBJ/"
        # TODO make configurable
        f = h5_virtual_file(
            [
                f"{taxibj_dir}BJ13_M32x32_T30_InOut.h5",
                f"{taxibj_dir}BJ14_M32x32_T30_InOut.h5",
                f"{taxibj_dir}BJ15_M32x32_T30_InOut.h5",
                f"{taxibj_dir}BJ16_M32x32_T30_InOut.h5",
            ]
        )
        return f.get("data")
    '''
    def get_trainloader(self, raw_data, shuffle=True):
        # NOTE note we do simple transformation, only approx within [0,1]
        dataset = SlidingWindowDataset(
            raw_data,
            self.input_time_window,
            self.output_time_horizon,
            lambda t: t ,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
        )

    def validate(self, val_dataloader):
        self.eval()

        sum_l1_loss = 0
        sum_l2_loss = 0
        with torch.no_grad():
            for i, (input, target) in enumerate(val_dataloader):
                frames_seq = []

                for indices in window(
                    range(self.input_time_window),
                    self.temporal_frames,
                    self.temporal_stride,
                ):
                    # batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])
                input = torch.stack(frames_seq, dim=0).to(self.device)
                target = target.to(self.device)
                #print(input.shape)
                l1_loss, l2_loss,output = self.loss(input, target)
                print(output.shape)
                print(target.shape)
                sum_l1_loss += l1_loss
                sum_l2_loss += l2_loss
                
        print(f"Validation L1:{sum_l1_loss / (i + 1)}; L2: {sum_l2_loss / (i + 1)}")
    
    def resume_train(self, args):
        # 2 weeks / 30min time step = 672
        self.data=np.fromfile(args.data_path,dtype=np.float32).reshape((-1,args.input_size[0],args.input_size[1]))[args.start_idx:args.end_idx]
        
        if args.data_max!=None:
            self.data=(self.data-args.data_min)/(args.data_max-args.data_min)
        if args.norm_to_tanh:
            self.data=self.data*2-1
        self.data=np.expand_dims(self.data,1)
        train_dataloader = self.get_trainloader(self.data[:-2000])#todo
        val_dataloader = self.get_trainloader(self.data[-2000:], False)#todo

        if args.resume:
            ckpt_path=args.save
            checkpoint = torch.load(ckpt_path)
            epoch = checkpoint["epoch"]

            self.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            ckpt_path=checkpoint["args"].save
        else:
            ckpt_path=args.save
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            epoch = 0
 
        while epoch < self.num_epoch:
            epoch += 1
            for i, (input, target) in enumerate(train_dataloader):
                frames_seq = []

                for indices in window(
                    range(self.input_time_window),
                    self.temporal_frames,
                    self.temporal_stride,
                ):
                    # batch x channels x time x window x height
                    frames_seq.append(input[:, :, indices[0] : indices[-1] + 1])

                input = torch.stack(frames_seq, dim=0).to(self.device)
                target = target.to(self.device)

                self.train()
                self.optimizer.zero_grad()
                l1_loss, l2_loss,_ = self.loss(input, target)
                loss = l1_loss + l2_loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()#added

                if i % 10 == 0:
                    print(
                        "Epoch: {}/{}, step: {}/{}, mse: {}".format(
                            epoch, self.num_epoch, i, len(train_dataloader), l2_loss
                        )
                    )
            if epoch % args.save_interval==0 or epoch==self.num_epoch:
                torch.save({"epoch": epoch, "state_dict": self.state_dict(),"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict(),
                    "args":args}, os.path.join(ckpt_path,"%d.pt" % epoch))
            self.validate(val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr','-l',type=float,default=1e-3)
    parser.add_argument('--data_path','-p',type=str,default="/home/jinyang.liu/lossycompression/NSTX-GPI/nstx_gpi_float_tenth.dat")
    parser.add_argument('--hidden_size','-hs',type=int,default=64)
    parser.add_argument('--batch_size','-b',type=int,default=32)
    parser.add_argument('--window','-w',type=int,default=4)
    parser.add_argument('--tau','-tau',type=float,default=2)
    parser.add_argument('--horizon','-ho',type=int,default=1)
    parser.add_argument('--start_idx','-si',type=int,default=0)
    parser.add_argument('--end_idx','-ei',type=int,default=20000)    
    parser.add_argument('--t_stride','-ts',type=int,default=1)
    parser.add_argument('--t_frames','-tf',type=int,default=2)
    parser.add_argument('--epoch','-e',type=int,default=100)
    parser.add_argument('--i_channel','-ic',type=int,default=1)
    parser.add_argument('--input_size','-is',type=int,nargs="+",default=[80,64])
    parser.add_argument('--layernum','-n',type=int,default=4)
    parser.add_argument('--lr_gamma','-lg',type=float,default=1)
    parser.add_argument('--data_max','-mx',type=float,default=4070)
    parser.add_argument('--data_min','-mi',type=float,default=0)
    parser.add_argument('--norm_to_tanh','-t',type=bool,default=False)
    parser.add_argument('--resume','-r',type=bool,default=False)
    #parser.add_argument('--double','-d',type=int,default=0)
    parser.add_argument('--save','-s',type=str,default="ckpts_nstxgpi_tenthdefault")
    parser.add_argument('--save_interval','-sv',type=int,default=5)
    parser.add_argument('--cpu','-c',type=bool,default=False)

    args = parser.parse_args()
    dmax=4070
    dmin=0
    
    if args.resume:
        save_interval=args.save_interval
        use_cpu=args.cpu
        ckpt=torch.load(args.save)
        ckpt_file=args.save
        args=ckpt["args"]
        args.save=ckpt_file
        args.save_interval=save_interval
        args.cpu=use_cpu
        args.resume=1
    print(args)

    trainer = Trainer(args)
    trainer.resume_train(args)
