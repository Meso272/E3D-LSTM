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

class Trainer(nn.Module):
    def __init__(self,epoch=100,lr=1e-3,batch_size=64,window=4,horizon=1,t_stride=1,t_frames=2,i_channel=1,i_size=[80,64],tau=2,hidden_size=64,layernum=4,lr_gamma=1):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.float
        self.dtype=dtype
        # TODO make all configurable:Done
        self.num_epoch =epoch
        self.batch_size = batch_size
        self.lr=lr
        self.input_time_window = window
        self.output_time_horizon = horizon
        self.temporal_stride = t_stride
        self.temporal_frames = t_frames
        self.time_steps = (
            self.input_time_window - self.temporal_frames + 1
        ) // self.temporal_stride

        # Initiate the network
        # CxT×H×W
        input_shape = (i_channel, self.temporal_frames, i_size[0], i_size[1])
        output_shape = (i_channel, self.output_time_horizon, i_size[0], i_size[1])

        self.tau = tau
        #hidden_size = 64
        kernel = (2, 5, 5) #Todo: Different kernel sizes
        lstm_layers = layer_num

        self.encoder = E3DLSTM(
            input_shape, hidden_size, lstm_layers, kernel, self.tau
        ).type(dtype)
        self.decoder = nn.Conv3d(
            hidden_size * self.time_steps, output_shape[0], kernel, padding=(0, 2, 2)
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
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                             gamma = lr_gamma)
        self.apply(weights_init())

    def forward(self, input_seq):
        return self.decoder(self.encoder(input_seq))

    def loss(self, input_seq, target):
        output = self(input_seq)

        l2_loss = F.mse_loss(output , target )
        l1_loss = F.l1_loss(output, target )

        return l1_loss, l2_loss

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
            shuffle=True,
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

                l1_loss, l2_loss = self.loss(input, target)
                sum_l1_loss += l1_loss
                sum_l2_loss += l2_loss

        print(f"Validation L1:{sum_l1_loss / (i + 1)}; L2: {sum_l2_loss / (i + 1)}")
    
    def resume_train(self, ckpt_path, data_path,start_idx,end_idx,resume=False,data_size=[80,64],data_max=None,data_min=None,norm_to_tanh=False,save_interval=20):
        # 2 weeks / 30min time step = 672
        self.data=np.fromfile(data_path,dtype=self.dtype).reshape((-1,data_size[0],data_size[1]))[start_idx:end_idx]
        
        if data_max!=None:
            self.data=(self.data-data_min)/(data_max-data_min)
        if norm_to_tanh:
            self.data=self.data*2-1

        train_dataloader = self.get_trainloader(self.data[:-672])#todo
        val_dataloader = self.get_trainloader(self.data[-672:], False)#todo

        if resume:
            checkpoint = torch.load(self, ckpt_path)
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            ckpt_path=os.path.dirname(ckpt_path)
        else:
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
                l1_loss, l2_loss = self.loss(input, target)
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
            if epoch % save_interval==0 or epoch==self.num_epoch-1:
                torch.save({"epoch": epoch, "state_dict": self.state_dict()}, os.path.join(ckpt_path,"%d.pt" % epoch))
            self.validate(val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr','-l',type=float,default=1e-3)
    parser.add_argument('--datapath','-p',type=str,default="/home/jliu447/lossycompression/NSTX-GPI/nstx_gpi_float.dat")
    parser.add_argument('--hidden_size','-hs',type=int,default=64)
    parser.add_argument('--batchsize','-b',type=int,default=64)
    parser.add_argument('--window','-w',type=int,default=4)
    parser.add_argument('--horizon','-ho',type=int,default=1)
    parser.add_argument('--start_idx','-si',type=int,default=0)
    parser.add_argument('--end_idx','-ei',type=int,default=10000)    
    parser.add_argument('--t_stride','-ts',type=int,default=1)
    parser.add_argument('--t_frames','-tf',type=int,default=2)
    parser.add_argument('--epoch','-e',type=int,default=100)
    parser.add_argument('--input_size','-is',type=int,nargs="+",default=[80,64])
    parser.add_argument('--layernum','-n',type=int,default=4)
    parser.add_argument('--lrgamma','-lg',type=float,default=1)
    parser.add_argument('--norm_tanh','-t',type=bool,default=False)
    parser.add_argument('--resume','-r',type=bool,default=False)
    #parser.add_argument('--double','-d',type=int,default=0)
    parser.add_argument('--save','-s',type=str,default="../ckpts_nstxgpi")
    parser.add_argument('--save_interval','-sv',type=int,default=20)
    
    args = parser.parse_args()
    dmax=4070
    dmin=0
    trainer = Trainer(self,epoch=args.epoch,lr=args.lr,batch_size=args.batchsize,window=args.window,horizon=args.horizon,t_stride=args.t_stride
        ,t_frames=args.t_frames,i_channel=1,i_size=args.input_size,tau=2,hidden_size=args.hidden_size,layernum=args.layernum,lr_gamma=args.lrgamma)
    trainer.resume_train(args.save ,args.datapath,start_idx=args.start_idx,end_idx=args.end_idx,
        resume=args.resume, data_size=args.input_size ,data_max=dmax,data_min=dmin,norm_to_tanh=args.norm_tanh,save_interval=args.save_interval)
