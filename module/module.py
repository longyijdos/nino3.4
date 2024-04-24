import torch
import torch.nn as nn
from module.cnn_module import cnn_module
from module.rnn_module import rnn_module

class total_module(nn.Module):
    """
    conv_channel代表卷积层通道数，use_rnn代表是否使用rnn
    """
    def __init__(self,args):
        super(total_module, self).__init__()

        #两个cnn模型分别用于sst和ssh
        self.conv_sst=cnn_module(args)
        self.conv_ssh=cnn_module(args)

        self.use_rnn=False
        #启用rnn模型
        if args.use_rnn==True:
            self.use_rnn=True
            self.rnn=rnn_module(args)
            self.activation=nn.Tanh()
        #全连接层，得到后12个月的预测
        self.fc1 = nn.Linear(args.conv_channels * 2 * 6 * 18, args.features)
        self.fc2=nn.Linear(args.features,12)

    def forward(self,sstA,sshA):
        x1=self.conv_sst(sstA)
        x2=self.conv_ssh(sshA)
        x1=x1.view(x1.size(0),12,-1)
        x2=x2.view(x2.size(0),12,-1)
        x=torch.cat((x1,x2),dim=2)

        if self.use_rnn==True:
            x,_=self.rnn(x)
            x=self.activation(x)
            y=self.fc2(x[:,-1,:])

        else:
            x=x.view(x.size(0),-1)
            x=self.fc1(x)
            y=self.fc2(x)

        return y