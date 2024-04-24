import torch.nn as nn

def cnn_module(args):
    #卷积模型
    conv=nn.Sequential(
        nn.Conv2d(12,args.conv_channels,kernel_size=(4,8),padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        nn.Conv2d(args.conv_channels,args.conv_channels,kernel_size=(4,2),padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        nn.Conv2d(args.conv_channels,args.conv_channels,kernel_size=(4,2),stride=(1,1),padding="same"),
        nn.Tanh()
    )
    return conv