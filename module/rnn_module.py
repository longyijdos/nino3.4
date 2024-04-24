import torch.nn as nn

def rnn_module(args):
    #定义rnn模型
    input = int(2*6*18*args.conv_channels/12)
    rnn=nn.LSTM(input_size=input, hidden_size=args.features, num_layers=args.num_layers, batch_first=True)
    return rnn