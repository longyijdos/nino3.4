from dataloader.train1_dataset import train1_dataset
from dataloader.train2_dataset import train2_dataset
from dataloader.val_dataset import val_dataset
from torch.utils.data import DataLoader, ConcatDataset

def get_dataloader(args):

    #训练集
    if args.mode=="train":
        dataset1 = train1_dataset(args.data_path)
        dataset2 = train2_dataset(args.data_path)
        #将两种训练集拼接起来
        dataloader = DataLoader(ConcatDataset((dataset1, dataset2)), batch_size=args.batch_size,shuffle=True)

    #验证集
    elif args.mode=="val":
        dataset=val_dataset(args.data_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError("unknown mode")

    return dataloader