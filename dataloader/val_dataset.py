import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset
import xarray as xr

class val_dataset(Dataset):
    def __init__(self,data_path):

        #start,end分别指数据开始和结束地年份
        super(val_dataset, self).__init__()

        #加载数据
        nino34I_path=data_path+"/errstv6nino34I.nc"
        sstA_path=data_path+"/errstv6sstA.nc"
        sshA_path=data_path+"/godassshA.nc"

        #将nan填充为0
        self.nino34I=xr.open_dataset(nino34I_path)["nino34I"]
        self.sstA=xr.open_dataset(sstA_path)["sstA"].fillna(0)
        self.sshA=xr.open_dataset(sshA_path)["sshA"].fillna(0)

        #截取选定时间内地数据
        sst_time_range=(self.sstA["time"].dt.year >= 1984) & (self.sstA["time"].dt.year <= 2008)
        ssh_time_range=(self.sshA["time"].dt.year >= 1984) & (self.sstA["time"].dt.year <= 2008)
        self.sstA=self.sstA[sst_time_range]
        self.sshA=self.sshA[ssh_time_range]
        self.nino34I=self.nino34I[sst_time_range]

        #定义时间长度
        self.time_len=self.nino34I.shape[0]

    def __getitem__(self,idx):

        #加载前12个月的sst和ssh
        x1=np.array(self.sstA[idx:idx+12])
        x2=np.array(self.sshA[idx:idx+12])
        sstA = torch.tensor(x1, dtype=torch.float32)
        sshA = torch.tensor(x2, dtype=torch.float32)

        #输出为后12个月的nino34
        y=np.array(self.nino34I[idx+12:idx+24])
        nino34I=torch.tensor(y,dtype=torch.float32)

        #返回一个字典
        data={}
        data["sstA"]=sstA
        data["sshA"]=sshA
        data["nino34I"]=nino34I
        return data

    def __len__(self):
        return int(self.time_len - 23)