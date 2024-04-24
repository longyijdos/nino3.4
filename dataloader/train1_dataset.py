import torch
import torch.nn
import numpy as np
from torch.utils.data import Dataset
import xarray as xr

class train1_dataset(Dataset):
    def __init__(self,data_path,start=None,end=None):

        #start,end分别指数据开始和结束地年份
        super(train1_dataset, self).__init__()

        #加载数据
        nino34I_path=data_path+"/cmip6nino34I.nc"
        tosA_path=data_path+"/cmip6tosA.nc"
        zosA_path=data_path+"/cmip6zosA.nc"

        #将nan填充为0
        self.nino34I=xr.open_dataset(nino34I_path)["nino34I"]
        self.tosA=xr.open_dataset(tosA_path)["tosA"].fillna(0)
        self.zosA=xr.open_dataset(zosA_path)["zosA"].fillna(0)

        #截取选定时间内地数据
        if start is not None:
            time_range=(self.nino34I["time"].dt.year >= start) & (self.nino34I["time"].dt.year <= end)
            self.nino34I=self.nino34I[time_range]
            self.tosA=self.tosA[time_range]
            self.zosA=self.zosA[time_range]

        #定义时间长度
        self.time_len=self.nino34I.shape[0]

    def __getitem__(self,idx):

        #加载前12个月的sst和ssh
        x1=np.array(self.tosA[idx:idx+12])
        x2=np.array(self.zosA[idx:idx+12])
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
        return int(self.time_len - 24)#cmip最后一个月的数据为nan