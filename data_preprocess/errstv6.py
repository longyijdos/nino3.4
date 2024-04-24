import wget
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def download(args):

    #保存路径和下载网站
    save_path=args.save_path
    url="https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/ersst.v5.%s%s.nc"

    #使用循环下载数据
    for year in range(1870, 2020):
        for month in range(1, 13):
            month=str(month).zfill(2)
            NeedUrl=url%(year,month)
            print(NeedUrl)
            fileName="ersstv5_%s%s.nc" % (year,month)
            print(fileName)
            path=save_path + '/' + fileName
            if not os.path.exists(path):
                wget.download(NeedUrl, path)

def preprocess(args):

    #原始路径和保存路径
    data_path = args.data_path
    save_path = args.save_path

    #将文件合并
    file_list = os.listdir(data_path)
    nc_list=[]
    for file in file_list:
        nc=xr.open_dataset(data_path + '/' + file)["sst"]
        nc_list.append(nc)
    sst = xr.concat(nc_list, dim="time")

    #统一时间范围
    time_range=pd.date_range(start="18700101", end="20191201", freq="MS")
    sst["time"]=time_range

    #获得nino3.4指数
    sstA = sst.groupby("time.month") - sst.groupby("time.month").mean()
    nino34I = sstA.loc[:, 0, -5:5, 190:240].mean(dim=["lat", "lon"])
    nino34I = nino34I.rolling(time=3, center=True).mean()

    #插值
    lat = np.arange(-55, 60.1, 5)
    lon = np.arange(0, 360, 5)
    sst_interped = sst.interp(lat=lat, lon=lon, method="linear")

    #计算异常值
    sstA_interped = sst_interped.groupby("time.month")-sst_interped.groupby("time.month").mean()
    sstA_interped=sstA_interped[:,0]

    #保存
    Nino34IDataset = xr.Dataset({"nino34I": nino34I})
    sstDataset = xr.Dataset({"sstA": sstA_interped})
    Nino34IDataset.to_netcdf(save_path + "/train_data/errstv6nino34I.nc")
    sstDataset.to_netcdf(save_path + "/train_data/errstv6sstA.nc")
    Nino34IDataset.to_netcdf(save_path + "/val_data/errstv6nino34I.nc")
    sstDataset.to_netcdf(save_path + "/val_data/errstv6sstA.nc")