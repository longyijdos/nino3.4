import wget
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def download(args):

    #保存路径和下载网址
    save_path=args.save_path
    url="https://coastwatch.pfeg.noaa.gov/erddap/griddap/hawaii_3e19_7ccd_16ff.nc?ssh%5B(1871-01-15T00:00:00Z):1:(2010-12-15T00:00:00Z)%5D%5B(-75.25):1:(89.25)%5D%5B(0.25):1:(359.75)%5D"

    #下载数据
    print(url)
    fileName="ssh.nc"
    print(fileName)
    path = save_path + '/' + fileName
    if not os.path.exists(path):
        wget.download(url, path)

def preprocess(args):

    #原始路径和保存路径
    data_path = args.data_path
    save_path = args.save_path

    #加载数据
    ssh=xr.open_dataset(data_path+"/ssh.nc", decode_times=False)["ssh"]

    #统一时间范围
    time_range=pd.date_range("18710101", "20101201", freq="MS")
    ssh["time"]=time_range

    #插值
    lat = np.arange(-55, 60.1, 5)
    lon = np.arange(0, 360, 5)
    ssh_interped = ssh.interp(latitude=lat, longitude=lon, method="linear")

    #计算异常值
    sshA_interped=ssh_interped.groupby("time.month") - ssh_interped.groupby("time.month").mean()

    #保存
    sshAdataset=xr.Dataset({"sshA": sshA_interped})
    sshAdataset.to_netcdf(save_path+"/train_data/sodasshA.nc")