import wget
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def download(args):

    #保存路径和下载网站
    save_path=args.save_path
    url="ftp://ftp2.psl.noaa.gov/Datasets/godas/sshg.%s.nc"

    #使用循环下载数据
    for year in range(1980, 2020):
        NeedUrl=url%(year)
        print(NeedUrl)
        fileName="godas%sSSH.nc" % (year)
        print(fileName)
        path = save_path + '/' + fileName
        if not os.path.exists(path):
            wget.download(NeedUrl, path)

def preprocess(args):

    #原始路径和保存路径
    data_path = args.data_path
    save_path = args.save_path

    #将文件合并
    file_list = os.listdir(data_path)
    nc_list = []
    for file in file_list:
        nc = xr.open_dataset(data_path + '/' + file)["sshg"]
        nc_list.append(nc)
    ssh = xr.concat(nc_list, dim="time")

    #统一时间范围
    time_range=pd.date_range("19800101", "20191230", freq="MS")
    ssh["time"]=time_range

    #插值
    lat = np.arange(-55, 60.1, 5)
    lon = np.arange(0, 360, 5)

    ssh_interped = ssh.interp(lat=lat, lon=lon)

    #计算异常值
    sshA_interped=ssh_interped.groupby("time.month") - ssh_interped.groupby("time.month").mean()

    #保存
    sshAdataset=xr.Dataset({"sshA":sshA_interped})
    sshAdataset.to_netcdf(save_path + "/val_data/godassshA.nc")