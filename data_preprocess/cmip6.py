import wget
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def download(args):

    #保存路径和下载网址
    save_path=args.save_path
    url=r"https://aims3.llnl.gov/thredds/fileServer/css03_data/CMIP6/CMIP/" + \
        r"NOAA-GFDL/GFDL-ESM4/historical/r1i1p1f1/Omon/%s/gr/v20190726/%s_Omon_GFDL" \
        r"-ESM4_historical_r1i1p1f1_gr_%s-%s.nc"

    #使用循环下载数据
    for var in ["tos", "zos"]:
        for year in range(1850, 2010, 20):
            time1 = str(year) + str(0) + str(1)
            time2 = str(year + 19) + str(12)
            NeedUrl = url % (var, var, time1, time2)
            print(NeedUrl)
            fileName = "GFDL-ESM4_%s_%s-%s.nc" % (var, time1, time2)
            print(fileName)
            path = save_path + '/' + fileName
            if not os.path.exists(path):
                wget.download(NeedUrl, path)

def preprocess(args):

    #原始路径和保存路径
    data_path = args.data_path
    save_path = args.save_path

    #按tos和zos分类
    file_list = os.listdir(data_path)
    tos_list = []
    zos_list = []
    for file_name in file_list:
        var_name = file_name.split('_')[1]
        if var_name=="tos":
            tos_list.append(xr.open_dataset(data_path + '/' + file_name)["tos"])
        else:
            zos_list.append(xr.open_dataset(data_path + '/'+file_name)["zos"])

    #把所有文件连接起来
    tos_array=xr.concat(tos_list,dim="time")
    zos_array=xr.concat(zos_list,dim="time")

    #统一时间范围
    time_range=pd.date_range(start="18500101", end="20091201", freq="MS")
    tos_array["time"]=time_range
    zos_array["time"]=time_range

    #获得nino3.4指数
    tosA = tos_array.groupby("time.month") - tos_array.groupby("time.month").mean()
    nino34I = tosA.loc[:, -5:5, 190:240].mean(dim=["lat", "lon"])

    #数据平滑
    nino34I = nino34I.rolling(time=3, center=True).mean()

    #插值
    lat = np.arange(-55, 60.1, 5, )
    lon = np.arange(0, 360, 5)
    tos_interped = tos_array.interp(lat=lat, lon=lon)
    zos_interped = zos_array.interp(lat=lat, lon=lon)

    #计算异常值
    tosA_interped = tos_interped.groupby("time.month") - tos_interped.groupby("time.month").mean()
    zosA_interped = zos_interped.groupby("time.month") - zos_interped.groupby("time.month").mean()

    #保存
    nino34ID = xr.Dataset({"nino34I": nino34I})
    nino34ID.to_netcdf(save_path + "/train_data/cmip6nino34I.nc" )

    tosAD = xr.Dataset({"tosA": tosA_interped})
    zosAD = xr.Dataset({"zosA": zosA_interped})
    tosAD.to_netcdf(save_path + "/train_data/cmip6tosA.nc")
    zosAD.to_netcdf(save_path + "/train_data/cmip6zosA.nc")

