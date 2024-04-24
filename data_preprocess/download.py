import argparse
import cmip6
import errstv6
import godas
import soda

def download(args):
    #下载数据
    if args.name =='cmip6':
        cmip6.download(args)
    elif args.name =='errstv6':
        errstv6.download(args)
    elif args.name =='godas':
        godas.download(args)
    elif args.name =='soda':
        soda.download(args)
    else:
        raise Exception('unsupported data')



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Download data")
    parser.add_argument("--name",'-n',type=str,choices=['cmip6','errstv6','godas','soda'],default='cmip6',help="data name")
    parser.add_argument("--save_path",'-s',type=str,default="./raw_data/cmip6",help="save path")
    args = parser.parse_args()
    download(args)