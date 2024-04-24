import argparse
import cmip6
import errstv6
import godas
import soda


def preprocess(args):

    #预处理
    if args.name=="cmip6":
        cmip6.preprocess(args)
    elif args.name=="errstv6":
        errstv6.preprocess(args)
    elif args.name=="godas":
        godas.preprocess(args)
    elif args.name=="soda":
        soda.preprocess(args)
    else:
        raise ValueError("unsupported dataset")



if __name__=="__main__":
    parser = argparse.ArgumentParser("preprocess")
    parser.add_argument("--name",'-n',type=str,choices=['cmip6','errstv6','godas','soda'],default='cmip6',help="data name")
    parser.add_argument("--data_path",'-d',type=str,required=True,help="path to dataset")
    parser.add_argument("--save_path",'-s',type=str,required=True,help="path to saved dataset")
    args = parser.parse_args()
    preprocess(args)