import argparse
import numpy as np
import torch
import scipy.stats as sps
from dataloader.get_dataloader import get_dataloader
from module.module import total_module
import logging
import sys


def create_logger(log_dir):
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 将日志记录到文件中
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)

    # 将日志打印到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def train(args):
    # 在gpu或者cpu上训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建日志
    log_dir = args.save_path + "/val.log"
    logger = create_logger(log_dir)

    # 加载数据集
    train_dataloader = get_dataloader(args)

    # 加载模型
    model = total_module(args)
    model.to(device)
    model_path = args.model_path + "/" + args.model_name
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # 损失函数
    criterion = torch.nn.MSELoss()

    # 总损失
    total_loss = 0

    # 保存真实值和预测值
    all_targets = []
    all_preds = []

    # 开始验证
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            targets = data["nino34I"].to(device)

            sstA = data["sstA"].to(device)
            sshA = data["sshA"].to(device)
            preds = model(sstA, sshA)

            loss = criterion(preds, targets)
            total_loss += loss.item()

            targets = targets.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()

            # 将所有的真实和预测分别保存下来
            # 删掉重复的月份
            start = 0
            end=len(targets)
            if i == 0:
                all_targets.extend(targets[0])
                all_preds.extend(preds[0])
                start += 1
            for j in range(start, end):
                all_targets.append(targets[j][-1])
                all_preds.append(preds[j][-1])

            # 计算每个batch的pear和prob
            for j in range(0, end):
                pear, prob = sps.pearsonr(preds[j], targets[j])
                logger.info(f"iteration {i + 1},batch {j+1}: Pearson correlation: {pear:.4f}, Probability: {prob:.4f}")

        avg_loss = total_loss / len(train_dataloader)

        total_pear, total_prob = sps.pearsonr(all_targets, all_preds)

    # 打印日志
    logger.info(f"Average loss: {avg_loss:.4f}")
    logger.info(f"total_pear: {total_pear:.4f}")
    logger.info(f"total_prob: {total_prob:.4f}")

    # 保存真实值和预测值
    targets_file = args.save_path + "/targets.txt"
    preds_file = args.save_path + "/predictions.txt"
    np.savetxt(targets_file, all_targets)
    np.savetxt(preds_file, all_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, required=True, help="path to data")
    parser.add_argument("--model_path", "-p", type=str, required=True, help="path to saved model")
    parser.add_argument("--model_name", "-n", type=str, choices=["best_model.pth", "final_model.pth"],
                        default="best_model.pth", help="name of model")
    parser.add_argument("--save_path", "-s", type=str, default='./save', help="path to save log")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="batch size for training")
    parser.add_argument("--conv_channels", "-c", type=int, default=32, help="number of channels")
    parser.add_argument("--features", "-f", type=int, default=32, help="number of features")
    parser.add_argument("--use_rnn", "-r", type=bool, default=False, help="use RNN model")
    parser.add_argument("--num_layers", "-l", type=int, default=2, help="number of layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--mode", "-m", type=str, default="val", help="mode for training")
    args = parser.parse_args()
    train(args)
