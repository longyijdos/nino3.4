import argparse
import torch
from dataloader.get_dataloader import get_dataloader
from module.module import total_module
import logging
import sys

def create_logger(log_dir):

    #配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #将日志记录到文件中
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)

    #将日志打印到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    #定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    #添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def train(args):

    #在gpu或者cpu上训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #创建日志
    log_dir = args.save_path + "/train.log"
    logger = create_logger(log_dir)

    #加载数据集
    train_dataloader=get_dataloader(args)

    #创建模型
    model=total_module(args)
    model.to(device)

    #创建优化器
    optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    #损失函数
    criterion=torch.nn.MSELoss()

    model.train()

    best_loss = float('inf')  # 初始化最佳损失值为正无穷

    #开始训练
    logger.info("Start training...")
    for epoch in range(args.epochs):
        #保存损失值
        epoch_loss = 0.0
        for i,data in enumerate(train_dataloader):

            # 真实值
            targets = data["nino34I"].to(device)

            sstA = data["sstA"].to(device)
            sshA = data["sshA"].to(device)
            #print(sstA.shape)

            optimizer.zero_grad()

            #预测值
            preds=model(sstA,sshA)

            loss=criterion(preds,targets)
            loss.backward()
            optimizer.step()##w=w-lr*grad

            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)

        # 更新最佳损失值并保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = f'{args.save_path}/best_model.pth'
            torch.save(model.state_dict(), best_model_path)

        #打印日志
        logger.info(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}, Best Loss: {best_loss:.4f}')

    #保存最终模型
    save_path = args.save_path + "/final_model.pth"
    torch.save(model.state_dict(), save_path)

    logger.info(f'train model saved to {save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", type=str, required=True, help="path to data")
    parser.add_argument("--save_path","-s", type=str,default='./save', help="path to save model")
    parser.add_argument("--batch_size", "-b",type=int, default=8, help="batch size for training")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="number of epochs")
    parser.add_argument("--conv_channels", "-c", type=int, default=32, help="number of channels")
    parser.add_argument("--features", "-f", type=int, default=32, help="number of features")
    parser.add_argument("--use_rnn", "-r",type=bool, default=False, help="use RNN model")
    parser.add_argument("--num_layers", "-l",type=int, default=2, help="number of layers")
    parser.add_argument("--learning_rate",type=float, default=0.001, help="learning rate")
    parser.add_argument("--mode", "-m", type=str, default="train", help="mode for training")
    args = parser.parse_args()
    train(args)