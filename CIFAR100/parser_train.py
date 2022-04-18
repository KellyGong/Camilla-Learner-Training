'''
Author: your name
Date: 2021-12-01 15:10:04
LastEditTime: 2021-12-15 21:12:04
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /pytorch-cifar100/parser_train.py
'''
import argparse
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="vgg16", help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-early_stop', type=int, default=15, help='epoch of early stop')
    parser.add_argument('--dataset', type=str, choices=['cifar100'], default='cifar100')
    args = parser.parse_args()

    print(f"***** Model: {args.net} *****")

    print(args)

    train(args)
