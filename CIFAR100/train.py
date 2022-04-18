'''
Author: your name
Date: 2021-12-01 15:11:45
LastEditTime: 2022-01-26 15:28:13
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /pytorch-cifar100/train.py
'''
import os
from conf import settings
from utils import get_network, get_cv_generator, get_test_dataloader, get_train_dataloader, get_cv_generator_balance_class, setup_seed
from trainer import Trainer
import nni


def train(args):
    # preprocessing

    setup_seed(2022)

    TRAIN_MEAN = settings.CIFAR100_TRAIN_MEAN
    TRAIN_STD = settings.CIFAR100_TRAIN_STD

    if args.dataset == 'cifar10':
        TRAIN_MEAN = settings.CIFAR10_TRAIN_MEAN
        TRAIN_STD = settings.CIFAR10_TRAIN_STD

    cv_generator = get_cv_generator_balance_class(
        TRAIN_MEAN,
        TRAIN_STD,
        dataset=args.dataset,
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )
    #
    train_loader = get_train_dataloader(
        TRAIN_MEAN,
        TRAIN_STD,
        dataset=args.dataset,
        num_workers=1,
        batch_size=args.b,
        shuffle=True
    )
    #
    test_loader = get_test_dataloader(
        TRAIN_MEAN,
        TRAIN_STD,
        dataset=args.dataset,
        num_workers=1,
        batch_size=args.b,
        shuffle=False
    )

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    models = ['resnext101', 'densenet161', 'wideresnet', 'resnext152', 'densenet201', 'resnet152', 'resnext50', 'densenet121', 'nasnet', 'inceptionv3', 'resnet101', 'xception', 'resnet50', 'seresnet152', 'seresnet101', 'resnet34', 'stochasticdepth101', 'seresnet34', 'seresnet50', 'googlenet', 'seresnet18', 'inceptionv4', 'stochasticdepth34', 'resnet18', 'preactresnet101', 'preactresnet152', 'preactresnet50', 'stochasticdepth50', 'preactresnet18', 'inceptionresnetv2', 'stochasticdepth18', 'vgg13', 'shufflenet', 'vgg16', 'shufflenetv2', 'squeezenet', 'mobilenetv2', 'vgg19', 'vgg11', 'mobilenet', 'attention56']

    for model_name in models:
        args.net = model_name
        model = get_network(args)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{model_name}, para: {pytorch_total_params}')

    trainer = Trainer(model, args)

    y_correct = trainer.cross_validation(cv_generator)

    # trainer.train(train_loader, test_loader)

    model = get_network(args)

    

    trainer = Trainer(model, args)

    trainer.train(train_loader)

    acc, test_id, test_correct = trainer.valid(test_loader)

    # nni.report_final_result(acc)

    trainer.save_result(checkpoint_path, y_correct, test_correct, {'acc': acc})
