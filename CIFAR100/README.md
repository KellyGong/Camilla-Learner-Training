# CIFAR-100 Learners

## the backbone code of learners is adapted from https://github.com/weiaicunzai/pytorch-cifar100

## Requirements

This is my experiment eviroument
- python3.7
- pytorch1.7.1+cu110


## Usage

### 1. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
$ python parser_train.py -net vgg16 -gpu
```

The supported net args are:
```
squeezenet
mobilenet
mobilenetv2
shufflenet
shufflenetv2
vgg11
vgg13
vgg16
vgg19
densenet121
densenet161
densenet201
googlenet
inceptionv3
inceptionv4
inceptionresnetv2
xception
resnet18
resnet34
resnet50
resnet101
resnet152
preactresnet18
preactresnet34
preactresnet50
preactresnet101
preactresnet152
resnext50
resnext101
resnext152
attention56
attention92
seresnet18
seresnet34
seresnet50
seresnet101
seresnet152
nasnet
wideresnet
stochasticdepth18
stochasticdepth34
stochasticdepth50
stochasticdepth101
```



### 2. obtain the response logs from learners to samples
In train_cifar100.csv, we obtain the response logs by 5-fold class_balanced cross validations for learners in the training set of CIFAR-100.

In test_cifar100.csv, we obtain the response logs by training learners on the training set of CIFAR-100 and testing them on the test set of CIFAR-100.




