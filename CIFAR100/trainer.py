import numpy as np
from conf import settings
import torch
import torch.nn as nn
import torch.optim as optim
from utils import WarmUpLR, EarlyStop
import json
from utils import get_network
import nni


class Trainer:
    def __init__(self, net, args):
        self.args = args
        self.net = net
    
    def init_net_parameter(self):
        for name, param in self.net.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def train(self, dataloader, valid_loader=None):
        # self.init_net_parameter()
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
                                                         gamma=0.2)  # learning rate decay
        iter_per_epoch = len(dataloader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * self.args.warm)

        # early_stop = EarlyStop(self.args.early_stop)

        # train
        for epoch in range(1, settings.EPOCH + 1):
            self.net.train()
            if epoch > self.args.warm:
                train_scheduler.step()
            for batch_index, (_, images, labels) in enumerate(dataloader):

                if self.args.gpu:
                    labels = labels.cuda()
                    images = images.cuda()

                optimizer.zero_grad()
                outputs = self.net(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                #     loss.item(),
                #     optimizer.param_groups[0]['lr'],
                #     epoch=epoch,
                #     trained_samples=batch_index * self.args.b + len(images),
                #     total_samples=len(dataloader.dataset)
                # ))

                if epoch <= self.args.warm:
                    warmup_scheduler.step()

            if valid_loader:
                acc, data_ids, data_correct = self.valid(valid_loader)
                # nni.report_intermediate_result(acc)
                print(f'Valid Epoch: {epoch}, accuracy: {round(acc, 5)}')
                # if early_stop(acc):
                #     break

        if valid_loader:
            return data_ids, data_correct
        else:
            return None

    def valid(self, valid_loader):
        self.net.eval()
        correct = 0.0
        data_id = []
        data_correct = []
        for (data_ids, images, labels) in valid_loader:
            if self.args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = self.net(images)
            _, preds = outputs.max(1)

            correct_tensor = preds.eq(labels)
            correct += correct_tensor.sum()
            data_id.extend(data_ids.numpy().tolist())
            data_correct.extend(correct_tensor.cpu().int().numpy().tolist())
        return float(correct.cpu()) / len(data_id), data_id, data_correct

    def cross_validation(self, cv_generator):
        data_ids, data_correct = [], []
        for train_loader, valid_loader in cv_generator:
            self.net = get_network(self.args)
            fold_data_ids, fold_data_correct = self.train(train_loader, valid_loader)
            data_ids.extend(fold_data_ids)
            data_correct.extend(fold_data_correct)

        id_correct_tuple = list(zip(data_ids, data_correct))
        id_correct_tuple = sorted(id_correct_tuple, key=lambda x: x[0])
        correct_list = [id_correct[1] for id_correct in id_correct_tuple]
        return np.array(correct_list, dtype=np.int)

    def save_result(self, path, y_correct, test_correct, result_dict):
        # save cv result and test set result
        cv_path = path + '/y_correct.npy'
        result_path = path + '/result.json'
        test_correct_path = path + '/test_correct.npy'
        weight_path = path + '/model.pt'

        with open(result_path, 'w', encoding='utf8') as f:
            json.dump(result_dict, f, indent=4)

        torch.save(self.net.state_dict(), weight_path)
        np.save(cv_path, y_correct)
        np.save(test_correct_path, test_correct)
        nni.report_final_result(sum(y_correct) / y_correct.size)
