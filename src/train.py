import os
import numpy as np
import argparse

from utils.metrics import *
from utils.loss import *
from utils.dataset import *

from net.inception import *

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

def parse_arg():
    model_nms = ["GoogLeNet", "HRGoogLeNet"]
    data_nms = ["PETA", "RAP", "PA100K", "PARSE27K"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name in ' + str(model_nms) + '.')
    parser.add_argument('-s', '--save', type=str, default="",
                        help='The model savename.')###"center/triplet_v1"
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-d', '--data', type=str, default="",
                        help='The dataset need to be trained.')
    parser.add_argument('-w', '--weight', type=str, default="",
                        help='The initial weight.')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The epochs need to be trained')
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='The batch size in the training progress.')
    parser.add_argument('-c', '--classes', type=int, default=0,
                        help='The class number.')
    parser.add_argument('-i', '--iteration', type=int, default=0,
                        help='The iteration number.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    if args.data == "" or args.data not in data_nms:
        raise RuntimeError('NO DATABASE FOUND IN ' + str(data_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------training begining---------------------")
    args = parse_arg()
    model_prefix = "../models/" + args.data + "/" + args.model + "/"
    os.makedirs(model_prefix, exist_ok=True)
    if args.save != "":
        model_prefix = model_prefix + args.save + "_"
    nb_epoch = args.epochs
    nb_class = args.classes
    batch_size = args.batch
    gpus = args.gpus.split(',')
    gpus = [int(x) for x in gpus if x != ""]
    
    ### Dataset & DataLoader
    if args.model == "GoogLeNet":
        indices = None
    else:
        if args.data == "PETA":
            low_level = [27,32,50,56]#, 61, 62, 63, 64
            mid_level = [0,6,7,8,9,11,12,13,17,20,21,22,23,24,25,26,28,29,30,33,35,36,37,38,39,41,42,43,44,45,46,47,48,49,51,52,53,54,55,57,58,59,60]
            high_level = [1,2,3,4,5,10,14,15,16,18,19,31,34,40]
        elif args.data == "RAP":
            low_level = [11]#,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91
            mid_level = [9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42]
            high_level = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]#,51,52,53,54,55,56,57,58,59,60,61,62
        indices = list(np.hstack((low_level, mid_level, high_level)))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = PARDataset(args.data, classes=nb_class, indices=indices, train=True, transforms=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = PARDataset(args.data, classes=nb_class, indices=indices, train=False, transforms=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    ### Network
    #net = models.googlenet(pretrained=False, progress=True, num_classes=nb_class, aux_logits=False)
    if args.model == "GoogLeNet":
        arguments = nb_class
    else:
        arguments = [len(low_level), len(mid_level), len(high_level)]
    net = globals()[args.model](classes = arguments)
    if args.weight != "":
        net.load_state_dict(torch.load(args.weight))
    if len(gpus) >= 1:
        device = torch.device("cuda:{}".format(gpus[0]) if torch.cuda.is_available() else "cpu")
        """
        os.environment["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net, device_ids=gpus)
        net.to(device=gpus[0])
    print(net)
    
    ### Train
    #sigmoid = nn.Sigmoid()
    criterion = nn.BCELoss()
    alpha = np.sum(train_dataset.labels, axis=0)
    alpha = alpha / len(train_dataset.labels)
    #criterion = WeightedBCEloss(alpha)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)
    net.train()
    min_loss = float('inf')
    for epoch in range(nb_epoch):
        for i, (inputs, labels) in enumerate(train_dataloader):
            if len(gpus) >= 1:
                inputs = inputs.to(device=gpus[0])## 训练数据放在主设备
                labels = labels.to(device=gpus[0])
                labels = labels.float()

            optimizer.zero_grad()
            outputs = net(inputs)
            predictions = outputs
            #print(predictions)
            #print(labels)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            """
            y_pred = torch.max(outputs, 1)[1]
            acc = float(sum(np.array((y_pred == labels).cpu().numpy()))) / len(labels)
            print('epc:%d,step:%d,loss:%f,acc:%f' % (epoch,i,loss,acc))
            """
            predictions_np = np.sign(predictions.cuda().data.cpu().numpy() - 0.5)
            labels_np = np.sign(labels.cuda().data.cpu().numpy() - 0.5)
            result = calculate_accuracy(labels_np, predictions_np)
            print('epc:%d/step:%d : loss:%f, mA:%f, acc:%f, prec:%f, rec:%f, f1:%f' % 
                  (epoch + 1, i, loss, result['label_ma'], result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1']))

        test_loss = 0.0
        predictions_np = []
        labels_np = []
        for inputs, labels in test_dataloader:
            if len(gpus) >= 1:
                inputs = inputs.to(device=gpus[0])## 训练数据放在主设备
                labels = labels.to(device=gpus[0])
                labels = labels.float()
                
            outputs = net(inputs)
            predictions = outputs
            test_loss += criterion(predictions, labels).item()
            if len(predictions_np) == 0:
                predictions_np = np.sign(predictions.cuda().data.cpu().numpy() - 0.5)
                labels_np = np.sign(labels.cuda().data.cpu().numpy() - 0.5)
            else:
                predictions_np = np.vstack((predictions_np, np.sign(predictions.cuda().data.cpu().numpy() - 0.5)))
                labels_np = np.vstack((labels_np, np.sign(labels.cuda().data.cpu().numpy() - 0.5)))
        test_loss /= len(test_dataloader.dataset)
        print(predictions_np.shape)
        result = calculate_accuracy(labels_np, predictions_np)
        print('VAL set: Average loss: {:.4f}, mA: {:.4f}, acc: {:.4f}, prec: {:.4f}, rec: {:.4f}, f1: {:.4f}'.format(test_loss, result['label_ma'], result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1']))

        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(net.state_dict(), model_prefix + 'epoch%03d_valloss%.6f.pkl' % (epoch + 1, min_loss))
            print("Save Model in: " + model_prefix + 'epoch%03d_valloss%.6f.pkl' % (epoch + 1, min_loss))
