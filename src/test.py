import os
import numpy as np
import glob
import argparse

from utils.metrics import *
from utils.loss import *
from utils.dataset import *

from net.inception import *

def parse_arg():
    model_nms = ["Inception", "HRInception"]
    data_nms = ["PETA", "RAP", "PA100K"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name in ' + str(model_nms) + '.')
    parser.add_argument('-s', '--save', type=str, default="",
                        help='The save name.')
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-d', '--data', type=str, default="",
                        help='The dataset need to be trained.')
    parser.add_argument('-w', '--weights', type=str, default="",
                        help='The weight file need to be loaded.')
    parser.add_argument('-c', '--classes', type=int, default=0,
                        help='The class number.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    if args.data == "" or args.data not in data_nms:
        raise RuntimeError('NO DATABASE FOUND IN ' + str(data_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------testing begining---------------------")
    args = parse_arg()
    model_prefix = "../models/" + args.data + "/" + args.model + "/"
    result_prefix = "../results/" + args.data + "/" + args.model + "/"
    os.makedirs(result_prefix, exist_ok=True)
    nb_class = args.classes
    save_name = args.save
    gpus = args.gpus.split(',')
    gpus = [int(x) for x in gpus if x != ""]
    
    ### Dataset & DataLoader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = PARDataset(args.data, train=False, transform=transform_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False)
    
    ### Network
    
    
    ### Test
    sigmoid = nn.Sigmoid()
    predictions_np = []
    labels_np = []
    for inputs, labels in test_dataloader:
        if len(gpus) >= 1:
            inputs = inputs.to(device=gpus[0])## 训练数据放在主设备
            labels = labels.to(device=gpus[0])
            labels = labels.float()
            
        outputs = net(inputs)
        predictions = sigmoid(outputs)
        test_loss += criterion(predictions, labels).item()
        if len(predictions_np) == 0:
            predictions_np = np.sign(predictions.cuda().data.cpu().numpy() - 0.5)
            labels_np = np.sign(labels.cuda().data.cpu().numpy() - 0.5)
        else:
            predictions_np = np.vstack((predictions_np, np.sign(predictions.cuda().data.cpu().numpy() - 0.5)))
            labels_np = np.vstack((labels_np, np.sign(labels.cuda().data.cpu().numpy() - 0.5)))
                
    np.save(predictions_np, result_prefix + save_name + "_" + w + "_predictions.npy")
    results = calculate_accuracy(labels_np, predictions_np)
    with open(result_prefix + save_name + "_" + w + "_results.file", "wb") as f:
        pickle.dump(results, f)
    print(result_prefix + save_name + "_" + w + '_results.file')