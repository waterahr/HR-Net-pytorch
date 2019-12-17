import os
import numpy as np
import pandas as pd
import random
import scipy.io as sio
import h5py
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

"""
data_transform={'train':transforms.Compose([
                    # transforms.RandomResizedCrop(image_size),
                    # transforms.Resize(224),
                    transforms.RandomResizedCrop(int(image_size*1.2)),
                    # transforms.ToPILImage(),
                    transforms.RandomAffine(15),
                    # transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.RandomGrayscale(),
                    transforms.TenCrop(image_size),
                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
                   
                    # transforms.FiveCop(image_size),
                    # Lambda(lambda crops: torch.stack([transfoms.ToTensor()(crop) for crop in crops])),
                    # transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]),
                "val":transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                "test":transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])}
                """

class PARDataset(Dataset):
    def __init__(self, dataset, classes, partion=0, indices=None, image_size=224, train=True, transforms=None, target_transforms=None):
        self.train = train
        self.image_size = image_size
        self.transforms = transforms
        self.target_transforms = target_transforms
        if dataset == "PETA":
            data_root = "/home/anhaoran/codes/pedestrian_attibutes_wpal/results/"
            data = np.array(pd.read_csv(data_root + "PETA.csv"))[:, 1:]
    
            labels_list_data = open("/home/anhaoran/data/pedestrian_attributes_PETA/PETA/labels.txt")
            lines = labels_list_data.readlines()
            attributes_list = []
            for line in lines:
                tmp = line.split()
                attributes_list.append(tmp[1])
            self.cls = attributes_list

            X_data = []
            y_data = []
            for i in range(len(data)):
                X_data.append(data[i, 0])
                y_data.append(data[i, 1:])
            X_data = np.asarray(X_data)
            y_data = np.asarray(y_data)
            if train:
                self.data, self.labels = X_data[:11400], y_data[:11400, :classes]
            else:
                self.data, self.labels = X_data[11400:], y_data[11400:, :classes]
        elif dataset == "RAP":
            data_root = "/home/anhaoran/data/pedestrian_attributes_RAP/"
            data = sio.loadmat(data_root + "RAP_annotation/RAP_annotation.mat")["RAP_annotation"]

            attributes_list = []
            for i in range(data["attribute_eng"][0][0].shape[0]):
                attributes_list.append(data["attribute_eng"][0][0][i][0][0])
            self.cls = attributes_list

            X_data = []
            y_data = []
            for i in range(41585):
                X_data.append(os.path.join(data_root + "RAP_dataset", data['imagesname'][0][0][i][0][0]))
                y_data.append(data['label'][0][0][i])
            X_data = np.asarray(X_data)
            y_data = np.asarray(y_data)

            train_indices, test_indices = data['partion'][0][0][partion][0][0][0]
            train_indices, test_indices = list(train_indices[0] - 1), list(test_indices[0] - 1)
            
            if train:
                self.data, self.labels = X_data[train_indices], y_data[train_indices, :classes]
            else:
                self.data, self.labels = X_data[test_indices], y_data[test_indices, :classes]
        elif dataset == "PA100K":
            data_root = "/home/anhaoran/data/pedestrian_attributes_PA-100K/"
            data = sio.loadmat(data_root + "annotation/annotation.mat")
    
            attributes_list = []
            for i in range(len(data["attributes"])):
                attributes_list.append(data["attributes"][i][0][0])
            self.cls = attributes_list

            X_train = []
            y_train = data['train_label']
            for i in range(len(data['train_images_name'])):
                X_train.append(data_root + "data/release_data/release_data/" + str(data['train_images_name'][i][0][0]))
            X_train = np.asarray(X_train)
            X_val = []
            y_val = data['val_label']
            for i in range(len(data['val_images_name'])):
                X_val.append(data_root + "data/release_data/release_data/" + str(data['val_images_name'][i][0][0]))
            X_val = np.asarray(X_val)
            X_test = []
            y_test = data['test_label']
            for i in range(len(data['test_images_name'])):
                X_test.append(data_root + "data/release_data/release_data/" + str(data['test_images_name'][i][0][0]))
            X_test = np.asarray(X_test)
            
            if train:
                self.data, self.labels = X_train, y_train[:, :classes]
            else:
                self.data, self.labels = X_val, y_val[:, :classes]
        elif dataset == "PARSE27K":
            data_root = "/home/anhaoran/data/pedestrian_attributes_parse-27k/"
            data = h5py.File(data_root + "train.hdf5", 'r')
            X_train = data["crops"][:].transpose((0,2,3,1)).shape
            y_train = data['labels'][:]
            data = h5py.File(data_root + "val.hdf5", 'r')
            X_val = data["crops"][:].transpose((0,2,3,1)).shape
            y_val = data['labels'][:]
            data = h5py.File(data_root + "test.hdf5", 'r')
            X_test = data["crops"][:].transpose((0,2,3,1)).shape
            y_test = data['labels'][:]
            
            if train:
                self.data, self.labels = X_train, y_train[:, :classes]
            else:
                self.data, self.labels = X_val, y_val[:, :classes]
        else:
            raise RuntimeError('NO DATASET FOUND!')
        
        if indices is not None:
            self.labels = self.labels[:, indices]
        print("=========================================")
        print(self.cls)
        print("The shape of the data is: ", self.data.shape)
        print("The shape of the label is: ", self.labels.shape)
        print("=========================================")       
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        img = Image.open(img)
        img = img.resize((self.image_size, self.image_size))#, Image.ANTIALIAS
        if self.transforms is not None:
            img = self.transforms(img)
        if self.target_transforms is not None:
            target = self.target_transforms(target)
        return img, target
    
    def show(self):
        for i in range(15):
            plt.subplot(3, 5, i + 1)
            idx = random.randint(0, len(self.data) - 1)
            img, label = Image.open(self.data[idx]), self.labels[idx]
            print(label)
            plt.axis('off')
            plt.imshow(img)
            attributes = []
            for j in len(label):
                if label[j] == 1:
                    attributes.append(self.cls[j])
            plt.title(str(attributes))
        plt.show()
        

if __name__ == "__main__":
    name = "RAP"
    train_dataset = PARDataset(name, train = True)
    test_dataset = PARDataset(name, train = False)
    print("OK!")