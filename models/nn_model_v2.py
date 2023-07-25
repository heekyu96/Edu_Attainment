# 2023/5/30 Acc = 92.01% Basic NN model

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from torchsummary import summary

torch.manual_seed(7)

def check_all_attr():
    df = pd.read_csv('dataset/HSLS_2023_short.csv')
    data = df.values
    attr = df.columns[2:]
    n = attr.tolist()
    offset = 5
    print("")
    print("==Data attributes included==")
    for i in range(0, len(n), offset):	# 인덱스 0을 시작으로 n 길이의 전 까지, 10씩
	    print(f'{i+1: 3d} ~{i+offset: 3d} | '," , ".join(["'"+x+"'" for x in n[i:i+offset]]))
    
    
    
def dataload_preprocessing(drops=[]):
    df = pd.read_csv('dataset/HSLS_2023_short.csv')
    data = df.values
    attr = df.columns[2:]
    # NaN data pre-processing
    for i, item in enumerate (data):
         for j, item2 in enumerate(item):
             if math.isnan(item2) : data[i,j] = 0.0

    y = np.asarray(data[:, 0])
    y2 = np.abs(1 - y)
    label = np.vstack([y, y2]).transpose().tolist()

    X = data[:,2:]

    scaled_X = X[:,7:9]
    scaler = StandardScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:,7:9] = scaled_X

    scaled_X = X[:,9:11]
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:,9:11] = scaled_X

    scaled_X = X[:,12].reshape(-1, 1)
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:,12] =  scaled_X.reshape(1, -1)

    scaled_X = X[:,5].reshape(-1, 1)
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:,5] =  scaled_X.reshape(1,-1)

    scaled_X = X[:, 6].reshape(-1, 1)
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:,6] =  scaled_X.reshape(1,-1)

    X_df = pd.DataFrame(X)
    X_df.columns = attr
    y_df = pd.DataFrame(label)
    print("")
    print("Number of data attributes: ",len(attr.tolist()))
    if len(drops)>0:
        print(len(drops)," data attributes are removed.")
        X_df.drop(columns=drops,inplace=True)
    print("Number of data attributes in Traing/Testing process: ",X_df.shape[1])
    
    return X_df, y_df

def dataload_preprocessing_svm(drops=[]):
    df = pd.read_csv('dataset/HSLS_2023_short.csv')
    data = df.values
    attr = df.columns[2:]
    # NaN data pre-processing
    for i, item in enumerate(data):
        for j, item2 in enumerate(item):
            if math.isnan(item2): data[i, j] = 0.0

    y = np.asarray(data[:, 0])
    # y2 = np.abs(1 - y)
    # label = np.vstack([y, y2]).transpose().tolist()

    X = data[:, 2:]

    scaled_X = X[:, 7:9]
    scaler = StandardScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:, 7:9] = scaled_X

    scaled_X = X[:, 9:11]
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:, 9:11] = scaled_X

    scaled_X = X[:, 12].reshape(-1, 1)
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:, 12] = scaled_X.reshape(1, -1)

    scaled_X = X[:, 5].reshape(-1, 1)
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:, 5] = scaled_X.reshape(1, -1)

    scaled_X = X[:, 6].reshape(-1, 1)
    scaler = MinMaxScaler().fit(scaled_X)
    scaled_X = scaler.transform(scaled_X)
    X[:, 6] = scaled_X.reshape(1, -1)

    X_df = pd.DataFrame(X)
    X_df.columns = attr
    y_df = pd.DataFrame(y)
    print("")
    print("Number of data attributes: ", len(attr.tolist()))
    if len(drops) > 0:
        print(len(drops), " data attributes are removed.")
        X_df.drop(columns=drops, inplace=True)
    print("Number of data attributes in Traing/Testing process: ", X_df.shape[1])

    return X_df, y_df

def show_data(X, y):

    y = np.argmax(y, axis=1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    clf = pca.fit_transform(X)

    fig1 = plt.figure(figsize=(8,8))
    ax = fig1.add_subplot(1,1,1, projection='3d')

    ax.scatter(clf[y==0,0], clf[y ==0,1], clf[y ==0,2], c='g', marker='o', label='y=0')
    ax.scatter(clf[y==1,0], clf[y ==1,1], clf[y ==1,2], c='r', marker='o', label='y=1')
    plt.title("Data distribution")
    plt.legend()
    plt.savefig('data_distribution.png')


def plot_decision_regions_2class(X, y):
    y = np.argmax(y.values, axis=1)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    clf = pca.fit_transform(X)

    h = .02
    x_min, x_max = clf[:, 0].min() - 0.1, clf[:, 0].max() + 0.1
    y_min, y_max = clf[:, 1].min() - 0.1, clf[:, 1].max() + 0.1
    # y_min, y_max = y.min() - 0.1 , y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    fig2 = plt.figure(figsize=(8, 8))
    plt.plot(clf[y == 0, 0], clf[y == 0, 1], 'o', label='y=0')
    plt.plot(clf[y == 1, 0], clf[y == 1, 1], 'ro', label='y=1')
    plt.title("decision region")
    plt.legend()
    plt.savefig('decision_region.png')

# def show_model(model, data):
#     #### Summarize and Visualize NN model in graph
#     from torchviz import make_dot
#     make_dot(model(data[0][0]), params=dict(model.named_parameters())).render('model_short', format='png')
#     summary(model, (len(data), input_dim))

class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out, dropout = 0.2):
        super(Net, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(D_in, H)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(H, 64)
        self.linear3 = nn.Linear(64, D_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x)).to(device=self.device)
        x = self.dp1(x).to(device=self.device)
        x = torch.tanh(self.linear2(x)).to(device=self.device)
        x = self.linear3(x).to(device=self.device)

        return x

# Create dataset object
class Edu_Data(Dataset):
    # Constructor
    def __init__(self, X,y):
        self.x = X
        self.y = y
        self.len = X.shape[0]
    # Getter
    def __getitem__(self, index):
        x = self.x.loc[index]
        y = self.y.loc[index]
        return torch.Tensor(x), torch.Tensor(y) #torch.tensor(y, dtype=torch.long)
    # Get Length
    def __len__(self):
        return self.len

# Define the train model
def train(model, criterion, train_loader, optimizer):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
    
    train_loss = 0
    success = 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        pred = model(x)
        loss = criterion(pred, y)
        train_loss += loss
        result = pred.argmax(dim=1, keepdim=True)
        label = y.argmax(dim=1, keepdim=True)
        success += result.eq(label).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    return train_loss/len(train_loader.dataset), success/len(train_loader.dataset)

def validate(model, val_loader):
    model.eval()
    loss = 0
    success =0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
            pred = model(x)
            loss += criterion(pred, y)
            result = pred.argmax(dim=1, keepdim=True)
            label = y.argmax(dim=1, keepdim=True)
            success += result.eq(label).sum().item()

    return loss/len(val_loader.dataset) , success/len(val_loader.dataset)
 

from sklearn.metrics import f1_score  
def test(model, test_loader):
    model.eval()
    loss = 0
    success =0
    total_result = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            pred = model(x)
            loss += criterion(pred, y)
                
            result = pred.argmax(dim=1, keepdim=True)
            label = y.argmax(dim=1, keepdim=True)
            success += result.eq(label).sum().item()
                                  
            
            for r, l in zip(result, label):
                if r == l:
                    total_result.append('pass')
                else:
                    total_result.append('fail')

    return loss/len(test_loader.dataset), success/len(test_loader.dataset), total_result

import torchmetrics
from torchmetrics.classification import BinaryRecall

def test_various_metric(model, test_loader):
    model.eval()
    loss = 0
    success =0
    total_f1 = 0
    total_recall = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            y = y.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            pred = model(x)
            
            result = pred.argmax(dim=1, keepdim=True).squeeze()
            label = y.argmax(dim=1, keepdim=True).squeeze()
            
            f1 = torchmetrics.F1Score(task="binary")
            recall = BinaryRecall()

            total_f1+=f1(result, label)
            total_recall+=recall(result,label)
            # print(recall(result,label))
            # print(f1_)
    return total_f1/len(test_loader), total_recall/len(test_loader)



learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()

best_validation_loss = float('inf')

# def precision(outputs, labels):
#     op = outputs.cpu()
#     la = labels.cpu()
#     _, preds = torch.max(op, dim=1)
#     return torch.tensor(precision_score(la,preds, average=‘weighted’))