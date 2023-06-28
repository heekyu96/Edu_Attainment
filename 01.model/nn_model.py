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
# from torchsummary import summary

torch.manual_seed(7)

def dataload_preprocessing(filename):
    df = pd.read_csv(filename)
    data = df.values

    # NaN data pre-processing
    for i, item in enumerate (data):
         for j, item2 in enumerate(item):
             if math.isnan(item2) : data[i,j] = 0.0

    y = np.asarray(data[:,0])
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

    return X, y

def show_data(dataset):
    X = dataset[:][0].numpy()
    y = dataset[:][1].numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    clf = pca.fit_transform(X)

    fig1 = plt.figure(figsize=(8,8))
    ax = fig1.add_subplot(1,1,1, projection='3d')

    ax.scatter(clf[y ==0,0], clf[y ==0,1], clf[y ==0,2], c='g', marker='o', label='y=0')
    ax.scatter(clf[y ==1,0], clf[y ==1,1], clf[y ==1,2], c='r', marker='o', label='y=1')
    plt.title("Data distribution")
    plt.legend()
    plt.show()


def plot_decision_regions_2class(val_dataset):
    x = val_dataset[:][0].numpy()
    y = val_dataset[:][1].numpy()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    clf = pca.fit_transform(x)

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
    plt.show()

def show_model(model, data):
    #### Summarize and Visualize NN model in graph
    from torchviz import make_dot
    make_dot(model(data[0][0]), params=dict(model.named_parameters())).render('model_short', format='png')
    summary(model, (64, input_dim))

class Net(nn.Module):
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(D_in, H)
        self.dp1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(H, 64)
        self.linear3 = nn.Linear(64, D_out)

    # Prediction
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dp1(x)
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x

# Create dataset object
class myData(Dataset):
    # Constructor
    def __init__(self, N_s=100):
        self.x = X
        self.y = y
        self.len = X.shape[0]
    # Getter
    def __getitem__(self, index):
        x = self.x[:][index]
        y = self.y[index]
        return torch.Tensor(x), torch.tensor(y, dtype=torch.long)
    # Get Length
    def __len__(self):
        return self.len

# Define the train model
def train(model, criterion, train_loader, optimizer):
    train_loss = 0
    success =0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        pred = model(x)
        loss = criterion(pred, y)
        result = pred.argmax(dim=1, keepdim=True)
        train_loss += loss
        success += result.eq(y.view_as(result)).sum().item()
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
            pred = model(x)
            loss += criterion(pred, y)
            result = pred.argmax(dim=1, keepdim=True)
            success += result.eq(y.view_as(result)).sum().item()
    return loss/len(val_loader.dataset) , success/len(val_loader.dataset)

def test(model, test_loader):
    model.eval()
    loss = 0
    success =0
    total_result = []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            loss += criterion(pred, y)
            result = pred.argmax(dim=1, keepdim=True)
            success += result.eq(y.view_as(result)).sum().item()

            for i, item in enumerate(result):
                if item == y[i]: total_result.append('pass')
                else : total_result.append('fail')
    return loss/len(test_loader.dataset) , success/len(test_loader.dataset), total_result

#######  Main program

input_dim = 66
hidden_dim = 4068
output_dim = 2
model = Net(input_dim, hidden_dim, output_dim)

learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

X, y = dataload_preprocessing(filename='HSLS_2023_short.csv')
data_set = myData()

train_size = int(len(X) * 0.8)
val_size = int(train_size * 0.1)
test_size = len(X) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(data_set, lengths=[train_size, val_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

epochs =10
best_validation_loss = float('inf')

for epoch in range (epochs):
    time_start = time.time()

    train_loss, train_accuracy = train(model, criterion, train_loader, optimizer)
    validation_loss, validation_accuracy = validate(model, val_loader)

    time_end = time.time()
    time_delta = time_end - time_start

    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model, 'nnmodel.pt')

    print(f'epoch number: {epoch + 1} | time elapsed: {time_delta}s')
    print(f'training loss: {train_loss:.3f} |  training accuracy: {train_accuracy * 100:.2f}%')
    print(f'validation loss: {validation_loss:.3f} |  validation accuracy: {validation_accuracy * 100:.2f}%')
    print()

best_model = torch.load('nnmodel.pt')
test_loss, test_accuracy, result = test(best_model, test_loader)
print(f'Test loss: {test_loss:.3f} | test: {test_accuracy * 100:.2f}%')

show_model(model, data_set)
show_data(data_set)
plot_decision_regions_2class(test_dataset)


