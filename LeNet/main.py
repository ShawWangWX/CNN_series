#!env
# Author
# Create Time: 2022/7/4
# License: None
"""
Info
"""
from tqdm import tqdm
from datasets import *
from model import *

import torch
import torch.nn.functional as F
from torch import optim


if __name__ == '__main__':
    train_data_loader, test_data_loader = data_loader(path='./data', batch_size=64)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    device = torch.device("cpu")

    net = LeNet(6, 16)
    net = net.to(device)

    opt = optim.Adam(net.parameters(), lr=0.001)
    # opt = opt.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    epochs = 10
    for _ in range(epochs):
        # train
        net.train()
        index = 0
        for data, label in tqdm(train_data_loader):
            data = data.to(device)
            label = label.to(device)

            index += 1
            output = net(data)
            loss = criterion(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # test
        net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, label in test_data_loader:
                data = data.to(device)
                label = label.to(device)
                output = net(data)
                _, pred = torch.max(output.data, 1)
                total += label.size(0)
                correct += (pred == label).sum().item()
            print("accuracy:{:.4f}%".format(correct/total * 100))






