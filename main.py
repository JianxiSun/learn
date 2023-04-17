import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

captcha_list = list("0123456789abcdefghijklmnopqrstuvwxyz_")
captcha_length = 6


def text2vec(text):
    vector = torch.zeros((captcha_length, len(captcha_list)))
    text_len = len(text)
    if text_len > captcha_length:
        raise ValueError("验证码超过6位")
    for i in range(text_len):
        vector[i, captcha_list.index(text[i])] = 1
    return vector


def vec2text(vec):
    label = torch.nn.functional.softmax(vec, dim=1)
    vec = torch.argmax(label, dim=1)
    for v in vec:
        text_list = [captcha_list[v] for v in vec]
    return ''.join(text_list)


def make_dataset(data_path):
    img_names = os.listdir(data_path)
    samples = []
    for img_name in img_names:
        img_path = data_path + img_name
        target_str = img_name.split('_')[0].lower()
        samples.append((img_path, target_str))
    return samples


class CaptchaData(Dataset):
    def __init__(self, data_path, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.samples = make_dataset(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target = self.samples[index]
        target = text2vec(target)
        target = target.view(1, -1)[0]
        img = Image.open(img_path)
        img = img.resize((140, 44))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(4*16*128, 1024),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 6*37)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


net = Net()


def calculate_acc(output, target):
    output, target = output.view(-1, len(captcha_list)), target.view(-1, len(captcha_list))
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    output, target = output.view(-1, captcha_length), target.view(-1, captcha_length)
    c = 0
    for i, j in zip(target, output):
        if torch.equal(i, j):
            c += 1
    acc = c/output.size()[0] * 100
    return acc


def train(epoch_nums):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CaptchaData("dataset/", transform=transform)
    train_data_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True, drop_last=True)

    test_data = CaptchaData('dataset/', transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=128, num_workers=0, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备是:", device)
    net.to(device)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    i = 1
    for epoch in range(epoch_nums):
        running_loss = 0.0
        net.train()
        for data in train_data_loader:
            inputs, labels = data
            device = torch.device('cuda:0')
            inputs = inputs.to(device)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                acc = calculate_acc(outputs, labels)
                print('第%s次训练正确率: %.3f %%, loss: %.3f' % (i, acc, running_loss / 2000))
                running_loss = 0

            i += 1

        net.eval()
        with torch.no_grad():
            for inputs, labels in test_data_loader:
                outputs = net(inputs)
                acc = calculate_acc(outputs, labels)
                print('测试集正确率: %.3f %%' % (acc))
                break
        if epoch % 5 == 4:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9


train(10)



