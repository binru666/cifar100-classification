from torchvision.datasets import CIFAR100
import torch
import torchvision.transforms as transforms
from torchvision.models import alexnet
from torchsummary import summary
import datetime
import torch.nn as nn
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# ImageNet中，使用mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
# 从零开始训练选择[0.5,0.5,0.5],否则选择上面的
train_dataset = CIFAR100(r'C:\Users\caeit\Desktop\cifar100', download=True, transform=transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
alexnet1 = alexnet()
alexnet1.load_state_dict(torch.load(r"C:\Users\caeit\.cache\torch\checkpoints\alexnet-owt-4df8aa71.tar"))  # 使用预训练好的参数
alexnet1.features[0] = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
alexnet1.classifier[6] = nn.Linear(4096, 100)
alexnet1 = alexnet1.cuda()
summary(alexnet1, (3, 32, 32))
alexnet1 = alexnet1.cuda(1)  # 第一个GPU在训练，换成第二个
cost = torch.nn.CrossEntropyLoss().cuda(1)
optimizer = torch.optim.Adam(alexnet1.parameters(), lr=1e-4)
start = datetime.datetime.now()
for i in range(30):
    correct = 0.0
    running_loss = 0.0
    accuracy = 0.0
    print('-----epoch', i + 1, '-----')
    for num, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.cuda(1), y_train.cuda(1)
        optimizer.zero_grad()
        output = alexnet1(x_train)
        loss = cost(output, y_train)
        _, predicted = torch.max(output, 1)
        correct += (predicted == y_train).sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if num % 500 == 1:
            print('loss:', running_loss, ',accuracy:{}%'.format((100 * correct / ((num + 1) * 16))),
                  'correct numbers:%s' % correct)
    print('---------------------------')
print(datetime.datetime.now() - start)

torch.save(alexnet1, 'cifar100.pkl')
