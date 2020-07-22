from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
test_dataset = CIFAR100(r'C:\Users\caeit\Desktop\cifar100', download=True, transform=transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
alexnet1 = torch.load('cifar100.pkl')
alexnet1.eval()
correct = 0 
for num, (x_train, y_train) in enumerate(test_loader):
    x_train,y_train = x_train.cuda(1),y_train.cuda(1)
    output = alexnet1(x_train)
    _, predicted = torch.max(output, 1)
    correct += (predicted == y_train).sum().item()
print('accuracy:{}%'.format((100*correct/((num+1)*16))),'correct numbers:%s'%correct,'total test number:',
                          (num+1)*16 )
print('-------------------------------------------------------------')
