import torch
import torchvision
import torchvision.transforms as transfroms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device('cuda:0')

transfrom = transfroms.Compose([transfroms.ToTensor(),transfroms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transfrom)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transfrom)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle = True,num_workers=0)

testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle = True,num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,img):
        img = F.relu(self.conv1(img))
        img = self.pool(img)
        img = F.relu(self.conv2(img))
        img = self.pool(img)
        img = img.view(-1,16*5*5)
        img = F.relu(self.fc1(img))
        img = F.relu(self.fc2(img))
        img = self.fc3(img)
        return img
net = Net().to(device)
'''


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, img):
        img = self.features(img)
        img = img.view(img.size(0), 256 * 2 * 2)
        img = self.classifier(img)
        return img

net = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

start = time.time()

for epoch in range(10):
    running_loss =0
    for i,data in enumerate(trainloader,0):
        images,labels = data
        images,labels = Variable(images).to(device),Variable(labels).to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 0:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i , running_loss / 1000))
            running_loss =0
print('Finished!')

correct = 0
total = 0
for data in testloader:
    images,labels = data
    images  = Variable(images).to(device)
    outputs = net(images)
    _,predict = torch.max(outputs.data,1)
    correct += (predict.cpu()==labels.cpu()).sum().item()
    total += labels.size(0)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data in testloader:
    images,labels = data
    images  = Variable(images).to(device)
    outputs = net(images)
    _,predicted= torch.max(outputs,1)
    predict = (predicted.cpu()==labels.cpu()).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += predict[i]
        class_total[label] +=1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

end = time.time()
print(end - start)











