# -*- coding: utf-8 -*-

import torch

import torchvision

import torchvision.transforms as transforms

import math

import methods

import time

import argparse

########################################################################

# The output of torchvision datasets are PILImage images of range [0, 1].

# We transform them to Tensors of normalized range [-1, 1]

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--always', type=bool, default=False, metavar='N',
                    help='always do rebalance after each update (default: True)')
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr2', type=float, default=0.2, metavar='LR2',
                    help='learning rate2 (default: 0.2)')
parser.add_argument('--momentum', type=float, default=0, metavar='M',
                    help='SGD momentum (default: 0)')
parser.add_argument('--momentum2', type=float, default=0, metavar='M2',
                    help='SGD momentum2 (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.seed = int(time.time()*1000)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


transform = transforms.Compose(

    [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True,

                                        download=True, transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,

                                          shuffle=True, num_workers=8)



testset = torchvision.datasets.CIFAR10(root='./data', train=False,

                                       download=True, transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,

                                         shuffle=False, num_workers=8)



classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



########################################################################

# Let us show some of the training images, for fun.



# import matplotlib.pyplot as plt

import numpy as np




from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)

        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)

        self.fc2 = nn.Linear(120, 84, bias=False)

        self.fc3 = nn.Linear(84, 10, bias=False)
        self.bn1 = nn.BatchNorm2d(6, affine=False)

        self.bn2 = nn.BatchNorm2d(16, affine=False)
        self.bnfc1 = nn.BatchNorm1d(120, affine=False)
        self.bnfc2 = nn.BatchNorm1d(84, affine=False)
  

    def forward(self, x):
        x = self.conv1(x)
        # self.conv1out.retain_grad()
        x = self.pool(F.relu(self.bn1(x)))
        # x = self.pool(F.relu(x))
        
        x = self.conv2(x)
        # self.conv2out.retain_grad()
        x = self.pool(F.relu(self.bn2(x)))
        # x = self.pool(F.relu(x))

        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        # self.fc1out.retain_grad()
        x = F.relu(self.bnfc1(x))
        # x = F.relu(x)
        
        x = self.fc2(x)
        # self.fc2out.retain_grad()
        x = F.relu(self.bnfc2(x))
        # x = F.relu(x)

        x = self.fc3(x)
        # self.fc3out.retain_grad()

        return x



def initialize_net(net_to_init, method='xavier_normal'):
    init_stdv = 0.001
    total_norm = 0
    i=0
    init_method_dict = {'xavier_normal':torch.nn.init.xavier_normal, 'orthogonal':torch.nn.init.orthogonal, 'kaiming_normal':torch.nn.init.kaiming_normal}
    for p in net_to_init.parameters():
        if p.data.ndimension() < 2:
            p.data = torch.zeros(p.data.size())
        else:
            init_method_dict[method](p.data)
        total_norm += p.data.norm() ** 2
    print("initialization succeed, total norm is", math.sqrt(total_norm))

net = Net()
net2 = Net()

if torch.cuda.is_available():
    net.cuda()
    net2.cuda()


########################################################################

# 3. Define a Loss function and optimizer

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Let's use a Classification Cross-Entropy loss and SGD with momentum


import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)#, weight_decay=1e-4)

optimizer2 = optim.SGD(net2.parameters(), lr=args.lr2, momentum=args.momentum2)#, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)
scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.2)
print("optimizer = optim.SGD(net.parameters(), lr=", args.lr," momentum=", args.momentum,") \n optimizer2 = optim.SGD(net2.parameters(), lr=",args.lr2," momentum=",args.momentum2,")")


########################################################################
def linear_grad(x, dout):
    return dout.unsqueeze(2).bmm(x.unsqueeze(1)).mean(dim=0)

# def reparametrize_fc_grad(self, grad_input, grad_output):
#     if grad_output[0] is not None:
#         grad_input[0].data = (grad_output[0].data / (self.weight.data.norm(p=2, dim=1)**2).unsqueeze(0)).mm(self.weight.data) 
#         # self.weight.grad = linear_grad(net.fc3in, grad_output[0]) / math.sqrt(self.weight.size(1))# not good
# net.fc3.register_backward_hook(reparametrize_fc_grad)
# net.fc2.register_backward_hook(reparametrize_fc_grad)
# net.fc1.register_backward_hook(reparametrize_fc_grad)

def reparametrize_fc_grad(self, grad_input, grad_output):
    if grad_output[0] is not None:
        grad_input[0].data = (grad_output[0].data).mm(self.weight.data) / (self.weight.data.norm(p=2, dim=0)**2).unsqueeze(0)
        # self.weight.grad = linear_grad(net.fc3in, grad_output[0]) / math.sqrt(self.weight.size(1))# not good
# net.fc3.register_backward_hook(reparametrize_fc_grad)
# net.fc2.register_backward_hook(reparametrize_fc_grad)
# net.fc1.register_backward_hook(reparametrize_fc_grad)
# 4. Train the network

def OurBlockDiagonal(net_):

    modulus = 1
    net_.fc3.weight.grad.data /= modulus
    modulus *= (net_.fc3.weight.data.norm(p=2, dim=0) **2).mean() 
    net_.fc2.weight.grad.data /= modulus / (net_.fc2.weight.data.norm(p=2, dim=1) **2).mean()
    modulus *= (net_.fc2.weight.data.norm(p=2, dim=0) **2).mean() / (net_.fc2.weight.data.norm(p=2, dim=1) **2).mean() 
    net_.fc1.weight.grad.data /= modulus / (net_.fc1.weight.data.norm(p=2, dim=1) **2).mean()
    modulus *= (net_.fc1.weight.data.norm(p=2, dim=0) **2).mean() / (net_.fc1.weight.data.norm(p=2, dim=1) **2).mean() / 2
    net_.conv2.weight.grad.data /= modulus / (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean() * 25 #/ 2 * 25
    modulus *= (net_.conv2.weight.data.view(net_.conv2.weight.data.size(1), -1).norm(p=2, dim=1) **2).mean() / 2 / (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean()
    net_.conv1.weight.grad.data /= modulus / (net_.conv1.weight.data.view(net_.conv1.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean() * 196 #/ 2 * 196

def OurBlockDiagonal4(net_):

    modulus = 1
    net_.fc3.weight.grad.data /= modulus
    modulus *= (net_.fc3.weight.data.norm(p=2, dim=0) **2).mean() 
    net_.fc2.weight.grad.data /= modulus / (net_.fc2.weight.data.norm(p=2, dim=1) **2).mean()
    # print("layer 3 rescaling coefficient", modulus /(net_.fc2.weight.data.norm(p=2, dim=1) **2).mean())

    modulus *= (net_.fc2.weight.data.norm(p=2, dim=0) **2).mean() / (net_.fc2.weight.data.norm(p=2, dim=1) **2).mean() 
    net_.fc1.weight.grad.data /= modulus / (net_.fc1.weight.data.norm(p=2, dim=1) **2).mean()
    # print("layer 2 rescaling coefficient", modulus /(net_.fc1.weight.data.norm(p=2, dim=1) **2).mean())
    modulus *= (net_.fc1.weight.data.norm(p=2, dim=0) **2).mean() / (net_.fc1.weight.data.norm(p=2, dim=1) **2).mean() / 4
    net_.conv2.weight.grad.data /= modulus / (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean() * 25 #/ 2 * 25
    # print("layer 1 rescaling coefficient", modulus /(net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean() * 25)
    modulus *= (net_.conv2.weight.data.view(net_.conv2.weight.data.size(1), -1).norm(p=2, dim=1) **2).mean() / 4 / (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean()
    net_.conv1.weight.grad.data /= modulus / (net_.conv1.weight.data.view(net_.conv1.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean() * 196 #/ 2 * 196
    # print("layer 0 rescaling coefficient", modulus /(net_.conv1.weight.data.view(net_.conv1.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean() * 196)


def OurBlockDiagonal2(net_):

    modulus = 1
    net_.fc3.weight.grad.data /= modulus
    # modulus *= (net_.fc3.weight.data.norm(p=2, dim=0) **2).mean() 
    # net_.fc2.weight.grad.data /= modulus / (net_.fc2.weight.data.norm(p=2, dim=1) **2).mean()
    # modulus *= (net_.fc2.weight.data.norm(p=2, dim=0) **2).mean() / (net_.fc2.weight.data.norm(p=2, dim=1) **2).mean() 
    # net_.fc1.weight.grad.data /= modulus / (net_.fc1.weight.data.norm(p=2, dim=1) **2).mean()
    # modulus *= (net_.fc1.weight.data.norm(p=2, dim=0) **2).mean() / (net_.fc1.weight.data.norm(p=2, dim=1) **2).mean() / 2
    net_.conv2.weight.grad.data *= (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1)).mean() #/ 2 * 25
    modulus *= (net_.conv2.weight.data.view(net_.conv2.weight.data.size(1), -1).norm(p=2, dim=1) **2).mean() / 2 / (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1) **2).mean()
    net_.conv1.weight.grad.data *= (net_.conv1.weight.data.view(net_.conv1.weight.data.size(0), -1).norm(p=2, dim=1)).mean() #* 196 #/ 2 * 196

def OurBlockDiagonal3(net_):

    modulus = 1
    net_.fc3.weight.grad.data /= modulus
    modulus *= (net_.fc3.weight.data.norm(p=2, dim=1)**2).mean() / 2
    net_.fc2.weight.grad.data /= modulus # / (net_.fc2.weight.data.norm(p=2, dim=1)).mean()
    modulus *= 1 #(net_.fc2.weight.data.norm(p=2, dim=1)).mean() / 2 / (net_.fc2.weight.data.norm(p=2, dim=1)).mean() 
    net_.fc1.weight.grad.data /= modulus #/ (net_.fc1.weight.data.norm(p=2, dim=1)).mean()
    modulus *= 1 #(net_.fc1.weight.data.norm(p=2, dim=0)).mean() / (net_.fc1.weight.data.norm(p=2, dim=1)).mean() 
    net_.conv2.weight.grad.data /= modulus * 25 /2 #/ (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1)).mean() * 25 #/ 2 * 25
    modulus *= 1 #(net_.conv2.weight.data.view(net_.conv2.weight.data.size(1), -1).norm(p=2, dim=1)).mean() / (net_.conv2.weight.data.view(net_.conv2.weight.data.size(0), -1).norm(p=2, dim=1)).mean()
    net_.conv1.weight.grad.data /= modulus *196 /2 #/ (net_.conv1.weight.data.view(net_.conv1.weight.data.size(0), -1).norm(p=2, dim=1)).mean() * 196 #/ 2 * 196

def DiagonalRescaling(net_):

    modulus = 1
    net_.fc3.weight.grad.data /= math.sqrt(net_.fc3.weight.data.size(1))
   
    net_.fc2.weight.grad.data /= math.sqrt(net_.fc2.weight.data.size(1))
    net_.fc1.weight.grad.data /= math.sqrt(net_.fc1.weight.data.size(1))
    net_.conv2.weight.grad.data /= math.sqrt(net_.conv2.weight.data.size(1) * 25) #/ 2 * 25
    net_.conv1.weight.grad.data /= math.sqrt(net_.conv1.weight.data.size(1) * 196)
       ########################################################################


# This is when things start to get interesting.

# We simply have to loop over our data iterator, and feed the inputs to the

# network and optimize

loss1_record = []
loss2_record = []
test1_record = []
test2_record = []


def train(epoch):
    running_loss = 0.0
    running_loss2 = 0.0
    # scheduler.step()
    # scheduler2.step()

    for i, data in enumerate(trainloader, 0):

        # get the inputs

        inputs, labels = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)

        # # zero the parameter gradients

        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = net(inputs)
    
        loss = criterion(outputs, labels)

        loss.backward()
        OurBlockDiagonal4(net)

        optimizer.step()

        # print statistics

        running_loss += loss.data[0]

        if i % args.log_interval == args.log_interval - 1:    # print every 2000 mini-batches

            print('Net1 [%d, %5d] loss: %.3f' %

                  (epoch, i, running_loss / (i+1)))
  
        # net2 without rebalance tricks
        optimizer2.zero_grad()

        outputs2 = net2(inputs)

        loss2 = criterion(outputs2, labels)

        loss2.backward()
        
        optimizer2.step()
        # print statistics

        running_loss2 += loss2.data[0]

        if i % args.log_interval == args.log_interval - 1:    # print every 2000 mini-batches

            print('Net2 [%d, %5d] loss: %.3f' %(epoch, i, running_loss2 / (i+1)))
    loss1_record.append(running_loss / (i+1))
    loss2_record.append(running_loss2 / (i+1))




########################################################################

# 5. Test the network on the test data


def test():
    correct = 0

    total = 0

    for data in testloader:

        images, labels = data
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(Variable(images))

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum()

    test1_record.append(correct)

    # print('Accuracy of the network on the 10000 test images: %d %%' % (

    #     100 * correct / total))
    print('Accuracy of the network on the 10000 test images: %d %%' % (correct))


    # Test of net2
    correct = 0

    total = 0

    for data in testloader:

        images, labels = data
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs2 = net2(Variable(images))

        _, predicted = torch.max(outputs2.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum()

    test2_record.append(correct)

    print('Accuracy of the Net2 on the 10000 test images: %d %%' % (correct))
    # print('\nModel1 Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))



for epoch in range(args.epochs):
    train(epoch)
    test()
    if epoch % 2== 0:
        for i, p in enumerate(net.parameters()):
            print("net layer ", i, " norm   ", p.data.norm(p=2))
        for i, p in enumerate(net2.parameters()):
            print("net2 layer ", i, " norm   ", p.data.norm(p=2))
    if epoch % 50 == 0:
        print("\nloss1 = ", loss1_record)
        print("test1 = ", test1_record)

        print("\nloss2 = ", loss2_record)
        print("test2 = ", test2_record)
    

print("optimizer = optim.SGD(net.parameters(), lr=", args.lr," momentum=", args.momentum,") \n optimizer2 = optim.SGD(net2.parameters(), lr=",args.lr2," momentum=",args.momentum2,")")
print("\nloss1 = ", loss1_record)
print("test1 = ", test1_record)

print("\nloss2 = ", loss2_record)
print("test2 = ", test2_record)

########################################################################