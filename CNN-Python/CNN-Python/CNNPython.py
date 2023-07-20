import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

#Loading training and testing set with data augementation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#Acts as the equation of a provided. Calculates the spatial average, applies a linear layer and then applies ReLU
class SpatialAveragePoolMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAveragePoolMLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_channels*1*1, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x

#Main class consisting of a modulelist containing the instances of blocks, 
#an examplenet class to increase depth and the classification class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = Classifier(in_channels, out_channels)
        self.blocks = nn.ModuleList([
            Block(in_channels if i==0 else out_channels+((i-1)*2) , out_channels+(i*2)) for i in range(num_blocks)#if i==0 else out_channels
        ])
        self.test = ExampleNet()
        

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.test(x)
        x = self.classifier(x)
        return x

#The classifier class uses the same average pooling function in the SAPMLP class to find the mean
#and applies a fully connected layer 
class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(1 * 1 * 16, 10)


    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x
    
#The ExampleNet class adds more depth to the model by adding more convolutional layers and linear layers
class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=4*32*32, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=16*5*5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = x.view(-1, 16, 5, 5)
        x = nn.functional.relu(self.conv3(x))
        return x    
    
#Block class consists of multiple convolutional layers and a single SpatialAveragePool layer (SpatialAveragePoolMLP class)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) for _ in range(out_channels)])
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool_mlp = SpatialAveragePoolMLP(in_channels, out_channels)

    def forward(self, x):
        final_output_flat = None
        a = self.pool_mlp(x)
        for i in range(len(self.layers)):
            col_1 = a[:, i]
            conv_output = self.layers[i](x)
            conv_output = self.bn(conv_output)
            col_1 = col_1.view(8, 1, 1, 1)
            if final_output_flat == None:
                final_output_flat = torch.mul(conv_output, col_1) 
            else:
                final_output_flat += torch.mul(conv_output, col_1) 

        return final_output_flat

in_channels = 3
out_channels = 8
num_blocks = 3

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

net = Net()#.to(device)


criterion = nn.CrossEntropyLoss()#.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
losses = []
train_accs = []
test_accs = []
epoch_num = []
correct1 = 0
total1 = 0
for epoch in range(30):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs#.to(device)
        labels = labels#.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)#.to(device)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()
        
        

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches for understanding loss progression
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
                
    train_acc = 100 * correct1 / total1
    train_accs.append(train_acc)
    epoch_num.append(epoch + 1)
    losses.append(round(loss.item(), 2))

print(losses[0])
print(losses[1])
# plot accuracies for current epoch
plt.figure(1)
plt.plot(epoch_num, train_accs, label='Training Accuracy')  

plt.figure(2)
plt.plot(epoch_num, losses, label='Loss Value')  
    

    
    
    

# plot labels and legend
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.figure(2)
plt.xlabel('Epoch')
plt.ylabel('Loss')

# show plots
plt.show()

print('Finished Training')



correct = 0
total = 0
#For the test set
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images#.to(device)
        labels = labels#.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)#.to(device)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    test_acc = 100 * correct / total
    test_accs.append(test_acc)
    
    
    
    

#prints the accuracy of the test set
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')