import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim




'Some implementations of shallow to medium sized CNNs used in the experiments. '
'The key factor of these CNN architectures is that does not require normalization layers.'

# McMahan et al., 2016; 1,663,370 parameters
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
      #   self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=(32 * 2) * (7 * 7), out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        #x = F.log_softmax(x, dim = 1)
        return x






class TwoNN(nn.Module):
    def __init__(self):
        super(TwoNN, self).__init__()
     #   self.name = name
        self.activation = nn.ReLU(True)
        #self.bn1  = nn.BatchNorm1d(200) 
       # self.bn2 = nn.BatchNorm1d(200)

        self.fc1 = nn.Linear(in_features=784, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=10, bias=True)

   
    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        #x = self.activation((self.fc1(x)))
        #x = self.activation(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        #x = F.log_softmax(x, dim = 1)
        return x

class LeNet5(nn.Module):
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet5, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # classification
        x = x.view(x.size(0),-1) # flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class Big_CNN(nn.Module):

    def __init__(self,ch_dim=3, num_classes=10):
        super(Big_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class CNN_CIFAR10(nn.Module):

    def __init__(self,ch_dim=3, fc_nodes=1024, num_classes=10):
        super(CNN_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(ch_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            )
        self.classifier = nn.Linear(fc_nodes, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




'Helper functions that initialize the models that are used in the simulations, including their optimizers and their lr schedulers, '
'enabling step lr implementations.'

def define_CIFAR10_CNN(dev, num_clients, lr, momentum, weight_decay, nesterov,milestones):
    client_models = []
    for _ in range(num_clients):
        net = CNN_CIFAR10().to(dev)
        #net = torch.compile(net, mode='reduce-overhead')
        client_models.append(net)


    optimizers = []
    for client_index in range(num_clients):
      
        opt = optim.SGD(client_models[client_index].parameters(), lr=lr, momentum = momentum, weight_decay=weight_decay, nesterov=nesterov)
        
        optimizers.append(opt)
    

   
    
    lr_s = []
    for client_index in range(num_clients):
        lrs = optim.lr_scheduler.MultiStepLR(optimizers[client_index], gamma = 0.1, milestones=milestones)
        lr_s.append(lrs)



    return client_models, optimizers, lr_s

def define_FashionMNIST_CNN(dev, num_clients, lr, momentum, weight_decay, nesterov,milestones):
    client_models = []
    for _ in range(num_clients):
        net = CNN().to(dev)
        client_models.append(net)


    optimizers = []
    for client_index in range(num_clients):
        opt = optim.SGD(client_models[client_index].parameters(), lr=lr, momentum = momentum, weight_decay=weight_decay, nesterov=nesterov)
        optimizers.append(opt)
    

   
    
    lr_s = []
    for client_index in range(num_clients):
        lrs = optim.lr_scheduler.MultiStepLR(optimizers[client_index], gamma = 0.1, milestones=milestones)
        lr_s.append(lrs)



    return client_models, optimizers, lr_s


