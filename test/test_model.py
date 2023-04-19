import torch
import torch.nn as nn
import torch.nn.functional as F
from mytorchsummary import summary


class Define_Module(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Define_Module, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Define_List(nn.Module):
    def __init__(self, stack_num=2):
        super(Define_List,self).__init__()
        self.layerlist = nn.ModuleList([Define_Module(30, 30) for i in range(stack_num)])

    def forward(self, x):
        for i, layer in enumerate(self.layerlist):
            x = layer(x)
        return x


class SingleInputNet(nn.Module):
    def __init__(self):
        super(SingleInputNet, self).__init__()
        self.conv_1 = nn.Conv2d(1, 40, kernel_size=3)
        self.bottleneck = Define_Module(40, 40)
        self.conv_2 = nn.Conv2d(40, 30, kernel_size=3, padding=1)
        self.defineList = Define_List(1)
        self.fc1 = nn.Linear(5070, 10)

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.bottleneck(x)
        x = self.conv_2(x)
        x = self.defineList(x)
        x = x.view(-1, 5070)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)


class MultipleInputNet(nn.Module):
    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    def __init__(self):
        super(MultipleInputNetDifferentDtypes, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)

        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.FloatTensor)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        # set x2 to FloatTensor
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


def test_single_input():
    model = SingleInputNet()
    input = (1, 28, 28)
    total_params, trainable_params = summary(model, input, device="cuda:0" if torch.cuda.is_available() else "cpu")


def test_multiple_input():
    model = MultipleInputNet()
    input1 = (1, 300)
    input2 = (1, 300)
    total_params, trainable_params = summary(model, [input1, input2], device="cuda:0" if torch.cuda.is_available() else "cpu")


def test_multiple_input_types():
    model = MultipleInputNetDifferentDtypes()
    input1 = (1, 300)
    input2 = (1, 300)
    dtypes = [torch.FloatTensor, torch.LongTensor]
    total_params, trainable_params = summary(model, [input1, input2], device="cuda:0" if torch.cuda.is_available() else "cpu", dtypes=dtypes)


if __name__ == '__main__':
    test_single_input()