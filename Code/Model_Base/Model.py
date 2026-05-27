import torch
from torch import nn

class MLP(torch.nn.Module):
    n_input=100
    def __init__(self,n_input,n_hidden,n_output,n_layers):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_layers = n_layers
    
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.d1  = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.d2  = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.d3  = nn.ReLU()
        self.fc4 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        n_batchSize = x.size()[0]
        x = x.view(x.size(0), -1)
        x = self.d1(self.fc1(x))
        x = self.d2(self.fc2(x))        
        x = self.d3(self.fc3(x))
        x = self.fc4(x)
        x = x.view(n_batchSize,-1)
        return x
    
class MLPGELU(torch.nn.Module):
    n_input=100
    def __init__(self,n_input,n_hidden,n_output,n_layers):
        super(MLPGELU, self).__init__()
        self.n_input = n_input
        self.n_layers = n_layers
    
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.d1  = nn.GELU()
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.d2  = nn.GELU()
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.d3  = nn.GELU()
        self.fc4 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        n_batchSize = x.size()[0]
        x = x.view(x.size(0), -1)
        x = self.d1(self.fc1(x))
        x = self.d2(self.fc2(x))        
        x = self.d3(self.fc3(x))
        x = self.fc4(x)
        x = x.view(n_batchSize,-1)
        return x



class AlexNet(nn.Module):
    def __init__(self, in_channels, datasize, classes=2):
        super().__init__()
        # 特征提取部分
        self.in_channels = in_channels
        self.datasize = datasize
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=5, stride=1, padding=3),  # 一维卷积
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 一维池化
            nn.Conv1d(96, 256, kernel_size=3, stride=1),  # 一维卷积
            nn.ReLU(),
            # nn.Dropout(0.95),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 一维池化
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 一维池化
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.ReLU(),
            # nn.Dropout(0.95),
            nn.Conv1d(512, 256, kernel_size=3, stride=1),  # 一维卷积
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3)  # 一维池化
        )
        
        # 计算展平后的维度
        self._initialize_flatten_size()

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, classes)
        )

    def _initialize_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1,self.in_channels, self.datasize)
            output = self.features(dummy_input)
            self.flatten_size = output.view(-1).shape[0]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
class AlexNetGELU(nn.Module):
    def __init__(self, in_channels, datasize, classes=2):
        super().__init__()
        # 特征提取部分
        self.in_channels = in_channels
        self.datasize = datasize
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=5, stride=1, padding=3),  # 一维卷积
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 一维池化
            nn.Conv1d(96, 256, kernel_size=3, stride=1),  # 一维卷积
            nn.GELU(),
            # nn.Dropout(0.95),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 一维池化
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.GELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 一维池化
            nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.GELU(),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.GELU(),
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),  # 一维卷积
            nn.GELU(),
            # nn.Dropout(0.95),
            nn.Conv1d(512, 256, kernel_size=3, stride=1),  # 一维卷积
            nn.GELU(),
            nn.AvgPool1d(kernel_size=3, stride=3)  # 一维池化
        )
        
        # 计算展平后的维度
        self._initialize_flatten_size()

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, classes)
        )

    def _initialize_flatten_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1,self.in_channels, self.datasize)
            output = self.features(dummy_input)
            self.flatten_size = output.view(-1).shape[0]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual
    
class ResNet(torch.nn.Module):
    def __init__(self,in_channels=1,classes=2):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            #
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            #
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            #
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x
    
class BottlrneckGELU(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(BottlrneckGELU, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.GELU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.GELU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.GELU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual
    
class ResNetGELU(torch.nn.Module):
    def __init__(self,in_channels=1,classes=2):
        super(ResNetGELU, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            BottlrneckGELU(64,64,256,False),
            BottlrneckGELU(256,64,256,False),
            BottlrneckGELU(256,64,256,False),
            #
            BottlrneckGELU(256,128,512, True),
            BottlrneckGELU(512,128,512, False),
            BottlrneckGELU(512,128,512, False),
            BottlrneckGELU(512,128,512, False),
            #
            BottlrneckGELU(512,256,1024, True),
            BottlrneckGELU(1024,256,1024, False),
            BottlrneckGELU(1024,256,1024, False),
            BottlrneckGELU(1024,256,1024, False),
            BottlrneckGELU(1024,256,1024, False),
            BottlrneckGELU(1024,256,1024, False),
            #
            BottlrneckGELU(1024,512,2048, True),
            BottlrneckGELU(2048,512,2048, False),
            BottlrneckGELU(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048,classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x