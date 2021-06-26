import torch,itertools
import torch.nn as nn
import torch.nn.functional as F

from fivestone_conv import log, FiveStoneState

class PVnet_cnn(nn.Module):
    def num_paras(self):
        return sum([p.numel() for p in self.parameters()])

    def num_layers(self):
        ax=0
        for name,child in self.named_children():
            ax+=1
        return ax

    def __str__(self):
        stru=[]
        for name,child in self.named_children():
            if 'weight' in child.state_dict():
                #stru.append(tuple(child.state_dict()['weight'].t().size()))
                stru.append(child.state_dict()['weight'].shape)
        return "%s %s %s"%(self.__class__.__name__,stru,self.num_paras())

    def __init__(self):
        super(PVnet_cnn, self).__init__()
        self.conv1=nn.Conv2d(3,128,kernel_size=5,padding=0)
        self.bn1=nn.BatchNorm2d(128)
        self.conv2=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(256)
        self.conv3=nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(512)
        self.conv4=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn4=nn.BatchNorm2d(512)
        self.fn1=nn.Linear(512*5*5,64)
        self.fn2=nn.Linear(64,64)
        self.fn3=nn.Linear(64,64)
        self.fnv=nn.Linear(64,1)
        self.fnp=nn.Linear(64,81)

    def forward(self, x):
        assert (x[:,0,:,:]*x[:,1,:,:]).abs().sum()==0
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn3(self.conv3(out)))
        out=F.relu(self.bn4(self.conv4(out)))
        out=F.relu(self.fn1(out.view(-1,512*5*5)))
        out=F.relu(self.fn2(out))
        out=F.relu(self.fn3(out))
        return self.fnp(out),self.fnv(out)

class BasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_planes)
        self.conv2=nn.Conv2d(out_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_planes)

        if in_planes!=out_planes or stride!=1:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_planes)
            )
        else:
            self.shortcut=nn.Sequential()

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        #print(out.isnan().any())
        out=self.conv2(out)
        #print(out.isnan().any());input()
        out=self.bn2(out)
        out+=self.shortcut(x)
        out=F.relu(out)
        return out

class PV_resnet(PVnet_cnn):
    def __init__(self):
        super(PV_resnet,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=6,padding=2,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.num_conv_layers=1
        self.block_strs=["%dx%dx%d(padding=%d)"%(self.conv1.weight.shape[0],self.conv1.weight.shape[2],self.conv1.weight.shape[3],self.conv1.padding[0])]

        self.layer1 = self._make_layer(64,128,stride=1)
        self.layer2 = self._make_layer(128,256,stride=1)
        self.layer3 = self._make_layer(256,512,stride=1)

        self.fnp=nn.Linear(512*2*2,81)
        self.fnv=nn.Linear(512*2*2,1)

    def _make_layer(self,in_planes,out_planes,stride=1):
        self.num_conv_layers+=4
        self.block_strs.append("basic(%d,%d)"%(in_planes,out_planes))
        layers=[BasicBlock(in_planes,out_planes,stride=stride),BasicBlock(out_planes,out_planes)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.max_pool2d(out,2)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.max_pool2d(out,2)
        out=out.view(-1,512*2*2)
        p=self.fnp(out)
        v=self.fnv(out)
        return p,v

    def __str__(self):
        return "%s %s %d %s"%(self.__class__.__name__,"-".join(self.block_strs),self.num_conv_layers,self.num_paras())

class PV_resnet_wide(PV_resnet):
    def __init__(self):
        super(PV_resnet,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=6,padding=2,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.num_conv_layers=1

        self.layer1 = self._make_layer(64,256,stride=1)
        self.layer2 = self._make_layer(256,512,stride=1)
        self.layer3 = self._make_layer(512,1024,stride=1)

        self.fnp=nn.Linear(1024*2*2,81)
        self.fnv=nn.Linear(1024*2*2,1)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.max_pool2d(out,2)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.max_pool2d(out,2)
        out=out.view(-1,1024*2*2)
        p=self.fnp(out)
        v=self.fnv(out)
        return p,v

class PV_resnet_wide_mid(PV_resnet):
    def __init__(self):
        super(PV_resnet,self).__init__()
        self.conv1=nn.Conv2d(3,36,kernel_size=6,padding=2,bias=False)
        self.bn1=nn.BatchNorm2d(36)
        self.num_conv_layers=1

        self.layer1 = self._make_layer(36,256,stride=1)
        self.layer2 = self._make_layer(256,512,stride=1)
        self.layer3 = self._make_layer(512,1024,stride=1)

        self.fnp=nn.Linear(1024*2*2,81)
        self.fnv=nn.Linear(1024*2*2,1)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.max_pool2d(out,2)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=F.max_pool2d(out,2)
        out=out.view(-1,1024*2*2)
        p=self.fnp(out)
        v=self.fnv(out)
        return p,v

class PV_resnet_small(PV_resnet):
    def __init__(self):
        super(PV_resnet,self).__init__()
        self.conv1=nn.Conv2d(3,16,kernel_size=6,padding=2,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.num_conv_layers=1
        self.block_strs=["%dx%d(padding=%d)"%(self.conv1.weight.shape[0],self.conv1.weight.shape[2],self.conv1.padding[0])]
        self.block_strs.append("max_pool2d(2)")

        self.layer1 = self._make_layer(16,32,stride=1)
        self.layer2 = self._make_layer(32,64,stride=1)
        self.layer3 = self._make_layer(64,64,stride=1)
        self.layer4 = self._make_layer(64,64,stride=1)
        self.block_strs.append("avg_pool2d(3,2)")

        self.fnp=nn.Linear(64*3*3,225)
        self.fnv=nn.Linear(64*3*3,1)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.max_pool2d(out,2)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        #print("3",out.isnan().any());input()
        out=F.avg_pool2d(out,3,stride=2)
        out=out.view(-1,64*3*3)
        p=self.fnp(out)
        v=self.fnv(out)
        return p,v
