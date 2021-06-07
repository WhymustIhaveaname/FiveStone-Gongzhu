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
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out

class PV_resnet(PVnet_cnn):
    def __init__(self):
        super(PV_resnet,self).__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=5,padding=0,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.num_conv_layers=1

        self.layer1 = self._make_layer(64,128,stride=1)
        self.layer2 = self._make_layer(128,256,stride=2)
        self.layer3 = self._make_layer(256,512,stride=2)

        self.fnp=nn.Linear(512*2*2,81)
        self.fnv=nn.Linear(512*2*2,1)

    def _make_layer(self,in_planes,out_planes,stride=1):
        self.num_conv_layers+=4
        layers=[BasicBlock(in_planes,out_planes,stride=stride),BasicBlock(out_planes,out_planes)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=out.view(-1,512*2*2)
        p=self.fnp(out)
        v=self.fnv(out)
        return p,v

    def __str__(self):
        return "%s-%d %s"%(self.__class__.__name__,self.num_conv_layers,self.num_paras())

class FiveStone_CNN(FiveStoneState):
    kern_5 = FiveStoneState.kern_5.cuda()
    kern_possact_5x5 = torch.tensor([[[[1.,1,1,1,1],[1,2,2,2,1],[1,2,-1024,2,1],[1,2,2,2,1],[1,1,1,1,1]]]],device="cuda")
    kern_possact_3x3 = torch.tensor([[[[1.,1,1],[1,-1024,1],[1,1,1]]]]).cuda()
    WIN_REWARD=10.0

    def __init__(self, model):
        self.board = torch.zeros(9,9).cuda()
        self.board[4,4] = 1.0
        self.currentPlayer = -1
        self.model = model

    def reset(self):
        FiveStoneState.reset(self)
        self.board = self.board.cuda()

    def getPossibleActions(self,surpress_warning=False):
        if not surpress_warning:
            log("obsolete");input()
        cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
        #cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]

    def getReward(self):
        conv1 = F.conv2d(self.board.view(1,1,9,9), FiveStone_CNN.kern_5, padding=2)
        if conv1.max() >= 0.9:
            return torch.tensor([FiveStone_CNN.WIN_REWARD], device="cuda")
        elif conv1.min() <= -0.9:
            return torch.tensor([-FiveStone_CNN.WIN_REWARD], device="cuda")
        if self.board.sum()==81:
            return torch.tensor([0.0], device="cuda")

        with torch.no_grad():
            input_data=self.gen_input().view((1,3,9,9))
            _,value = self.model(input_data)
            value=value.view(1).clip(-FiveStone_CNN.WIN_REWARD*0.99,FiveStone_CNN.WIN_REWARD*0.99)
        return value

    def gen_input(self):
        return torch.stack([(self.board==1).type(torch.cuda.FloatTensor),
                            (self.board==-1).type(torch.cuda.FloatTensor),
                            torch.ones(9,9,device="cuda")*self.currentPlayer])

    def policy_choice_best(self):
        input_data=self.gen_input().view((1,3,9,9))
        policy,value=self.model(input_data)
        policy=policy.view(9,9)
        lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if self.board[i,j]==0]
        best=max(lkv,key=lambda x: x[1])
        return best[0]

    def policy_choice_softmax(self):
        input_data=self.gen_input().view((1,3,9,9))
        policy,value=self.model(input_data)
        policy=policy.view(9,9)
        lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if self.board[i,j]==0]
        lv=F.softmax(torch.tensor([v for k,v in lkv]),dim=0)
        r=torch.multinomial(lv,1)
        return lkv[r][0]