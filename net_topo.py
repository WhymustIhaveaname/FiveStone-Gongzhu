import torch
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
        self.conv1=nn.Conv2d(3,64, kernel_size=5,padding=0)
        #self.bn1=nn.BatchNorm2d(128)
        self.conv2=nn.Conv2d(64,128,kernel_size=3,padding=1)
        #self.bn2=nn.BatchNorm2d(128)
        self.fn1=nn.Linear(128*5*5,1024)
        self.fn2=nn.Linear(1024,256)
        self.fn3=nn.Linear(256,256)
        self.fnv=nn.Linear(256,1)
        self.fnp=nn.Linear(256,81)

    def forward(self, x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=F.relu(self.fn1(out.view(-1,3200)))
        out=F.relu(self.fn2(out))
        out=F.relu(self.fn3(out))
        return self.fnp(out),self.fnv(out)

class FiveStone_CNN(FiveStoneState):
    kern_5 = FiveStoneState.kernal_5.cuda()
    kern_possact = torch.tensor([[[[1.,1,1],[1,-1024,1],[1,1,1]]]]).cuda()

    def __init__(self, model):
        self.board = torch.zeros(9,9).cuda()
        self.board[4,4] = 1.0
        self.currentPlayer = -1
        self.model = model

    def reset(self):
        FiveStoneState.reset(self)
        self.board = self.board.cuda()

    def getPossibleActions(self,printflag=False):
        cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact, padding=1)
        if printflag:
            print(cv.type(torch.int8))
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]

    def isTerminal(self):
        conv1 = F.conv2d(self.board.view(1,1,9,9), self.kern_5, padding=2)
        if conv1.max() >= 0.9 or conv1.min() <= -0.9:
            return True
        if self.board.abs().sum()==81:
            return True
        return False

    def getReward(self):
        conv1 = F.conv2d(self.board.view(1,1,9,9), FiveStone_CNN.kern_5, padding=2)
        if conv1.max() >= 0.9:
            return torch.tensor([1.0], device="cuda")
        elif conv1.min() <= -0.9:
            return torch.tensor([-1.0], device="cuda")
        if self.board.sum()==81:
            return torch.tensor([0.0], device="cuda")

        with torch.no_grad():
            input_data=self.gen_input().view((1,3,9,9))
            _,value = self.model(input_data).clip(-0.99,0.99)
            value=value.view(1)
        return value

    def gen_input(self):
        return torch.stack([(self.board==1).type(torch.cuda.FloatTensor),
                                (self.board==-1).type(torch.cuda.FloatTensor),
                                torch.ones(9,9,device="cuda")*self.currentPlayer])