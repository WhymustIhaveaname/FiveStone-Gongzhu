import torch
import torch.nn as nn
import torch.nn.functional as F

from fivestone_conv import log, FiveStoneState

class PVnet(nn.Module):
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
        super(PVnet, self).__init__()
        self.conv1=nn.Conv2d(2, 128, kernel_size=5,padding=0)
        #self.bn1=nn.BatchNorm2d(128)
        self.conv2=nn.Conv2d(128,128,kernel_size=3,padding=1)
        #self.bn2=nn.BatchNorm2d(128)
        self.fn1=nn.Linear(128*5*5,1024)
        self.fn2=nn.Linear(1024,256)
        self.fn3=nn.Linear(256,256)
        self.fnv=nn.Linear(256,1)
        self.fnp=nn.Linear(256,81)
        
    def forward(self, x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=out.view(-1,3200)
        #print(out.shape)
        out=F.relu(self.fn1(out))
        out=F.relu(self.fn2(out))
        out=F.relu(self.fn3(out))
        return self.fnp(out),self.fnv(out)

class FiveStone_NN(FiveStoneState):
    # kernal_hori = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[1/5,1/5,1/5,1/5,1/5],[0,0,0,0,0],[0,0,0,0,0]]])
    # kernal_diag = torch.tensor([[[1/5,0,0,0,0],[0,1/5,0,0,0],[0,0,1/5,0,0],[0,0,0,1/5,0],[0,0,0,0,1/5]]])
    # kernal_5 = torch.stack((kernal_hori, kernal_diag, kernal_hori.rot90(1,[1,2]), kernal_diag.rot90(1,[1,2])))
    kernal_5 = FiveStoneState.kernal_5.cuda()
    kern_possact = torch.tensor([[[[1.,1,1],[1,-1024,1],[1,1,1]]]]).cuda()

    def __init__(self, model):
        self.board = torch.zeros(9,9).cuda()
        self.board[4,4] = 1.0
        self.currentPlayer = -1
        self.model = model

    def getPossibleActions(self,printflag=False):
        cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact, padding=1)
        if printflag:
            print(cv.type(torch.int8))
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]

    def getReward(self):
        conv1 = F.conv2d(self.board.view(1,1,9,9), FiveStone_NN.kernal_5, padding=2)
        if conv1.max() >= 0.9:
            return torch.tensor([1.0], device="cuda")
        elif conv1.min() <= -0.9:
            return torch.tensor([-1.0], device="cuda")
        elif self.board.sum()==81:
            return torch.tensor([0.0], device="cuda")
        with torch.no_grad():
            input_data=torch.stack([self.board==1,self.board==-1]).view(1,2,9,9).type(torch.cuda.FloatTensor)
            _,value = self.model(input_data)
            value=value.view(1)
        return value