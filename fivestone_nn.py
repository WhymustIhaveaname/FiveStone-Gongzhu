from copy import deepcopy

from torch import tensor
from MCTS.mcts import abpruning

from fivestone_conv import log, FiveStoneState

import torch,re,time,copy
import torch.nn as nn
import torch.nn.functional as F

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
        self.fn4=nn.Linear(256,1)
        
    def forward(self, x):
        out=torch.stack([x==1,x==-1]).view(-1,2,9,9).type(torch.FloatTensor).cuda()
        out=F.relu(self.conv1(out))
        out=F.relu(self.conv2(out))
        out=out.view(-1,3200)
        #print(out.shape)
        out=F.relu(self.fn1(out))
        out=F.relu(self.fn2(out))
        out=F.relu(self.fn3(out))
        out=F.relu(self.fn4(out))
        
        return out


class FiveStone_NN(FiveStoneState):
    # kernal_hori = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[1/5,1/5,1/5,1/5,1/5],[0,0,0,0,0],[0,0,0,0,0]]])
    # kernal_diag = torch.tensor([[[1/5,0,0,0,0],[0,1/5,0,0,0],[0,0,1/5,0,0],[0,0,0,1/5,0],[0,0,0,0,1/5]]])
    # kernal_5 = torch.stack((kernal_hori, kernal_diag, kernal_hori.rot90(1,[1,2]), kernal_diag.rot90(1,[1,2])))
    kernal_5 = FiveStoneState.kernal_5.cuda()
    kern_possact = torch.tensor([[[[1.,1,1,1,1],[1,2,2,2,1],[1,2,-1024,2,1],[1,2,2,2,1],[1,1,1,1,1]]]]).cuda()

    def __init__(self, model):
        self.board = torch.zeros(9,9).cuda()
        self.board[4,4] = 1.0
        self.currentPlayer = -1
        self.model = model

    def getPossibleActions(self,printflag=False):
        cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact, padding=2)
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
        with torch.no_grad():
            value = self.model(self.board).view(1)
        return value

def select_by_prob(children,player):
    #tik=time.time()
    l=[(k,v) for k,v in children.items()]
    lv=torch.tensor([v for k,v in l])*player
    lv=F.softmax(lv,dim=0)
    r=torch.multinomial(lv,1)
    #tok=time.time()
    #log((tok-tik)*1000)
    return l[r][0]

def gen_data(model, epoch=-1, data_round=16):
    train_datas=[]
    searcher=abpruning(deep=1,n_killer=2)
    for _ in range(data_round):
        state = FiveStone_NN(model)
        #log("playing")
        while not state.isTerminal():
            searcher.search(initialState=state)
            if state.currentPlayer==1:
                best_value=max(searcher.children.values())
            elif state.currentPlayer==-1:
                best_value=min(searcher.children.values())
            else:
                log("what's your problem?!",l=2)
            train_datas.append((state.board,best_value))
            next_action=select_by_prob(searcher.children,state.currentPlayer)
            state=state.takeAction(next_action)

            # best=[(k,v) for k,v in searcher.children.items()]
            # best=max(best,key=lambda x:x[1]*state.currentPlayer)
            # train_datas.append((state.board,best[1]))
            # state=state.takeAction(best[0])
    #log("collected %d datas"%(len(train_datas)))
    if epoch%10==0:
        log("last board of epoch %d\n%s"%(epoch,state.board.type(torch.int8)))
    return train_datas

def benchmark_color(model,nn_color):
    color_dict={1:"bk",-1:"wt"}
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    for _ in range(10):
        state_nn = FiveStone_NN(model)
        state_conv = FiveStoneState()
        while not state_nn.isTerminal():
            if state_nn.currentPlayer==nn_color:
                searcher.search(initialState=state_nn)
            elif state_nn.currentPlayer==nn_color*-1:
                searcher.search(initialState=state_conv)
            else:
                log("what's your problem?!",l=2)

            best=[(k,v) for k,v in searcher.children.items()]
            best=max(best,key=lambda x:x[1]*state_nn.currentPlayer)
            state_nn=state_nn.takeAction(best[0])
            state_conv=state_conv.takeAction(best[0])
        l_ans.append(nn_color*state_conv.getReward())
    win=len([0 for i in l_ans if i==10000])
    loss=len([0 for i in l_ans if i==-10000])
    wtf=[i for i in l_ans if (i!=-10000 and i!=10000)]
    log("nn_color %s: %.1f%% %.1f%% %s"%(color_dict[nn_color],win/len(l_ans)*100,loss/len(l_ans)*100,wtf))

def benchmark(model):
    log("benchmarking...")
    benchmark_color(model,1)
    benchmark_color(model,-1)

def train_model():
    model = PVnet().cuda()
    log(model)
    critien = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(),lr=0.007)

    for epoch in range(100):
        if epoch%20==0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)
            benchmark(model)
        train_datas = gen_data(model,epoch=epoch,data_round=8)
        log("epoch %d with %d datas"%(epoch,len(train_datas)))
        trainloader = torch.utils.data.DataLoader(train_datas,batch_size=32,shuffle=True)
        for age in range(3):
            running_loss = 0.0
            for batch in trainloader:
                value = model(batch[0])
                loss = critien(batch[1], value)
                optim.zero_grad()
                loss.backward()
                optim.step()
                running_loss += loss.item()
            log("    age %d: %.6f"%(age,running_loss/len(train_datas)))

def test_eg1():
    state = FiveStone_NN(None)
    state = state.track_hist([(3,3),])
    print(state.board)
    print(state.getPossibleActions(printflag=True))

if __name__=="__main__":
    #test_eg1()
    train_model()
            
    