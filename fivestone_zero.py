#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch,re,time,copy,itertools
import torch.nn as nn
import torch.nn.functional as F

from MCTS.mcts import abpruning
from fivestone_conv import log, pretty_board
from net_topo import PV_resnet, FiveStone_CNN#, PVnet_cnn
from fivestone_cnn import open_bl,benchmark_color,benchmark,vs_rand,vs_noth

SOFTK=3/FiveStone_CNN.WIN_REWARD
ACTION_NUM=8
AB_DEEP=2
POSSACT_RAD=2
log("softk: %.1f, ACTION_NUM: %d, AB_DEEP: %d, POSSACT_RAD: %d"%(SOFTK,ACTION_NUM,AB_DEEP,POSSACT_RAD))

class FiveStone_ZERO(FiveStone_CNN):
    def getPossibleActions(self,target_num=ACTION_NUM):
        """cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
        #cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]"""
        input_data=self.gen_input().view((1,3,9,9))
        policy,value=self.model(input_data)
        policy=policy.view(9,9)
        if POSSACT_RAD==1:
            cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if cv[0,0,i,j]>0]
        elif POSSACT_RAD==2:
            cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if cv[0,0,i,j]>0]
        elif POSSACT_RAD>2:
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if self.board[i,j]==0]
        if len(lkv)<target_num:
            return [k for k,v in lkv]
        else:
            lkv.sort(key=lambda x:x[1],reverse=True)
            return [lkv[i][0] for i in range(target_num)]


def gen_data(model,num_games):
    train_datas=[]
    searcher=abpruning(deep=AB_DEEP,n_killer=2)
    state = FiveStone_ZERO(model)
    for i in range(num_games):
        state.reset()
        state.track_hist(open_bl[i%len(open_bl)],rot=i//len(open_bl))
        #duplicated_flag=False
        vidata=[]
        while not state.isTerminal():
            #searcher.counter=0
            searcher.search(initialState=state)
            #log(searcher.counter)

            if state.currentPlayer==1:
                best_value=max(searcher.children.values())
            elif state.currentPlayer==-1:
                best_value=min(searcher.children.values())
            else:
                log("what's your problem?!",l=2)

            lkv=[(k,v) for k,v in searcher.children.items()]
            lv=torch.tensor([v for k,v in lkv])*state.currentPlayer*SOFTK
            lv=F.softmax(lv,dim=0)
            target_p=torch.zeros(9,9).cuda()
            for j in range(len(lkv)):
                target_p[lkv[j][0]]=lv[j]
            legal_mask=(state.board==0).type(torch.cuda.FloatTensor)
            in_mat=state.gen_input()
            for j in range(4):
                this_data=(torch.rot90(in_mat,j,[1,2]),best_value,
                           torch.rot90(target_p,j,[0,1]).reshape(81),
                           torch.rot90(legal_mask,j,[0,1]).reshape(81))
                train_datas.append(this_data)
            if len(vidata)==0 and best_value.abs()>=FiveStone_CNN.WIN_REWARD*0.9:
                vidata=[in_mat,best_value,target_p,legal_mask]

            r=torch.multinomial(lv,1)
            state=state.takeAction(lkv[r][0])

        if len(vidata)>0 and vidata[1]*state.getReward()>0:
            in_mat,best_value,target_p,legal_mask=vidata
            vid_dup=4*int((state.board.abs().sum().item()-3)/8)
            #log("duplicating vid: %s"%(vidata,));input()
            for j in range(vid_dup):
                this_data=(torch.rot90(in_mat,j,[1,2]),best_value,
                           torch.rot90(target_p,j,[0,1]).reshape(81),
                           torch.rot90(legal_mask,j,[0,1]).reshape(81))
                train_datas.append(this_data)

    return train_datas

def train(model):
    optim = torch.optim.Adam(model.parameters(),lr=0.0005,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    loss_p_wt = 0.4
    log(model)
    log("loss_p_wt: %.1f, optim: %s"%(loss_p_wt,optim.__dict__['defaults'],))

    for epoch in range(400):
        if (epoch<40 and epoch%5==0) or epoch%20==0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)
            vs_noth(model,epoch)
            vs_rand(model,epoch)
            benchmark(model,epoch)

        train_datas = gen_data(model,10)
        trainloader = torch.utils.data.DataLoader(train_datas,batch_size=64,shuffle=True,drop_last=True)

        if epoch<3 or (epoch<40 and epoch%5==0) or epoch%20==0:
            print_flag=True
        else:
            print_flag=False

        if print_flag:
            log("epoch %d with %d datas"%(epoch,len(train_datas)))
            for batch in trainloader:
                policy,value = model(batch[0])
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")
                optim.zero_grad()
                loss_p.backward(retain_graph=True)
                log("loss_p: %6.4f, grad_p_conv1: %.8f"%(loss_p.item(),model.conv1.weight.grad.abs().mean().item()))

                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                optim.zero_grad()
                loss_v.backward(retain_graph=True)
                log("loss_v: %6.4f, grad_p_conv1: %.8f"%(loss_v.item(),model.conv1.weight.grad.abs().mean().item()))
                break

        for age in range(3):
            running_loss = 0.0
            for batch in trainloader:
                policy,value = model(batch[0])
                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")

                optim.zero_grad()
                loss=loss_v+loss_p*loss_p_wt
                loss.backward()
                optim.step()
                running_loss += loss.item()
            if print_flag and (age<3 or (age+1)%5==0):
                log("    age %2d: %.6f"%(age,running_loss/len(train_datas)))

def test_must_win(model):
    state = FiveStone_ZERO(model)
    for i in range(4):
        state.board[5,4+i]=-1
    state.currentPlayer=-1
    input_data=state.gen_input().view((1,3,9,9))
    policy,value=state.model(input_data)
    policy=policy.view(9,9)
    s=[]
    for i in range(9):
        s.append(" ".join(["%6.2f"%(policy[i,j]) for j in range(9)]))
    log("\n%s"%("\n".join(s)))
    log(value)
    pretty_board(state.takeAction(state.policy_choice_best()))

if __name__=="__main__":
    #model = PVnet_cnn().cuda()
    model = PV_resnet().cuda()
    start_file="./logs/6_1/PV_resnet-16-15857234-180.pkl"
    model.load_state_dict(torch.load(start_file,map_location="cuda"))
    log("load from %s"%(start_file))
    #test_must_win(model)
    train(model)