#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MCTS.mcts import abpruning
from fivestone_conv import log, FiveStoneState,pretty_board
from net_topo import PVnet_cnn, FiveStone_CNN

import torch,re,time,copy,random,itertools
import torch.nn as nn
import torch.nn.functional as F

# unbalance openings with white/black in advantage
# the first black stone (0,0) is omitted
open_unbl_white=[[(1,1),(-4,-4)],
                [(1,1),(4,4),(2,0),(4,-4)],
                [(1,1),(4,4),(2,0),(4,-4),(2,2),(-4,4)],
                [(1,1),(4,4),(2,0),(4,-4),(2,2),(-4,4),(2,1),(-4,-4)]]
open_unbl_black=[[(-4,-4),(1,1)],
                [(-4,-4),(1,1),(4,4),(2,2)],
                [(-4,-4),(1,1),(4,4),(2,2),(4,-4),(0,2)],
                [(-4,-4),(1,1),(4,4),(2,2),(4,-4),(0,2),(-4,4),(0,1)]]
# balance openings, the first black stone (0,0) is omitted
open_bl=[[(1,1),(2,2)],[(1,1),(2,-2)],
        [(0,1),(0,2)],[(0,1),(1,2)],
        [(0,1),(2,-2)],[(1,1),(2,-2)],
        [(1,1),(2,1)],[(0,1),(2,2)],
        [(1,1),(2,0)],[(0,1),(1,1)]]

def vs_noth(state_nn,epoch):
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    for i1,i2 in itertools.product(range(len(open_bl)),[100,101,102,103]):
        state_nn.reset()
        state_nn.track_hist(open_bl[i1],rot=i2)
        while not state_nn.isTerminal():
            state_nn.currentPlayer=1
            state_nn=state_nn.takeAction(state_nn.policy_choice_best())
        l_ans.append(state_nn.board.abs().sum().item())
    log("epoch %d avg win steps: %d/%d=%.1f"%(epoch,sum(l_ans),len(l_ans),sum(l_ans)/len(l_ans)))

def vs_rand(state_nn,epoch):
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    l_loss=[]
    for i1,nn_color,rot in itertools.product(range(len(open_bl)),[-1,1],[100,101,102,103]):
        state_nn.reset()
        state_nn.track_hist(open_bl[i1],rot=rot)
        while not state_nn.isTerminal():
            if state_nn.currentPlayer==nn_color:
                state_nn=state_nn.takeAction(state_nn.policy_choice_best())
            else:
                state_nn=state_nn.takeAction(random.choice(state_nn.getPossibleActions()))
        result=nn_color*state_nn.getReward()
        if result==1:
            l_ans.append(state_nn.board.abs().sum().item())
        else:
            l_loss.append(result)
    win_rate=len(l_ans)/(len(l_ans)+len(l_loss))*100
    log("epoch %d avg win steps: %d/%d=%.1f, %.1f%%"%(epoch,sum(l_ans),len(l_ans),sum(l_ans)/len(l_ans),win_rate))

def benchmark_color(state_nn,nn_color,openings,epoch):
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    state_conv = FiveStoneState()
    for i1,i2 in itertools.product(range(len(openings)),[100,101,102,103]):
        state_nn.reset()
        state_conv.reset()
        state_nn.track_hist(openings[i1],rot=i2)
        state_conv.track_hist(openings[i1],rot=i2)
        while not state_nn.isTerminal():
            if state_nn.currentPlayer==nn_color:
                action=state_nn.policy_choice_best()
            else:
                searcher.search(initialState=state_conv)
                best=[(k,v) for k,v in searcher.children.items()]
                best=max(best,key=lambda x:x[1]*state_nn.currentPlayer)
                action=best[0]
            state_nn=state_nn.takeAction(action)
            state_conv=state_conv.takeAction(action)
        result=nn_color*state_conv.getReward()
        if result==10000:
            l_ans.append("w")
        elif result==-10000:
            l_ans.append("l")
        elif result==0:
            l_ans.append("d")
        else:
            l_ans.append("%s"%(i))
    color_dict={1:"bk",-1:"wt"}
    l_ans=" ".join(l_ans)
    log("epoch %d nn_color %s: %s"%(epoch,color_dict[nn_color],l_ans))
    return l_ans

def benchmark(model,epoch):
    l_bk=benchmark_color(model,1,open_unbl_black,epoch)
    l_wt=benchmark_color(model,-1,open_unbl_white,epoch)
    if l_bk.startswith("w w"):
        benchmark_color(model,1,open_bl,epoch)
    if l_wt.startswith("w w"):
        benchmark_color(model,-1,open_bl,epoch)
"""
def select_by_prob(children,player,softk):
    l=[(k,v) for k,v in children.items()]
    lv=torch.tensor([v for k,v in l])*player*softk
    lv=F.softmax(lv,dim=0)
    r=torch.multinomial(lv,1)
    return l[r][0]

def gen_data_supervised():
    softk=0.5
    train_datas=[]
    searcher=abpruning(deep=1,n_killer=2)
    for i in range(len(open_bl)):
        state = FiveStoneState()
        state = state.track_hist(open_bl[i])
        while not state.isTerminal():
            searcher.search(initialState=state)
            if state.currentPlayer==1:
                best_value=max(searcher.children.values())
            elif state.currentPlayer==-1:
                best_value=min(searcher.children.values())
            else:
                log("what's your problem?!",l=2)

            input_data=FiveStone_CNN.gen_input(state)
            target_v=torch.tensor([best_value/10]).cuda()
            lkv=[(k,v) for k,v in searcher.children.items()]
            lv=torch.tensor([v for k,v in lkv])
            lv=lv*state.currentPlayer*softk
            lv=F.softmax(lv,dim=0)
            target_p=torch.zeros(9,9,device="cuda")
            for j in range(len(lkv)):
                target_p[lkv[j][0]]=lv[j]
            legal_mask=(state.board==0).type(torch.cuda.FloatTensor)
            train_datas.append((input_data,target_v,target_p.view(-1),legal_mask.view(-1)))

            next_action=select_by_prob(searcher.children,state.currentPlayer,softk=softk)
            state=state.takeAction(next_action)
    return train_datas

def train_supervised():
    model = PVnet_cnn().cuda()
    log(model)
    optim = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optim: %s"%(optim.__dict__['defaults'],))

    for epoch in range(100):
        if epoch%1==0 and epoch>0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)
            #benchmark(model)
            vs_rand(model)
        train_datas=[]
        for _ in range(1):
            train_datas += gen_data_supervised()
        log("epoch %d with %d datas"%(epoch,len(train_datas)))
        trainloader = torch.utils.data.DataLoader(train_datas,batch_size=64,shuffle=True)
        if True:
            for batch in trainloader:
                policy,value = model(batch[0])
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")
                optim.zero_grad()
                loss_p.backward(retain_graph=True)
                log("loss_p: %8.4f, grad_p_fn1: %.8f, grad_p_conv1: %.8f"%(loss_p.item(),
                    model.fn1.weight.grad.abs().mean().item(),
                    model.conv1.weight.grad.abs().mean().item()))

                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                optim.zero_grad()
                loss_v.backward(retain_graph=True)
                log("loss_v: %8.4f, grad_v_fn1: %.8f, grad_p_conv1: %.8f"%(loss_v.item(),
                    model.fn1.weight.grad.abs().mean().item(),
                    model.conv1.weight.grad.abs().mean().item()))
                break

        for age in range(21):
            running_loss = 0.0
            for batch in trainloader:
                policy,value = model(batch[0])
                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")
                optim.zero_grad()
                loss=loss_v+loss_p*0.3
                loss.backward()
                optim.step()
                running_loss += loss.item()
            if age<3 or age%10==0:
                log("    age %2d: %.6f"%(age,running_loss/len(train_datas)))

if __name__=="__main__":
    train_supervised()
"""