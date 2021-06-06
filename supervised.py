#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MCTS.mcts import abpruning
from fivestone_conv import log, FiveStoneState
from net_topo import PVnet, FiveStone_NN

import torch,re,time,copy
import torch.nn as nn
import torch.nn.functional as F

openings_unbl=[[(1,1),(-4,-4)],
          [(1,1),(4,4),(2,0),(4,-4)],
          [(1,1),(4,4),(2,0),(4,-4),(2,2),(-4,4)],
          [(1,1),(4,4),(2,0),(4,-4),(2,2),(-4,4),(2,1),(-4,-4)],
          [(-4,-4),(1,1)],
          [(-4,-4),(1,1),(4,4),(2,2)],
          [(-4,-4),(1,1),(4,4),(2,2),(4,-4),(0,2)],
          [(-4,-4),(1,1),(4,4),(2,2),(4,-4),(0,2),(-4,4),(0,1)],
          [(-4,-4),(1,1),(4,4),(2,2),(4,-4),(3,3)]]
openings_bl=[[(1,1),(2,2)],[(1,1),(2,-2)],
             [(0,1),(0,2)],[(0,1),(1,2)],
             [(0,1),(2,-2)],[(1,1),(2,-2)],
             [(1,1),(2,1)],[(0,1),(2,2)],
             [(1,1),(2,0)],[(0,1),(1,1)]]

def select_by_prob(children,player,softk=2.0):
    l=[(k,v) for k,v in children.items()]
    lv=torch.tensor([v for k,v in l])*player*softk
    lv=F.softmax(lv,dim=0)
    r=torch.multinomial(lv,1)
    return l[r][0]

def gen_data_supervised():
    softk=0.5
    train_datas=[]
    searcher=abpruning(deep=1,n_killer=2)
    for i in range(len(openings_bl)):
        state = FiveStoneState()
        state = state.track_hist(openings_bl[i])
        # print(state.board)
        while not state.isTerminal():
            searcher.search(initialState=state)
            if state.currentPlayer==1:
                best_value=max(searcher.children.values())
            elif state.currentPlayer==-1:
                best_value=min(searcher.children.values())
            else:
                log("what's your problem?!",l=2)
            input_data=torch.stack([state.board==1,state.board==-1]).type(torch.cuda.FloatTensor)
            target_v=torch.tensor(best_value/10).view(-1).cuda()
            target_p=torch.zeros(9,9).cuda()
            lkv=[(k,v) for k,v in searcher.children.items()]
            lv=torch.tensor([v for k,v in lkv])*state.currentPlayer*softk
            lv=F.softmax(lv,dim=0)
            for j in range(len(lkv)):
                target_p[lkv[j][0]]=lv[j]
            legal_mask=(state.board==0).type(torch.cuda.FloatTensor)
            train_datas.append((input_data,target_v,target_p.view(-1),legal_mask.view(-1)))
            next_action=select_by_prob(searcher.children,state.currentPlayer,softk=softk)
            state=state.takeAction(next_action)
        # print(state.board)
        # input()
        # log(len(train_datas))
    return train_datas

def train_supervised():
    model = PVnet().cuda()
    log(model)
    optim = torch.optim.Adam(model.parameters(),lr=0.01,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optim: %s"%(optim.__dict__['defaults'],))

    for epoch in range(100):
        if epoch%20==0 and epoch>0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)
            benchmark(model)
        train_datas=[]
        for _ in range(5):
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
                log("loss_p: %6.4f, grad_p_fn1: %.12f"%(loss_p.item(),model.fn1.weight.grad.abs().mean().item()))
                log("loss_p: %6.4f, grad_p_conv1: %.12f"%(loss_p.item(),model.conv1.weight.grad.abs().mean().item()))
                
                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                optim.zero_grad()
                loss_v.backward(retain_graph=True)
                log("loss_v: %6.4f, grad_v_fn1: %.12f"%(loss_v.item(),model.fn1.weight.grad.abs().mean().item()))
                log("loss_v: %6.4f, grad_v_conv1: %.12f"%(loss_v.item(),model.conv1.weight.grad.abs().mean().item()))
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

def benchmark_color(model,nn_color):
    color_dict={1:"bk",-1:"wt"}
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    for i in range(len(openings_unbl)):
        state_nn = FiveStone_NN(model)
        state_conv = FiveStoneState()
        state_nn = state_nn.track_hist(openings_unbl[i])
        state_conv = state_conv.track_hist(openings_unbl[i])
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
    l_print=[]
    for i in l_ans:
        if i==10000:
            l_print.append("w")
        elif i==-10000:
            l_print.append("l")
        elif i==0:
            l_print.append("d")
        else:
            l_print.append(",%s,"%(i))
    log("nn_color %s: %s"%(color_dict[nn_color]," ".join(l_print)))
    #win=len([0 for i in l_ans if i==10000])
    #loss=len([0 for i in l_ans if i==-10000])
    #wtf=[i for i in l_ans if (i!=-10000 and i!=10000)]
    #log("nn_color %s: %.1f%% %.1f%% %s"%(color_dict[nn_color],win/len(l_ans)*100,loss/len(l_ans)*100,wtf))

def benchmark(model):
    log("benchmarking...")
    benchmark_color(model,1)
    benchmark_color(model,-1)

def self_play():
    searcher1=abpruning(deep=1,n_killer=2)
    for i in range(len(openings)):
        state = FiveStoneState()
        state = state.track_hist(openings[i])
        print(state.board.type(torch.int8))
        input()
        while not state.isTerminal():
            searcher1.search(initialState=state)
            children=searcher1.children.items()
            best_action=max(children,key=lambda x: x[1]*state.currentPlayer)
            state=state.takeAction(best_action[0])
        print(state.board.type(torch.int8))
        print(state.getReward())

def test_eg1():
    state = FiveStone_NN(None)
    state = state.track_hist([(3,3),])
    print(state.board)
    print(state.getPossibleActions(printflag=True))

if __name__=="__main__":
    #test_eg1()
    #train_model()
    train_supervised()
    #self_play()