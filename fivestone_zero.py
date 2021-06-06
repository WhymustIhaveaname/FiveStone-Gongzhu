#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MCTS.mcts import abpruning
from fivestone_conv import log, FiveStoneState
from net_topo import PVnet, FiveStone_NN

import torch,re,time,copy
import torch.nn as nn
import torch.nn.functional as F

from supervised import openings_unbl,openings_bl,select_by_prob,benchmark

def gen_data(model):
    softk=0.5
    train_datas=[]
    searcher=abpruning(deep=1,n_killer=2)
    for i in range(len(openings_bl)):
        state = FiveStone_NN(model)
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
            target_v=best_value
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

def train():
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
            train_datas += gen_data(model)
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
                loss=loss_v+loss_p*2
                loss.backward()
                optim.step()
                running_loss += loss.item()
            if age<3 or age%10==0:
                log("    age %2d: %.6f"%(age,running_loss/len(train_datas)))

if __name__=="__main__":
    #test_eg1()
    #train_model()
    train()
    #self_play()