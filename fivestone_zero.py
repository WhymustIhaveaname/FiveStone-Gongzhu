#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch,re,time,copy,itertools
import torch.nn as nn
import torch.nn.functional as F

from MCTS.mcts import abpruning
from fivestone_conv import log, FiveStoneState, pretty_board
from net_topo import PVnet_cnn, FiveStone_CNN
from fivestone_cnn import open_bl,select_by_prob,benchmark_color,benchmark,vs_rand,vs_noth

SOFTK=3

def gen_data(model,duplicate):
    train_datas=[]
    searcher=abpruning(deep=1,n_killer=2)
    state = FiveStone_CNN(model)
    for i1,i2 in itertools.product(range(len(open_bl)),range(1024,1024+duplicate)):
        state.reset()
        state = state.track_hist(open_bl[i1])
        #pretty_board(state)
        while not state.isTerminal():
            searcher.search(initialState=state)
            if state.currentPlayer==1:
                best_value=max(searcher.children.values())
            elif state.currentPlayer==-1:
                best_value=min(searcher.children.values())
            else:
                log("what's your problem?!",l=2)

            input_data=state.gen_input()
            target_v=best_value

            lkv=[(k,v) for k,v in searcher.children.items()]
            lv=torch.tensor([v for k,v in lkv])*state.currentPlayer*SOFTK
            lv=F.softmax(lv,dim=0)
            target_p=torch.zeros(9,9).cuda()
            for j in range(len(lkv)):
                target_p[lkv[j][0]]=lv[j]
            legal_mask=(state.board==0).type(torch.cuda.FloatTensor)
            train_datas.append((input_data,target_v,target_p.view(-1),legal_mask.view(-1)))

            next_action=select_by_prob(searcher.children,state.currentPlayer,SOFTK)
            state=state.takeAction(next_action)
        #pretty_board(state);input()
    return train_datas

def train():
    model = PVnet_cnn().cuda()
    optim = torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    loss_p_wt = 0.3
    log(model)
    log("loss_p_wt: %.1f, optim: %s"%(loss_p_wt,optim.__dict__['defaults'],))

    for epoch in range(400):
        if epoch%5==0:# and epoch>0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)
            vs_noth(model,epoch)
            vs_rand(model,epoch)
            benchmark(model,epoch)

        train_datas = gen_data(model,5)
        trainloader = torch.utils.data.DataLoader(train_datas,batch_size=32,shuffle=True,drop_last=True)

        if epoch<5 or epoch%5==0:
            print_flag=True
        else:
            print_flag=False

        if print_flag:
            log("epoch %d with %d datas"%(epoch,len(train_datas)))
            for batch in trainloader:
                policy,value = model(batch[0])
                log(["%.2f"%(i) for i in value])
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")
                optim.zero_grad()
                loss_p.backward(retain_graph=True)
                log("loss_p: %6.4f, grad_p_fn1: %.8f, grad_p_conv1: %.8f"%(loss_p.item(),
                    model.fn1.weight.grad.abs().mean().item(),
                    model.conv1.weight.grad.abs().mean().item()))

                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                optim.zero_grad()
                loss_v.backward(retain_graph=True)
                log("loss_v: %6.4f, grad_v_fn1: %.8f, grad_p_conv1: %.8f"%(loss_v.item(),
                    model.fn1.weight.grad.abs().mean().item(),
                    model.conv1.weight.grad.abs().mean().item()))
                break

        for age in range(5):
            running_loss = 0.0
            for batch in trainloader:
                policy,value = model(batch[0])
                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")
                optim.zero_grad()
                loss=loss_v+loss_p*1.0
                loss.backward()
                optim.step()
                running_loss += loss.item()
            if print_flag and (age<3 or (age+1)%5==0):
                log("    age %2d: %.6f"%(age,running_loss/len(train_datas)))

if __name__=="__main__":
    train()