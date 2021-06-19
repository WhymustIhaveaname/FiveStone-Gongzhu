#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from os import stat
import time,sys,traceback,math,numpy
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE=sys.argv[0].split(".")
LOGFILE[-1]="log"
LOGFILE=".".join(LOGFILE)
def log(msg,l=1,end="\n",logfile=None,fileonly=False):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    #now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    now_str="%s %03d"%(time.strftime("%H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    if l<3:
        tempstr="%s [%s,%s:%d] %s%s"%(now_str,lstr,st.name,st.lineno,str(msg),end)
    else:
        tempstr="%s [%s,%s:%d] %s:\n%s%s"%(now_str,lstr,st.name,st.lineno,str(msg),traceback.format_exc(limit=5),end)
    if not fileonly:
        print(tempstr,end="")
    if l>=1 or fileonly:
        if logfile==None:
            logfile=LOGFILE
        with open(logfile,"a") as f:
            f.write(tempstr)

from copy import deepcopy
from MCTS.mcts import abpruning

import torch,re
import torch.nn as nn
import torch.nn.functional as F

def gen_kern_diag(kern_hori):
    return torch.stack([torch.diag(kern_hori[i][0][0]) for i in range(kern_hori.shape[0])]).view(kern_hori.shape[0],1,kern_hori.shape[3],kern_hori.shape[3])

kern_a_hori = torch.tensor([[0,1.,1,0],[1,0,0,0],[0,0,0,1]]).view(3,1,1,4)
kern_a_diag = gen_kern_diag(kern_a_hori)

kern_b_hori = torch.tensor([[[[1,1.0,1,0]]],[[[0,0,0,1]]]])
kern_b_diag = gen_kern_diag(kern_b_hori)

kern_c1_hori = torch.tensor([[[[1.,1,0,1,0]]],[[[0,0,1,0,0]]],[[[0,0,0,0,1]]]])
kern_c1_diag = gen_kern_diag(kern_c1_hori)

kern_c2_hori = torch.tensor([[[[0,1.,1,0,1]]],[[[1,0,0,0,0]]],[[[0,0,0,1,0]]]])
kern_c2_diag = gen_kern_diag(kern_c2_hori)

kern_d_hori = torch.tensor([[[[0,1.,1,1,0]]],[[[1,0,0,0,0]]],[[[0,0,0,0,1]]]])
kern_d_diag = gen_kern_diag(kern_d_hori)

kern_e_hori = torch.tensor([[0,1.,1,0,1,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]]).view(4,1,1,6)
kern_e_diag = gen_kern_diag(kern_e_hori)

kern_f_hori = torch.tensor([[1.,1,1,1,0],[0,0,0,0,1]]).view(2,1,1,5)
kern_f_diag = gen_kern_diag(kern_f_hori)

kern_g_hori = torch.tensor([[1.,1,0,1,1],[0,0,1,0,0]]).view(2,1,1,5)
kern_g_diag = gen_kern_diag(kern_g_hori)

kern_h_hori = torch.tensor([[1.,0,1,1,1],[0,1,0,0,0]]).view(2,1,1,5)
kern_h_diag = gen_kern_diag(kern_h_hori)

kern_i_hori = torch.tensor([[0,1.,1,1,1,0],[1,0,0,0,0,0],[0,0,0,0,0,1]]).view(3,1,1,6)
kern_i_diag = gen_kern_diag(kern_i_hori)

#kern_possact_5x5 = torch.tensor([[[[1.,1,1,1,1],[1,2,2,2,1],[1,2,-1024,2,1],[1,2,2,2,1],[1,1,1,1,1]]]])
kern_possact_3x3 = torch.tensor([[[[1.,1,1],[1,-1024,1],[1,1,1]]]])

class FiveStoneState():
    kernal_hori = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[1/5,1/5,1/5,1/5,1/5],[0,0,0,0,0],[0,0,0,0,0]]])
    kernal_diag = torch.tensor([[[1/5,0,0,0,0],[0,1/5,0,0,0],[0,0,1/5,0,0],[0,0,0,1/5,0],[0,0,0,0,1/5]]])
    kern_5 = torch.stack((kernal_hori, kernal_diag, kernal_hori.rot90(1,[1,2]), kernal_diag.rot90(1,[1,2])))

    def __init__(self,argv=None):
        self.board = torch.zeros(9,9)
        self.board[4,4] = 1.0
        self.currentPlayer = -1
        self.attack_factor = 0.8

    def reset(self):
        self.board = torch.zeros(9,9)
        self.board[4,4] = 1.0
        self.currentPlayer = -1

    def track_hist(self,hists,rot=0):
        for i in hists:
            if rot%4==0:
                ip=(4-i[1],4+i[0])
            elif rot%4==1:  # i[1]-->i[0]; i[0]--> -i[1]
                ip=(4-i[0],4-i[1])
            elif rot%4==2:
                ip=(4+i[1],4-i[0])
            elif rot%4==3:
                ip=(4+i[0],4+i[1])
            if self.board[ip[0]][ip[1]]!=0:
                log(self.board)
                raise Exception("Put stone on existed stone?")
            self.board[ip[0]][ip[1]] = self.currentPlayer
            self.currentPlayer *= -1

    def getCurrentPlayer(self):
        return self.currentPlayer

    def getPossibleActions(self,printflag=False):
        cv = F.conv2d(self.board.abs().view(1,1,9,9), kern_possact_3x3, padding=1)
        if printflag:
            print(cv)
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]

    def takeAction(self, action):
        newState = deepcopy(self)
        if newState.board[action[0]][action[1]]!=0:
            log(self.board)
            print(self.getPossibleActions(printflag=True))
            raise Exception("Put stone on existed stone?")
        newState.board[action[0]][action[1]] = self.currentPlayer
        newState.currentPlayer *= -1
        return newState

    def isTerminal(self):
        conv1 = F.conv2d(self.board.view(1,1,9,9), self.kern_5, padding=2)
        if conv1.max() >= 0.9 or conv1.min() <= -0.9:
            return True
        if self.board.abs().sum()==81:
            return True
        return False

    def getReward(self):
        conv1 = F.conv2d(self.board.view(1,1,9,9), self.kern_5, padding=2)
        if conv1.max() >= 0.9:
            return 10000
        elif conv1.min() <= -0.9:
            return -10000
        if self.board.sum()==81:
            return 0

        boards = torch.stack((self.board.view(1,9,9), self.board.view(1,9,9).rot90(1,[1,2]),
                              self.board.view(1,9,9).rot90(2,[1,2]), self.board.view(1,9,9).rot90(3,[1,2])))

        bk_reward=self.getReward_sub(boards,1)
        wt_reward=self.getReward_sub(boards,-1)
        #print(bk_reward,wt_reward)
        if self.currentPlayer == 1:
            return bk_reward-self.attack_factor*wt_reward
        else:
            return self.attack_factor*bk_reward-wt_reward

    def getReward_sub(self,boards,player):
        cv = F.conv2d(boards, kern_a_hori, padding=0)
        pt_a = ((cv[:,0]==player*2) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        cv = F.conv2d(boards, kern_a_diag, padding=0)
        pt_a += ((cv[:,0]==player*2) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_b_hori, padding=0)
        pt_b = ((cv[:,0]==player*3) & (cv[:,1]==0)).sum().item()
        cv = F.conv2d(boards, kern_b_diag, padding=0)
        pt_b += ((cv[:,0]==player*3) & (cv[:,1]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_c1_hori, padding=0)
        pt_c1 = ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        cv = F.conv2d(boards, kern_c1_diag, padding=0)
        pt_c1 += ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_c2_hori, padding=0)
        pt_c2 = ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        cv = F.conv2d(boards, kern_c2_diag, padding=0)
        pt_c2 += ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_d_hori, padding=0)
        pt_d = ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        cv = F.conv2d(boards, kern_d_diag, padding=0)
        pt_d += ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_e_hori, padding=0)
        pt_e = ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0) & (cv[:,3]==0)).sum().item()
        cv = F.conv2d(boards, kern_e_diag, padding=0)
        pt_e += ((cv[:,0]==player*3) & (cv[:,1]==0) & (cv[:,2]==0) & (cv[:,3]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_f_hori, padding=0)
        pt_f = ((cv[:,0]==player*4) & (cv[:,1]==0)).sum().item()
        cv = F.conv2d(boards, kern_f_diag, padding=0)
        pt_f += ((cv[:,0]==player*4) & (cv[:,1]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_g_hori, padding=0)
        pt_g = ((cv[:,0]==player*4) & (cv[:,1]==0)).sum().item()
        cv = F.conv2d(boards, kern_g_diag, padding=0)
        pt_g += ((cv[:,0]==player*4) & (cv[:,1]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_h_hori, padding=0)
        pt_h = ((cv[:,0]==player*4) & (cv[:,1]==0)).sum().item()
        cv = F.conv2d(boards, kern_h_diag, padding=0)
        pt_h += ((cv[:,0]==player*4) & (cv[:,1]==0)).sum().item()
        del cv

        cv = F.conv2d(boards, kern_i_hori, padding=0)
        pt_i = ((cv[:,0]==player*4) & (cv[:,1]==0) & (cv[:,2]==0)).sum().item()
        cv = F.conv2d(boards, kern_i_diag, padding=0)
        pt_i += ((cv[:,0]==player*4) & (cv[:,1]==0)& (cv[:,2]==0)).sum().item()
        del cv

        # print("pt_a  | ", pt_a)
        # print("pt_b  | ", pt_b)
        # print("pt_c1 | ", pt_c1)
        # print("pt_c2 | ", pt_c2)
        # print("pt_d  | ", pt_d)
        # print("pt_e  | ", pt_e)
        # print("pt_f  | ", pt_f)
        # print("pt_g  | ", pt_g)
        # print("pt_h  | ", pt_h)
        # print("pt_i  | ", pt_i)

        return (5*pt_a + 50*pt_b + 20*pt_c1 + 10*pt_c2 + 200*pt_d +\
               450*pt_e + 450*pt_f + 245*pt_g + 100*pt_h + 4100*pt_i)/10

def pretty_board(gamestate):
    d_stone={1:"\u25cf",-1:"\u25cb",0:" "} #"\u25cb" "\u25cf"
    li=[]
    for i,r in enumerate(gamestate.board):
        lj="|".join([d_stone[j.item()] for j in r])
        li.append("%2d|%s|"%(4-i,lj))
    li.append("           0 1 2 3 4 ")
    li="\n".join(li)
    log("\n%s"%(li),l=0)
    return li

def play_tui(human_color=-1,deep=3):
    searcher=abpruning(deep=deep,n_killer=4)
    state=FiveStoneState()
    state.reset()
    while not state.isTerminal():
        if state.currentPlayer==human_color:
            get_tui_input(state)
        else:
            searcher.counter=0
            log("searching...")
            searcher.search(initialState=state)
            log("searched %d cases"%(searcher.counter))
            best_action=max(searcher.children.items(),key=lambda x: x[1]*state.currentPlayer)
            log(best_action)
            state=state.takeAction(best_action[0])

def get_tui_input(state):
    pretty_board(state)
    while True:
        istr=input("your action: ")
        r=re.match("[\\-0-9]+([,.])[\\-0-9]+",istr)
        if r:
            myaction=tuple([int(i) for i in istr.split(r.group(1))])
            try:
                state=state.track_hist([myaction])
            except:
                log("take action failed",l=3)
            else:
                return state
        else:
            log("input format error!")

def test_rot():
    state = FiveStoneState()
    for i in range(4):
        state.reset()
        state.track_hist([(1,1),(2,-2)],rot=i)
        pretty_board(state)

if __name__=="__main__":
    #test_rot()
    play_tui(human_color=-1,deep=3)