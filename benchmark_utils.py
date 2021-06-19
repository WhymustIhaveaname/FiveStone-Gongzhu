#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MCTS.mcts import abpruning
from fivestone_conv import log, FiveStoneState,pretty_board
import itertools

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
# taken from http://chiuinan.github.io/game/game/intro/ch/c41/ms52/rule/5ch_rule.htm
open_bl=[   [(0,1),(0,2)], [(0,1),(1,2)], [(0,1),(2,2)], [(0,1),(1,1)],
            [(0,1),(2,1)], [(0,1),(1,0)], [(0,1),(2,0)], [(0,1),(0,-1)],
            [(0,1),(1,-1)], [(0,1),(2,-1)], [(0,1),(0,-2)], [(0,1),(1,-2)],
            [(0,1),(2,-2)],
            [(1,1),(2,2)], [(1,1),(2,1)], [(1,1),(2,0)], [(1,1),(2,-1)],
            [(1,1),(2,-2)], [(1,1),(1,0)], [(1,1),(1,-1)], [(1,1),(1,-2)],
            [(1,1),(0,-1)], [(1,1),(0,-2)], [(1,1),(-1,-1)], [(1,1),(-1,-2)],
            [(1,1),(-2,-2)],]

def vs_noth(state_nn,epoch):
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    for i1,i2,nn_color in itertools.product(range(2,7),range(2,7),[-1,1]):
        state_nn.reset()
        state_nn.board[4,4]=0
        state_nn.board[i1,i2]=nn_color
        #pretty_board(state_nn);input()
        while not state_nn.isTerminal():
            state_nn.currentPlayer=nn_color
            state_nn=state_nn.takeAction(state_nn.policy_choice_best())
        l_ans.append(state_nn.board.abs().sum().item())
        #log(l_ans)
        #pretty_board(state_nn);input()
    log("epoch %d avg win steps: %d/%d=%.1f"%(epoch,sum(l_ans),len(l_ans),sum(l_ans)/len(l_ans)))

"""def vs_rand(state_nn,epoch):
    searcher=abpruning(deep=1,n_killer=2)
    l_ans=[]
    l_loss=[]
    for nn_color,i in itertools.product([1,-1],range(50)):
        state_nn.reset()
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
    log("epoch %d avg win steps: %d/%d=%.1f, %.1f%%"%(epoch,sum(l_ans),len(l_ans),sum(l_ans)/len(l_ans),win_rate))"""

def benchmark_color(state_nn,nn_color,openings,epoch):
    searcher=abpruning(deep=1)
    color_dict={1:"bk",-1:"wt"}
    result_dict={10000:"w",-10000:"l",0:"d"}
    l_ans=""
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
        l_ans+=result_dict.get(result,"?")
    log("epoch %d nn_color %s: %.1f%%\n%s"%(epoch,color_dict[nn_color],l_ans.count("w")/len(l_ans)*100,l_ans))
    return l_ans

def benchmark(state_nn,epoch):
    """l_bk=benchmark_color(model,1,open_unbl_black,epoch)
    l_wt=benchmark_color(model,-1,open_unbl_white,epoch)
    if l_bk.startswith("w w"):
        benchmark_color(model,1,open_bl,epoch)
    if l_wt.startswith("w w"):
        benchmark_color(model,-1,open_bl,epoch)"""
    benchmark_color(state_nn,1,open_bl,epoch)
    benchmark_color(state_nn,-1,open_bl,epoch)

def test_open():
    """passed"""
    state_conv = FiveStoneState()
    for i in range(len(open_bl)):
        state_conv.reset()
        state_conv.track_hist(open_bl[i],rot=0)
        pretty_board(state_conv)
        input()

if __name__=="__main__":
    test_open()