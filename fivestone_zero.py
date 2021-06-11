#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch,re,time,copy,itertools,pickle,tempfile,os,random
import torch.nn as nn
import torch.nn.functional as F
#from multiprocessing import Process,Queue
from torch.multiprocessing import Process,Queue

from MCTS.mcts import abpruning
from fivestone_conv import log, pretty_board

torch.set_default_dtype(torch.float16)
from net_topo import PV_resnet, FiveStone_CNN
from benchmark_utils import open_bl,benchmark,vs_noth

PARA_DICT={ "ACTION_NUM":100, "POSSACT_RAD":1, "AB_DEEP":1, "SOFTK":4,
            "LOSS_P_WT":1.0, "LOSS_P_WT_RATIO": 0.5, "STDP_WT": 5.0, "BATCH_SIZE":64,
            "FINAL_LEN": 4, "FINAL_BIAS": 0.5, "UID_ROT": 4, "SHIFT_MAX":3}

class FiveStone_ZERO(FiveStone_CNN):
    def reset(self):
        self.board = torch.zeros(9,9,device="cuda",dtype=torch.float16)
        self.board[4,4] = 1.0
        self.currentPlayer = -1

    def getPossibleActions(self):
        """cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
        #cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]"""
        input_data=self.gen_input().view((1,3,9,9))
        policy,value=self.model(input_data)
        policy=policy.view(9,9)
        if self.radius in (1,2):
            if self.radius==1:
                cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
            elif self.radius==2:
                cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if cv[0,0,i,j]>0]
        elif self.radius>2:
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if self.board[i,j]==0]

        if len(lkv)<self.target_num:
            return [k for k,v in lkv]
        else:
            lkv.sort(key=lambda x:x[1],reverse=True)
            return [lkv[i][0] for i in range(self.target_num)]

def push_data(datalist,in_mat,best_value,target_p,legal_mask,rots=[0,],flip=True):
    if flip:
        in_mts=[in_mat,in_mat.flip(2)]
        bst_vals=[best_value,best_value]
        tg_ps=[target_p,target_p.flip(1)]
        lg_msks=[legal_mask,legal_mask.flip(1)]
    else:
        in_mts=[in_mat,]
        bst_vals=[best_value,]
        tg_ps=[target_p,]
        lg_msks=[legal_mask,]

    for rot,i in itertools.product(rots,range(len(in_mts))):
        this_data=[ in_mts[i].rot90(rot,[1,2]),
                    bst_vals[i],
                    tg_ps[i].rot90(rot,[0,1]).reshape(81),
                    lg_msks[i].rot90(rot,[0,1]).reshape(81)]
        datalist.append(this_data)
        #print(this_data[0]);input()

def balance_bkwt(datalist):
    for i in range(len(datalist)):
        in_mat,best_val,target_p,legal_msk=datalist[i]
        in_mat_s=in_mat[(1,0,2),:,:]
        in_mat_s[2,:,:]=-1*in_mat_s[2,:,:]
        this_data=[in_mat_s,-1*best_val,target_p,legal_msk]
        datalist.append(this_data)

def gen_data(model,num_games,randseed,data_q,PARA_DICT):
    train_datas=[]
    searcher=abpruning(deep=PARA_DICT["AB_DEEP"])
    state = FiveStone_ZERO(model)
    state.target_num=PARA_DICT["ACTION_NUM"]
    state.radius=PARA_DICT["POSSACT_RAD"]
    random.seed(str(randseed))
    for i in range(num_games):
        state.reset()
        open_num=random.randint(0,len(open_bl)-1)
        rot_num=random.randint(0,3)
        shifts=[random.randint(-PARA_DICT["SHIFT_MAX"],PARA_DICT["SHIFT_MAX"]) for i in range(2)]
        state.track_hist(open_bl[open_num],rot=rot_num)
        state.board=state.board.roll(shifts=shifts,dims=(0,1))
        #log("%d, rot %d, %s"%(open_num,rot_num,shifts))
        #pretty_board(state);input()
        dlen_1=len(train_datas)
        while not state.isTerminal():
            #in_mat and best_value
            in_mat=state.gen_input()
            searcher.search(initialState=state)
            if state.currentPlayer==1:
                best_action,best_value=max(searcher.children.items(),key=lambda x:x[1])
            else:
                best_action,best_value=min(searcher.children.items(),key=lambda x:x[1])

            # target_p and legal_mask
            lkv=[(k,v) for k,v in searcher.children.items()]
            lv=torch.tensor([v for k,v in lkv],dtype=torch.float32)*state.currentPlayer
            lv=F.softmax(lv*PARA_DICT["SOFTK"],dim=0)
            target_p=torch.zeros(9,9,device="cuda",dtype=torch.float32)
            for j in range(len(lkv)):
                target_p[lkv[j][0]]=lv[j]
            legal_mask=(state.board==0).float()

            push_data(train_datas,in_mat.float(),best_value.float(),target_p,legal_mask,rots=range(PARA_DICT["UID_ROT"]),flip=True)

            #r=torch.multinomial(lv,1)
            #state=state.takeAction(lkv[r][0])
            state=state.takeAction(best_action)
        #pretty_board(state);input()
        dlen_2=len(train_datas)
        result=state.getReward().item()

        fin_len=PARA_DICT["FINAL_LEN"]*2*PARA_DICT["UID_ROT"]
        for j in range(min(dlen_2-dlen_1,fin_len)):
            wt=max(0,j/fin_len-PARA_DICT["FINAL_BIAS"])
            train_datas[-j-1][1]=train_datas[-j-1][1]*wt+result*(1-wt)
    return train_datas

def gen_data_sub(model,num_games,randseed,data_q,PARA_DICT):
    try:
        datalist=gen_data(model,num_games,randseed,data_q,PARA_DICT)
    except:
        log("",l=3)
        datalist=[]
    fd,fname=tempfile.mkstemp(suffix='.fivestone.tmp',prefix='',dir='/tmp')
    with open(fd,"wb") as f:
        pickle.dump(datalist,f)
    #data_q.put((fd,fname,tuple(lre)))
    data_q.put((fd,fname))

def gen_data_multithread(model,num_games):
    data_q=Queue()
    plist=[]
    t=int(time.time())
    for i in range(3):
        plist.append(Process(target=gen_data_sub,args=(copy.deepcopy(model).eval().half(),num_games,i+t,data_q,PARA_DICT)))
        plist[-1].start()
    rlist=[]
    for p in plist:
        p.join()
        fd,fname=data_q.get(False)
        with open(fname,"rb") as f:
            rlist+=pickle.load(f)
        os.unlink(fname)
    return rlist

def train(model):
    torch.set_default_dtype(torch.float32)
    optim = torch.optim.Adam(model.parameters(),lr=0.0005,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optim: %s"%(optim.__dict__['defaults'],))
    log("PARA_DICT: %s"%(PARA_DICT))

    for epoch in range(1200):
        if epoch<3 or (epoch<40 and epoch%5==0) or epoch%10==0:
            print_flag=True
        else:
            print_flag=False

        if print_flag and epoch%20==0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)

        if print_flag and epoch%5==0:
            state_nn=FiveStone_ZERO(copy.deepcopy(model).eval().half())
            state_nn.target_num=100
            state_nn.radius=2
            vs_noth(state_nn,epoch)
            benchmark(state_nn,epoch)

        #train_datas = gen_data(copy.deepcopy(model).eval().half(),5,time.time(),None,PARA_DICT)
        train_datas = gen_data_multithread(model,10)
        balance_bkwt(train_datas)
        trainloader = torch.utils.data.DataLoader(train_datas,batch_size=PARA_DICT["BATCH_SIZE"],shuffle=True,drop_last=True)

        if print_flag:
            log("epoch %d with %d datas"%(epoch,len(train_datas)))
            for batch in trainloader:
                policy,value = model(batch[0])
                log("sampled output_value, output_policy:\n%s\n%s"%(" ".join(["%.2f"%(i[0]) for i in value][0:32]),
                                                                    " ".join(["%.2f"%(i) for i in (policy*batch[3]).std(dim=1)][0:32]) ))

                #log_p = F.log_softmax(policy*batch[3],dim=1)
                softmax_p = (policy*batch[3]).softmax(dim=1)
                loss_p = F.kl_div(softmax_p.log(),batch[2],reduction="batchmean")
                optim.zero_grad()
                loss_p.backward(retain_graph=True)
                grad_p=model.conv1.weight.grad.abs().mean().item()

                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                optim.zero_grad()
                loss_v.backward(retain_graph=True)
                grad_v=model.conv1.weight.grad.abs().mean().item()

                lp_std=softmax_p.std(dim=1).sum()/PARA_DICT["BATCH_SIZE"]
                optim.zero_grad()
                lp_std.backward(retain_graph=True)
                grad_pv=model.conv1.weight.grad.abs().mean().item()

                log("loss_v: %6.4f, grad_conv1: %.8f"%(loss_v.item(),grad_v))
                log("loss_p: %6.4f, grad_conv1: %.8f"%(loss_p.item(),grad_p))
                log("lp_std: %6.4f, grad_conv1: %.8f"%(lp_std.item(),grad_pv))
                PARA_DICT["LOSS_P_WT"]+=0.5*(PARA_DICT["LOSS_P_WT_RATIO"]*grad_v/grad_p-PARA_DICT["LOSS_P_WT"])
                log("update LOSS_P_WT to %.2f"%(PARA_DICT["LOSS_P_WT"]))
                break

        for age in range(3):
            running_loss = 0.0
            ax=0
            for batch in trainloader:
                ax+=1
                policy,value = model(batch[0])
                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                #log_p = F.log_softmax(policy*batch[3],dim=1)
                softmax_p = (policy*batch[3]).softmax(dim=1)
                loss_p = F.kl_div(softmax_p.log(),batch[2],reduction="batchmean")
                lp_std = softmax_p.std(dim=1).sum()/PARA_DICT["BATCH_SIZE"]

                optim.zero_grad()
                loss=loss_v+PARA_DICT["LOSS_P_WT"]*(loss_p-lp_std*PARA_DICT["STDP_WT"])
                loss.backward()
                optim.step()
                running_loss += loss_v.item()
            if print_flag and (age<3 or (age+1)%5==0):
                log("    age %2d: %.6f"%(age,running_loss/ax))

def test_must_win(model):
    state = FiveStone_ZERO(model)
    for i in range(3):
        state.board[4+i,4+i]=-1
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

def play_tui(model,human_color=-1):
    searcher=abpruning(deep=3,n_killer=2,gameinf=1024)
    state = FiveStone_ZERO(model.eval().half())
    state.target_num=20
    state.radius=2
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

def test_push_data():
    state = FiveStone_ZERO(model)
    state.track_hist(open_bl[3])
    mat_in=state.gen_input()
    best_value=torch.tensor([3.5],device="cuda")
    target_p=torch.rand(9,9,device="cuda")
    legal_mask=(state.board==0).type(torch.cuda.FloatTensor)

    push_data([],mat_in,best_value,target_p,legal_mask,rots=[0,1])


def test_vs_noth(model):
    """passed"""
    state=FiveStone_ZERO(model.eval().half())
    vs_noth(state,-1)

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_dtype(torch.float32)
    model=PV_resnet().cuda()
    start_file=None
    #start_file="./logs/6_1/PV_resnet-16-15857234-180.pkl"
    #start_file="./logs/8/PV_resnet-16-15857234-40.pkl"
    #start_file="./logs/16/PV_resnet-16-15859346-420.pkl"
    if start_file!=None:
        model.load_state_dict(torch.load(start_file,map_location="cuda"))
        log("load from %s"%(start_file))
    else:
        log("init model %s"%(model))
    #test_must_win(model)
    #play_tui(model)
    #test_push_data()
    #test_vs_noth(model)
    train(model)