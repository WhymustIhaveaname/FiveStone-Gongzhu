#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch,re,time,copy,itertools,pickle,tempfile,os,random
import torch.nn as nn
import torch.nn.functional as F
#from multiprocessing import Process,Queue
from torch.multiprocessing import Process,Queue

from MCTS.mcts import abpruning
from fivestone_conv import log, pretty_board
from net_topo import PV_resnet, FiveStone_CNN#, PVnet_cnn
from fivestone_cnn import open_bl,benchmark_color,benchmark,vs_rand,vs_noth

PARA_DICT={ "ACTION_NUM":16, "AB_DEEP":1, "POSSACT_RAD":1,
            "BATCH_SIZE":64, "LOSS_P_WT":4.0, "SOFTK":4/FiveStone_CNN.WIN_REWARD,
            "VID_THRES_BK": 0.2*FiveStone_CNN.WIN_REWARD, "VID_THRES_WT": -0.2*FiveStone_CNN.WIN_REWARD,
            "VID_LR":0.1, "UID_ROT":4, "VID_DUP_CONST": 8,
            "FINAL_LEN": 6, "FINAL_BIAS": 0.5}

class FiveStone_ZERO(FiveStone_CNN):
    def getPossibleActions(self,target_num=PARA_DICT["ACTION_NUM"]):
        """cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
        #cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
        l_temp=[(cv[0,0,i,j].item(),(i,j)) for i in range(9) for j in range(9) if cv[0,0,i,j]>0]
        l_temp.sort(key=lambda x:-1*x[0])
        return [i[1] for i in l_temp]"""
        input_data=self.gen_input().view((1,3,9,9))
        policy,value=self.model(input_data)
        policy=policy.view(9,9)
        if PARA_DICT["POSSACT_RAD"] in (1,2):
            if PARA_DICT["POSSACT_RAD"]==1:
                cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_3x3, padding=1)
            elif PARA_DICT["POSSACT_RAD"]==2:
                cv = F.conv2d(self.board.abs().view(1,1,9,9), self.kern_possact_5x5, padding=2)
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if cv[0,0,i,j]>0]
        elif PARA_DICT["POSSACT_RAD"]>2:
            lkv=[((i,j),policy[i,j].item()) for i,j in itertools.product(range(9),range(9)) if self.board[i,j]==0]

        if len(lkv)<target_num:
            return [k for k,v in lkv]
        else:
            lkv.sort(key=lambda x:x[1],reverse=True)
            return [lkv[i][0] for i in range(target_num)]

def push_data(datalist,in_mat,best_value,target_p,legal_mask,rots=[0,],flip=True):
    if flip:
        in_mts=[in_mat,in_mat.flip(2)]
        bst_vals=[best_value,best_value]
        tg_ps=[target_p,target_p.flip(1)]
        lg_msks=[legal_mask,legal_mask.flip(1)]
        lsf=2
    else:
        in_mts=[in_mat,]
        bst_vals=[best_value,]
        tg_ps=[target_p,]
        lg_msks=[legal_mask,]
        lsf=1

    for rot,i in itertools.product(rots,range(lsf)):
        this_data=[ in_mts[i].rot90(rot,[1,2]),
                    bst_vals[i],
                    tg_ps[i].rot90(rot,[0,1]).reshape(81),
                    lg_msks[i].rot90(rot,[0,1]).reshape(81)]
        datalist.append(this_data)

def balance_bkwt(datalist):
    for i in range(len(datalist)):
        in_mat,best_val,target_p,legal_msk=datalist[i]
        in_mat_s=in_mat[(1,0,2),:,:]
        in_mat_s[2,:,:]=-1*in_mat_s[2,:,:]
        this_data=[in_mat_s,best_val*-1,target_p,legal_msk]
        datalist.append(this_data)

def gen_data(model,num_games,rot_bias,data_q,PARA_DICT):
    train_datas=[]
    searcher=abpruning(deep=PARA_DICT["AB_DEEP"])
    state = FiveStone_ZERO(model)
    lre=[0,0]
    for i in range(num_games):
        state.reset()
        state.track_hist(open_bl[i%len(open_bl)],rot=i//len(open_bl)+rot_bias)
        state.board.roll(shifts=(random.randint(-2,2),random.randint(-2,2)),dims=(0,1))
        vidata=None
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
            lv=torch.tensor([v for k,v in lkv])*state.currentPlayer*PARA_DICT["SOFTK"]
            lv=F.softmax(lv,dim=0)
            target_p=torch.zeros(9,9,device="cuda")
            for j in range(len(lkv)):
                target_p[lkv[j][0]]=lv[j]
            legal_mask=(state.board==0).type(torch.cuda.FloatTensor)

            push_data(train_datas,in_mat,best_value,target_p,legal_mask,rots=range(PARA_DICT["UID_ROT"]),flip=True)

            if vidata==None\
                and (   FiveStone_CNN.WIN_REWARD>best_value>PARA_DICT["VID_THRES_BK"] or
                       -FiveStone_CNN.WIN_REWARD<best_value<PARA_DICT["VID_THRES_WT"] ):
                vidata=len(train_datas)-1
                #log("set vidata to %d"%(vidata))
            elif vidata!=None\
                and ( (train_datas[vidata][1]>0 and best_value<PARA_DICT["VID_THRES_BK"]) or\
                      (train_datas[vidata][1]<0 and best_value>PARA_DICT["VID_THRES_WT"]) ):
                vidata=None

            #r=torch.multinomial(lv,1)
            #state=state.takeAction(lkv[r][0])
            state=state.takeAction(best_action)
        #pretty_board(state)
        dlen_2=len(train_datas)
        result=state.getReward().item()

        vid_flag=False
        if vidata!=None and train_datas[vidata][1]*result>0:
            vid_flag=True
            in_mat,best_value,target_p,legal_mask=train_datas[vidata]
            vid_dup=PARA_DICT["VID_DUP_CONST"]
            if best_value>0:
                lre[0]+=(best_value.item()-PARA_DICT["VID_THRES_BK"])
            else:
                lre[1]+=(best_value.item()-PARA_DICT["VID_THRES_WT"])
            num_1=in_mat[0].sum().item()+in_mat[1].sum().item()
            num_2=state.board.abs().sum().item()
            log("correct predict at %d/%d: %.4f! dup %d, thres %.2f, %.2f"\
                %(num_1,num_2,best_value,vid_dup,PARA_DICT["VID_THRES_BK"],PARA_DICT["VID_THRES_WT"]))

        #log("final_result: %.2f, %d, %d, %d"%(result,state.board.abs().sum().item(),dlen_1,dlen_2))
        #log(["%.2f"%(train_datas[i][1]) for i in range(dlen_1,dlen_2)])
        fin_len=PARA_DICT["FINAL_LEN"]*2*PARA_DICT["UID_ROT"]
        for j in range(min(dlen_2-dlen_1,fin_len)):
            wt=max(0,j/fin_len-PARA_DICT["FINAL_BIAS"])
            train_datas[-j-1][1]=train_datas[-j-1][1]*wt+result*(1-wt)
        #log(["%.2f"%(train_datas[i][1]) for i in range(dlen_1,dlen_2)]);input()

        if vid_flag and vid_dup>0:
            push_data(train_datas,in_mat,best_value,target_p.view(9,9),legal_mask.view(9,9),rots=range(vid_dup*2*PARA_DICT["UID_ROT"]),flip=True)
            del vid_flag

    fd,fname=tempfile.mkstemp(suffix='.fivestone.tmp',prefix='',dir='/tmp')
    with open(fd,"wb") as f:
        pickle.dump(train_datas,f)
    data_q.put((fd,fname,tuple(lre)))

def gen_data_multithread(model):
    data_q=Queue()
    plist=[]
    for i in range(3):
        plist.append(Process(target=gen_data,args=(copy.deepcopy(model),10,i,data_q,PARA_DICT)))
        plist[-1].start()
    rlist=[]
    for p in plist:
        p.join()
        fd,fname,lre=data_q.get(False)
        with open(fname,"rb") as f:
            rlist+=pickle.load(f)
        os.unlink(fname)
        PARA_DICT["VID_THRES_BK"]+=lre[0]*PARA_DICT["VID_LR"]
        PARA_DICT["VID_THRES_WT"]+=lre[1]*PARA_DICT["VID_LR"]
    return rlist

def train(model):
    optim = torch.optim.Adam(model.parameters(),lr=0.0005,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log(model)
    log("optim: %s"%(optim.__dict__['defaults'],))
    log("PARA_DICT: %s"%(PARA_DICT))

    for epoch in range(1200):
        if (epoch<60 and epoch%5==0) or epoch%10==0:
            save_name='./model/%s-%s-%s-%d.pkl'%(model.__class__.__name__,model.num_layers(),model.num_paras(),epoch)
            torch.save(model.state_dict(),save_name)
            vs_noth(model,epoch)
            vs_rand(model,epoch)
            benchmark(model,epoch)

        #train_datas = gen_data(model,20,0,None,PARA_DICT)
        train_datas = gen_data_multithread(model)
        balance_bkwt(train_datas)
        trainloader = torch.utils.data.DataLoader(train_datas,batch_size=PARA_DICT["BATCH_SIZE"],shuffle=True,drop_last=True)

        if epoch<3 or (epoch<40 and epoch%5==0) or epoch%20==0:
            print_flag=True
        else:
            print_flag=False

        if print_flag:
            log("epoch %d with %d datas"%(epoch,len(train_datas)))
            for batch in trainloader:
                log("sampled value, policy:\n%s\n%s"%(", ".join(["%.1f"%(batch[1][i,0]) for i in range(20)]),
                                                      ", ".join(["%.2f"%(i.item()) for i in batch[2][0,:] if i!=0]) ))
                policy,value = model(batch[0])
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")
                optim.zero_grad()
                loss_p.backward(retain_graph=True)
                grad_p=model.conv1.weight.grad.abs().mean().item()
                log("loss_p: %6.4f, grad_p_conv1: %.8f"%(loss_p.item()/PARA_DICT["BATCH_SIZE"],grad_p))

                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                optim.zero_grad()
                loss_v.backward(retain_graph=True)
                grad_v=model.conv1.weight.grad.abs().mean().item()
                log("loss_v: %6.4f, grad_p_conv1: %.8f"%(loss_v.item()/PARA_DICT["BATCH_SIZE"],model.conv1.weight.grad.abs().mean().item()))
                PARA_DICT["LOSS_P_WT"]+=0.3*(0.5*grad_v/grad_p-PARA_DICT["LOSS_P_WT"])
                log("update LOSS_P_WT to %.2f"%(PARA_DICT["LOSS_P_WT"]))
                break

        for age in range(3):
            running_loss = 0.0
            ax=0
            for batch in trainloader:
                ax+=1
                policy,value = model(batch[0])
                loss_v = F.mse_loss(batch[1], value, reduction='mean').sqrt()
                log_p = F.log_softmax(policy*batch[3],dim=1)
                loss_p = F.kl_div(log_p,batch[2],reduction="batchmean")

                optim.zero_grad()
                loss=loss_v+loss_p*PARA_DICT["LOSS_P_WT"]
                loss.backward()
                optim.step()
                running_loss += loss.item()
            if print_flag and (age<3 or (age+1)%5==0):
                log("    age %2d: %.6f"%(age,running_loss/(PARA_DICT["BATCH_SIZE"]*ax)))

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

def test_push_data():
    state = FiveStone_ZERO(model)
    state.track_hist(open_bl[3])
    mat_in=state.gen_input()
    best_value=torch.tensor([3.5],device="cuda")
    target_p=torch.rand(9,9,device="cuda")
    legal_mask=(state.board==0).type(torch.cuda.FloatTensor)

    push_data([],mat_in,best_value,target_p,legal_mask,rots=[0,1])

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    model = PV_resnet().cuda()
    #start_file="./logs/6_1/PV_resnet-16-15857234-180.pkl"
    #model.load_state_dict(torch.load(start_file,map_location="cuda"))
    #log("load from %s"%(start_file))
    #test_must_win(model)
    #test_push_data()
    train(model)