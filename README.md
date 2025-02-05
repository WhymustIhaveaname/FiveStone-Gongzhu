# FiveStone-Gongzhu

We solved Gomoku (Five in a Row) using both alpha-beta pruning and AlphaZero's architecture.
For simplicity, this Gomoku applies no professional rules ([Renju's 33, 44 and overlines rules applied to black](https://en.wikipedia.org/wiki/Gomoku#Specific_variations)).
We show that a 16 layer convolutional network can reach a higher level than man-made evaluation only by self-learning.
We investigate some statistical properties of our AIs.
This project takes advantage of [Gongzhu-Society](https://github.com/Gongzhu-Society/MCTS)'s MCTS and alpha-beta pruning package.

我们用两种方法 —— alpha-beta 剪枝和类 AlphaZero 方法解决了五子棋的问题。
出于演示的目的，我们没有采用任何禁手规则。
一个 16 层的卷积神经网络通过自学就能达到比 alpha-beta 剪枝中人为指定的 evaluation 更高的水平。
我们对我们 AI 的行为进行了一些统计。
本项目使用了 [Gongzhu-Society](https://github.com/Gongzhu-Society/MCTS) 的蒙特卡洛树搜索和 alpha-beta 剪枝包。

## Features

* Generate MANY train data from one by rotation, flip, and translation.
* Beautiful terminal output.
* Visualization of convolutional layer and interesting (also unsolved) phenomena in visualization.

## Environment Setup and Get Started

```
git clone https://github.com/WhymustIhaveaname/FiveStone-Gongzhu.git
cd FiveStone-Gongzhu
git clone https://github.com/Gongzhu-Society/MCTS.git
```

* Use the following command to play with alpha-beta pruning AI.
```
./fivestone_conv.py
```
Input like `-1,1` to put a stone.
To change your color or set search deep, please modify the codes.

* Use the following command to play with neural network AI.
```
./fivestone_zero.py
```

* The following screenshot is a typical game against fivestone_zero.
I am ○ while AI ●.
He decided to play `(4, 7)` (this is matrix index) after 886 searches.
Then the board is printed. Then I input my next step, `2,-1` (in XY coordinates).
Clever friends may find that, in the current situation, my defeat is doomed (AI will get to a three-three/four-three after two rounds).
And my AI indeed did that!

<div align=center>
   <img width="70%" src="https://github.com/WhymustIhaveaname/FiveStone-Gongzhu/blob/main/figures/typical_game.png?raw=true"/>
</div>

## Kernel Visualization

* The input of the neural network has three channels, stand for black stones in one hot, white stones in one hot, and who's turn (black 1 white -1).
Then the input goes through a convolution layer with 64 6x6 kernels. And then some resnet layers, etc.
The first convolution layer can be visualized by encoding the three input channels as RGB.
We show black and white stones in Green and Blue, who's turn in Red.
The following pictures are the kernels for a typical training.

<table border="0">
    <tr>
        <th><img src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/17th-520-R2G0B1-x4.png"/></th>
        <th><img src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/17th-520-ch2-x4.png"/></th>
        <th><img src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/17th-520-ch0-x4.png"/></th>
        <th><img src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/17th-520-ch1-x4.png"/></th>
    </tr>
    <tr>
        <th>Three Channels</th>
        <th>Who's Turn</th>
        <th>Black stones</th>
        <th>White stones</th>
    </tr>
</table>

* This gif shows how these kernels changed during the training.

<div align=center>
   <img width="40%" src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/17th-500ms-min.gif"/>
</div>

* __Interesting (also unsolved) Phenomena__ We cannot understand why some kernel gets black (indicating all zeros) after about 240 epochs. We have some conjectures.
    * ~~The training data size is not large enough.
      So the neural network can only learn some rough knowledge rather than details.~~
      (try22) After enlarging train data by 4 in every epoch, the blacken phenomena even comes earlier (after 60 epoch)!
      And the win rate against alpha-beta pruning AI is keeping increasing.
      So maybe "blacken" is not a bad thing.
    * ~~I minus the target function by `5*policy_output.std()`.
      Maybe this term (although very small) is too singular.~~ (try23)
    * The resnet following the first layer is not wide enough, not enough to analyze these pieces of information.
      I think this conjecture is the correct one.
      (try 24) Enlarge the number of 3x3 kernel by 2 (the disk size of the network, as a result, gets larger by 3 times).
      The number of kernels blackened after 200 epochs decrease from about 35 to 28.
      (try 25) However, after decreasing the number of 6x6 kernel of the first layer from 64 to 36,
      the number of survived kernels decreased.
    * The learning rate is too large.
      (try 26) Train with lr=0.0001 rather than 0.0005 as usual, the number of kernels blackened after 360 epochs is 19.
    * This training architecture has the ability to "purify itself" or "distillate simultaneously".
      It abandons useless features during the training just like we humans.

## File Description

* fivestone_conv.py: Gomoku using "classical" alpha-beta pruning. It contains
    * `def log(msg,l=1,end="\n",logfile=None,fileonly=False):` a utility function for logging.
    * `class FiveStoneState():` the class representing a board with stones to be used in search algorithm (MCTS or alpha-beta pruning). Its evaluation function takes advantage of PyTorch's convolution function. Thus comes its name "conv".
    * `def pretty_board(gamestate):` a utility function for printing pretty board to terminal.
    * `def play_tui(human_color=-1,deep=3):` interface between human and AI.

* net_topo.py: net topology and a new GameState class adapted neural network. It contains
    * `class PVnet_cnn(nn.Module):` an obsolete policy-value network.
    * `class PV_resnet(PVnet_cnn):` the policy-value network in use. It is a 16 layer resnet.
    * `class FiveStone_CNN(FiveStoneState):` the gamestate adapted neural network.

* benchmark_utils.py: benchmark utilities.
    * `def vs_noth(state_nn,epoch):` v.s. nothing. Examine how many steps an AI can get five in a row without an opponent.
    * `def benchmark(state_nn,epoch):` benchmark raw network (one network eval using policy output) against classical alpha-beta pruning with man craft evaluation function under deep 1 search.

* fivestone_zero.py: AlphaZero architecture AI. It contains
    * `class FiveStone_ZERO(FiveStone_CNN):` a gamestate with minor changes made based on `FiveStone_CNN`.
    * `def gen_data(model,num_games,randseed,data_q,PARA_DICT):` generate train data by self playing.
    * `def gen_data_sub(model,num_games,randseed,data_q,PARA_DICT):` and `def gen_data_multithread(model,num_games):` multiprocessing version of `gen_data`.
    * `def train(model):` train the model.

## Parameter in `fivestone_zero` Explained

There is a `PARA_DICT` in `fivestone_zero.py`.
By diving into these parameters, one can get a better understanding of our program.
Its keys have the following meanings.

* `ACTION_NUM` and `POSSACT_RAD`: control the behaviour of `FiveStone_ZERO.getPossibleActions()`.
* `AB_DEEP`: search deep of alpha-beta tree. Restricted by our computational power (a single 3070), this value is always 1.
* `SOFTK`: the factor multiplied to values before the softmax generating target policy distribution.
* `LOSS_P_WT`, `LOSS_P_WT_RATIO`: controls the weight of policy loss w.r.t value loss.
* `STDP_WT`: weight of policy's standard error for generating a more decisive (larger standard error) policy.
* `FINAL_LEN` and `FINAL_BIAS`: the last `FINAL_LEN` steps of a game will be thought of as "final". Their values will be modified based on the final result. `FINAL_BIAS` controls the portion of mixing.
* `SHIFT_MAX=3`: the train boards are shifted by a random integer between -3 and 3.
* `UID_ROT=4`: each board will be rotated 4 times to generate 4 different train data.

Among these parameters, __SHIFT_MAX__ is the most important one.
AI will improve but very slow when SHIFT_MAX is set to 0.

## Training and Abelation Experiments

* The following figure is for the 17th try.
Its main parameters are `{"ACTION_NUM": 100, "POSSACT_RAD": 1, "FINAL_LEN": 4, "FINAL_BIAS": 0.5, "UID_ROT": 4, "SHIFT_MAX":3}`.
The dashed blue line indicates the number of steps taken to make five in a line (FIAL) with no opponent assigned, corresponding to the left vertical axis.
Its minimal possible value is 5, while a randomly initialized network's FIAL is around 30.
The solid lines with round data points are the win rate in percent of the raw network against man-craft AI with search deep 1, corresponding to the right vertical axis.
The win rate stables at half-half after 200 epochs (each epoch contains 30 games).
The most impressive point is, the FIAL drops to around 5 after only 5 epochs!

<div align=center>
   <img width="50%" src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try17.png"/>
</div>

* __Ablation Experiment of SHIFT_MAX__
The openings are shifted by a random integer between -SHIFT_MAX and SHIFT_MAX so that the neural network may learn more general cases.
To study the influence of SHIFT_MAX, we train the network with other parameters fixed and SHIFT_MAX varies from 0 to 3.
The result is as follows.
`SHIFT_MAX!=0` is better than `SHIFT_MAX==0`.
Although the value of SHIFT_MAX has little effect as long as it is greater than 0, I decide to fix it to 3 in future experiments.

<table border="0">
    <tr>
        <th><img width="576px" src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try17-2.png"/></th>
        <th><img width="576px" src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try18.png"/></th>
        <th><img width="576px" src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try20.png"/></th>
        <th><img width="576px" src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try19.png"/></th>
    </tr>
    <tr>
        <th>SHIFT_MAX=3</th>
        <th>SHIFT_MAX=2</th>
        <th>SHIFT_MAX=1</th>
        <th>SHIFT_MAX=0</th>
    </tr>
</table>

* __Ablation Experiment of FINAL_LEN__
The last `FINAL_LEN` steps of a game will be thought of as "final".
The final steps' values will be modified based on the final result.
I believed that this trick can make the neural network learn faster.
However, as shown by the following figures, we'd better turn it off.

<table border="0">
    <tr>
        <th><img src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try17-3.png"/></th>
        <th><img src="https://raw.githubusercontent.com/WhymustIhaveaname/FiveStone-Gongzhu/main/figures/try21.png"/></th>
    </tr>
    <tr>
        <th>FINAL_LEN=4</th>
        <th>FINAL_LEN=0</th>
    </tr>
</table>

## Persuing Better Performence

## Statistics and Behaviour Analyse

## Todo List

- [x] Ablation experiment of `FINAL_LEN` and `FINAL_BIAS`.
- [x] Ablation experiment of `SHIFT_MAX`.
- [ ] Ablation experiment of `ACTION_NUM` and `POSSACT_RAD`.
- [ ] Opening statistics.
- [ ] Behaviour statistics.
- [ ] Puzzles solving. (尤其是双杀)
- [x] Add more pretty figures.
