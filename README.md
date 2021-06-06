# FiveStone-Gongzhu

Solving Fivestone using AlphaZero's architecture and [Gongzhu-Society](https://github.com/Gongzhu-Society/MCTS)'s package.

### Enviroment Setup

```
git clone https://github.com/Victerose/FiveStone-Gongzhu.git
cd FiveStone-Gongzhu
git clone https://github.com/Gongzhu-Society/MCTS.git
cd MCTS
git checkout alphabeta
cd ..
```

### File Description

* fivestone_conv.py: fivestone using "classical" alpha-beta pruning. Its evaluation function takes advantage of PyTorch's convolution function. Thus comes its name "conv".
* net_topo.py: net topology and other utility functions including benchmark, etc.
* fivestone_cnn.py: supervised learning neural network AI.
* fivestone_zero.py: self-learning AI.