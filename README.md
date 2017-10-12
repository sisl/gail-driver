# gail-driver

Utilities and scripts used to perform experiments described in "[Imitating Driver Behavior with Generative Adversarial Networks](https://arxiv.org/abs/1701.06699)". Built on [rllab](https://github.com/openai/rllab) and source code for [generative adversarial imitation learning](https://github.com/openai/imitation.git).

Train a model from the command line by running:

```
python scripts/train_gail_model.py
```

![](https://github.com/sisl/gail-driver/blob/master/gifs/congested.gif?raw=true)
An ego vehicle trained through Generative Adversarial Imitation Learning (blue) navigating a congested highway scene.

# Requirements

Julia 0.5

ForwardNets.jl ([nextgen branch](https://github.com/tawheeler/ForwardNets.jl/tree/nextgen))

AutomotiveDrivingModels.jl ([gail branch](https://github.com/akuefler/AutomotiveDrivingModels.jl))

Note: This repository is not up to date with recent changes to the following Julia packages. We recommend using the following commits of these packages:

[AutoViz.jl](https://github.com/sisl/autoviz.jl) (commit 274dd08)

[NGSIM.jl](https://github.com/sisl/NGSIM.jl) (commit f16d684)

# References
Jonathan Ho, Stefano Ermon. "[Generative Adversarial Imitation Learning](https://cs.stanford.edu/~ermon/papers/imitation_nips2016_main.pdf)". _Advances in Neural Information Processing Systems (NIPS), 2016_

Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel. "[Benchmarking Deep Reinforcement Learning for Continuous Control](http://arxiv.org/abs/1604.06778)". _Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016._

