# Model-based Offline Reinforcement Learning for Sustainable Fishery Management
This repository is the implementation of [MOOR](https://doi.org/10.1111/exsy.13324).

## Instructions
An exmaple command is as follows. 
The descrption of arguments can be found in `get_args.py`.
```
python expt_for_args.py --rho 2.0 --seed 4020 --rnn-noise 0.1 --random-state 5051 --class-name 'BH_gac' --num-missing-years 5 --noise-std 0.1 --K 10000 --B0 5000 --rnn-trials 40 --runs 500 --simlen 100 --despot-t 0.1 --c 10 
```

### Note
* Install [DESPOT](https://github.com/AdaCompNUS/despot) and specify DESPOT path by `--despot`.


