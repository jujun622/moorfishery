import argparse
from rnn.BH_FishRNN import BevertonHoltRNN_gac, BevertonHoltRNN_gtc
from rnn.SP_FIshRNN import SurplusRNN_gac, SurplusRNN_gtc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--class-name', type=str, default='BH_gtc',
                        help='the RNN class to train the parameters')
    parser.add_argument('--rho', type=float, default=3.0,
                        help='the proliferation rate used to generate synthetic data')
    parser.add_argument('--K', type=float, default=10000,
                        help='the carrying capacity used to generate synthetic data')
    parser.add_argument('--B0', type=float, default=5000,
                        help='the beginning biomass used to generate synthetic data')
    parser.add_argument('--q', type=float, default=0.005,
                        help='the catchability constant used to generate synthetic data')
    parser.add_argument('--c', type=float, default=10,
                        help='the constant to generate effort data')
    parser.add_argument('--num', type=int, default=50,
                        help='number of steps in synthetic data')
    # parser.add_argument('--random-state', nargs='+', type=int, default=[42, 42],
    #                     help='random state for RNN training')
    parser.add_argument('--random-state', type=int, default=42,
                        help='random state for generating data and RNN training')
    parser.add_argument('--noise-std', type=float, default=0.0,
                        help='the noise used to generate synthetic data, 0.0 for deterministic cases')
    parser.add_argument('--num-missing-years', type=int, default=0,
                        help='number of missing years in dataset')
    parser.add_argument('--effort-mode', type=str, default='constant',
                        help='the options of effort mode, can be "no decrease", "with decrease"')

    parser.add_argument('--despot', type=str, default='pomdpx_models/pomdpx',
                        help='the directory of despot')
    parser.add_argument('--reset-world-mode', type=str, default='uniform',
                        help='the mode to reset the world model for each simulation when planning')
    parser.add_argument('--m', type=int, default='2',
                        help='the constant in Surplus model')

    parser.add_argument('--seed', type=int, default=345,
                        help='the random seed for training POMDP models')
    parser.add_argument('--dis-size', type=int, default=1000,
                        help='the sample size when discretizing the POMDP models')

    parser.add_argument('--rnn-noise', type=float, default=0.0,
                        help='the noise when training RNN, 0.0 for determinstic cases')
    parser.add_argument('--rho-noise-std', type=float, default=0.1,
                        help='the noise to random sample rhos when constructing POMDP models')
    parser.add_argument('--rnn-trials', type=int, default=30,
                        help='the number of training RNN to find a good estimation of parameters')
    parser.add_argument('--num-epochs-rnn', type=int, default=15,
                        help='number of epochs when updating parametes')
    parser.add_argument('--l2-lambda', type = float, default=0.0, 
                        help='lambda value when adding l2 regularisation')

    parser.add_argument('--action-step', type=int, default=15,
                        help='the interval of action gaps')
    parser.add_argument('--state-step', type=int, default=1000,
                        help='the interval of state gaps')
    parser.add_argument('--obs-step', type=int, default=1000,
                        help='the interval of obs gaps')

    parser.add_argument('--maxtry', type=int, default=5,
                        help='the maximum try times to update thresholds in one iteration in SCO')
    parser.add_argument('--updatetime', type=int, default=10,
                        help='the number of iterations for updating the thresholds in SCO')
    parser.add_argument('--populationsize', type=int, default=50,
                        help='number of popultion size for genetic alg to optimize the thresholds')
    parser.add_argument('--generationtime', type=int, default=10,
                        help='number of generation time for genetic alg to optimize the thresholds')
    parser.add_argument('--gen-method', type=str, default='SCO',
                        help='type of genetic method for optimizing thresholds (now supporting SCO, CA)')

    parser.add_argument('--runs', type=int, default=100,
                        help='the number of simulations for planning')
    parser.add_argument('--simlen', type=int, default=90,
                        help='the length of one simulation when planning')
    parser.add_argument('--despot-t', type=float, default=0.001,
                        help='the search time used in DESPOT')

    parser.add_argument('--log', type=bool, default=True,
                        help='whether to record the printing into the log file')

    return parser.parse_args()

