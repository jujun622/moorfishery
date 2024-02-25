from rnn.trainRNN import *
from tp.train import *
import torch
import timeit
import sys
import pickle
import functools
import argparse
from get_args import get_args
from rnn.BH_FishRNN import BevertonHoltRNN_gac, BevertonHoltRNN_gtc
from rnn.SP_FIshRNN import SurplusRNN_gac, SurplusRNN_gtc

print = functools.partial(print, flush=True) # NOTE: disable this if need to print a lot of output


def run_expt(class_name, sampler_args, hyperparameters, despot, m=2, reset_world_mode='uniform', path=None):
    print(class_name)

    # generate and check data
    print('m=', m)
    start = timeit.default_timer()

    rnn_noise = hyperparameters['rnn_noise']
    seed = hyperparameters['seed']
    random_state = sampler_args['random_state']
    trials = hyperparameters['rnn_trials']
    l2_lambda = hyperparameters['l2_lambda']
    num_epochs = hyperparameters['num_epochs_rnn']

    np.random.seed(seed)
    print('seed in {run_expt}:', seed)
    data_dic = class_name.sample_data(**sampler_args)

    if check_negative(data_dic['complete data']['biomasses'],
                      data_dic['complete data']['efforts'],
                      data_dic['complete data']['catches']):
        raise Exception('negative values detected')

    # sampled_data = {'efforts': efforts, 'catches': catches, 'biomasses': biomasses}

    if sampler_args['noise_std'] == 0.0:
        batch_size = 1
    else:
        batch_size = 5

    learned_params = {}
    learned_nets = []

    if 'gtc' in str(class_name):
        bh_net = BevertonHoltRNN_gtc()
        sp_net = SurplusRNN_gtc(m=m)
        hyperparameters['growth_mode'] = 'gtc'
        print('growth mode: first grow then catch')
    if 'gac' in str(class_name):
        bh_net = BevertonHoltRNN_gac()
        sp_net = SurplusRNN_gac(m=m)
        hyperparameters['growth_mode'] = 'gac'
        print('growth mode: grow and catch at the same time')

    # learn BH model
    optimizer = torch.optim.LBFGS(bh_net.parameters(), lr=0.5)
    best_sol_bh, best_mse_bh, best_bh_net = find_sol(bh_net, optimizer, data_dic, num_epochs=num_epochs, batch_size=batch_size,
                                        mse_threshold=1.5,
                                        restart=True, random_init=True, noise_std=rnn_noise, verbose=False,
                                        normalize=True,
                                        weight_loss=False, random_state=random_state, trials=trials, l2_lambda=l2_lambda)
    print('best sol', best_sol_bh)
    # print('best sol', best_sol_bh, file = sys.stderr)
    print('best mse', best_mse_bh)

    learned_nets.append(best_bh_net)

    # learn SP model
    optimizer = torch.optim.LBFGS(sp_net.parameters(), lr=0.5)
    best_sol_sp, best_mse_sp, best_sp_net = find_sol(sp_net, optimizer, data_dic, num_epochs=num_epochs, batch_size=batch_size,
                                        mse_threshold=1,
                                        m=m, restart=True, random_init=True, noise_std=rnn_noise, verbose=False,
                                        normalize=True,
                                        weight_loss=False, random_state=random_state, trials=trials)
    print('best sol', best_sol_sp)
    print('best mse', best_mse_sp)

    learned_params['BH params'] = best_sol_bh
    learned_params['BH MSE'] = best_mse_bh
    learned_params['Surplus params'] = best_sol_sp
    learned_params['Surplus MSE'] = best_mse_sp

    learned_nets.append(best_sp_net)

    print('')
    print('------BH params learnt from RNN')
    print('best sol', best_sol_bh)
    print('best mse', best_mse_bh)
    print('')
    print('------Surplus params learnt from RNN')
    print('best sol', best_sol_sp)
    print('best mse', best_mse_sp)
    print('')

    # generate POMDPs
    if 'Beverton' in str(class_name):
        tmp = sampler_args['rho']
        m3 = 'BH'
    elif 'Surplus' in str(class_name):
        tmp = sampler_args['r']
        m3 = 'Surplus'

    model_params_list = [
        {'ID': 'BH', 'K': best_sol_bh['K'], 'rho': best_sol_bh['rho'], 'B0': best_sol_bh['B0'], 'q': best_sol_bh['q'],
         'population_model': 'BH'},
        {'ID': 'SP', 'K': best_sol_sp['K'], 'rho': best_sol_sp['rho'], 'B0': best_sol_sp['B0'], 'q': best_sol_sp['q'],
         'population_model': 'Surplus'},
        {'ID': 'GT', 'K': sampler_args['K'], 'rho': tmp, 'B0': sampler_args['B0'], 'q': sampler_args['q'],
         'population_model': m3}
        ]

    models, policies, rslts = expt(model_params_list, hyperparameters, path, despot, reset_world_mode=reset_world_mode,
                                   m=m, random_state= random_state)

    end = timeit.default_timer()
    print('Finish the whole experiment used:', end - start)

    final_results = {'args': sampler_args,
                     'sampled data': data_dic,
                     'hyperparameters': hyperparameters,
                     'learned params': learned_params,
                     'learned nets': learned_nets,
                     'pomdp models': models,
                     'threshold policies': policies,
                     'value tables': rslts}

    pklfile = os.path.join(path, 'results.pkl')
    pickle.dump(final_results, open(pklfile, 'wb'))

    return learned_params, learned_nets, models, policies, rslts


if __name__ == "__main__":
    args = get_args()
    print('ARGUMENTS used:', args)

    if args.class_name == 'BH_gtc':
        class_name = BevertonHoltRNN_gtc
    elif args.class_name == 'BH_gac':
        class_name = BevertonHoltRNN_gac
    elif args.class_name == 'SP_gtc':
        class_name = SurplusRNN_gtc
    elif args.class_name == 'SP_gac':
        class_name = SurplusRNN_gac

    class_name = BevertonHoltRNN_gac

    sampler_args = {'rho': args.rho, 'K': args.K, 'B0': args.B0, 'q': args.q, 'c': args.c, 'num': args.num,
                    'random_state': args.random_state,
                    'noise_std': args.noise_std, 'num_missing_years': args.num_missing_years,
                    'effort_mode': 'no decrease'}

    hyperparameters = {'seed': args.seed, 'rnn_noise': args.rnn_noise, 'rho_noise_std': args.rho_noise_std,
                       'rnn_trials': args.rnn_trials,
                       'action_step': args.action_step, 'state_step': args.state_step, 'obs_step': args.obs_step,
                       'MaxTry': args.maxtry, 'UpdateTime': args.updatetime,
                       'runs': args.runs, 'simlen': args.simlen, 'despot_t': args.despot_t,
                       'discretisation_size': args.dis_size,
                       'PopulationSize': args.populationsize, 'gen_method': args.gen_method,
                       'GenerationTime': args.generationtime, 
                       'l2_lambda': args.l2_lambda, 'num_epochs_rnn': args.num_epochs_rnn}

    despot = args.despot
    reset_world_mode = args.reset_world_mode
    m = args.m

    log = args.log

    data_paras = 'rho%s_noise%s_seed%s_state%s_rnn%s_my%s' % (sampler_args['rho'], sampler_args['noise_std'], hyperparameters['seed'], sampler_args['random_state'], hyperparameters['rnn_noise'], sampler_args['num_missing_years'])
    path = os.path.join(os.getcwd(), 'tmp', 'new_efforts_nodecrease', str(class_name).replace("'", '.').split('.')[-2], data_paras)
    if not os.path.exists(path):
        os.makedirs(path)

    if log:
        logfile = os.path.join(path, 'log.log')
        stdout = sys.stdout
        sys.stdout = open(logfile, 'w')

    print('ARGUMENTS used:', args)

    run_expt(class_name, sampler_args, hyperparameters, despot, m, reset_world_mode, path)

    if log:
        sys.stdout.close()
        sys.stdout = stdout


