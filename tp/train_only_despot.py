import numpy as np
import pandas as pd
from IPython.display import display

from .models.BH_POMDP import BevertonHolt_POMDP_gtc, BevertonHolt_POMDP_gac
from .models.Surplus_POMDP import Surplus_POMDP_gtc, Surplus_POMDP_gac
from tp.simulation import simulate, evaluate
from .policy.threshold_policy import POMDPThresholdPolicy
from .policy.SCO import SCO
# from .policy.GA import GA1, GA2, CA

import timeit
import os

import functools
print = functools.partial(print, flush=True)

import inspect


def restrict_class(params, myclass):
    '''
    return a subdictionary of params containing only arguments used in myclass.__init__
    '''
    signature = inspect.signature(myclass)
    args = list(signature.parameters.keys())
    dic = dict((key, params[key]) for key in args if key in params)
    return dic


def restrict(params, method):
    '''
    return a subdictionary of params containing only arguments used in method
    '''
    args = inspect.getfullargspec(method).args
    return dict((key, params[key]) for key in args if key in params)


def train_policy(model, seed=345, MaxTry=5, UpdateTime=15, PopulationSize=50, GenerationTime=10, search_time=5, p_c=0.5, p_m=1, gen_method='SCO',
                 reset_mode='uniform', random_state=42):
    # policy learning
    np.random.seed(seed)
    #     print('seed in {train_policy}:', seed)
    b0 = model.reset(mode=reset_mode)
    t = model.transition_model
    A = model.A
    K = model.K
    print('number of thresholds', len(A) - 1)
    # print('trasition dynamics', t)
    f = lambda X: simulate(model, POMDPThresholdPolicy(A, X, K), initial_belief=b0, gamma=0.95, simlen=100, runs=10,
                           seed=seed)
    print('seed in {simulate}:', seed)

    if gen_method == 'SCO':
        best = SCO(f, 0.6, 0.5, thd_num=len(A) - 1, K=K, num_sample_thds=10, MaxTry=MaxTry, UpdateTime=UpdateTime)
    elif gen_method == 'GA1':
        best = GA1(f, thd_num=len(A) - 1, K=K, PopulationSize=PopulationSize, GenerationTime=GenerationTime, p_c=p_c, p_m=p_m)
    elif gen_method == 'GA2':
        best = GA2(f, thd_num=len(A) - 1, K=K, GenerationTime=GenerationTime, random_state=random_state)
    elif gen_method == 'CA':
        best = CA(f, thd_num=len(A) - 1, K=K, random_state=random_state, GenerationTime=GenerationTime,
                      search_time=search_time)

    thresholds = best[0]
    value = simulate(model, POMDPThresholdPolicy(A, thresholds, K), initial_belief=b0, gamma=0.95, simlen=100, runs=500,
                     eval=True, seed=seed)

    print('')
    print('learnt thresholds:', thresholds)
    print('value:', value)

    return thresholds


def evaluate_despot(simulator, world, pomdpx_path, despot_path, info, despot_t, despot, reset_world_mode = 'uniform',
             gamma=0.95, runs=100, simlen=90, seed=345, verbose=False):
    print('evaluating now')
    np.random.seed(seed)
    print('seed in {evaluate}:', seed)

    despot_eval_rslt = world.despot_evaluate(simulator.pomdpx_filename, despot_t, runs, simlen, despot_path, despot, info)

    return despot_eval_rslt


def expt_despot(model_params_list, hyperparameters, path, despot, reset_world_mode='uniform', m=2, random_state=42):
    start = timeit.default_timer()

    pomdpx_path = os.path.join(path, 'pomdpx_files')
    despot_path = os.path.join(path, 'despot_rslts')

    os.mkdir(pomdpx_path)
    os.mkdir(despot_path)

    models = []
    policies = []

    seed = hyperparameters['seed']

    despot_results = []
    for params in model_params_list:
        print('')
        print('*****Constructing POMDP model', params['ID'], 'now')
        params = dict(params)
        params['rho_noise_std'] = hyperparameters['rho_noise_std']
        params['seed'] = seed
        params['state_step'] = hyperparameters['state_step']
        params['action_step'] = hyperparameters['action_step']
        params['obs_step'] = hyperparameters['obs_step']
        params['m'] = m
        params['N'] = hyperparameters['discretisation_size']     #number to discritize
        # params['action_step'] = hyperparameters['action_step']

        # generate POMDP model
        growth_mode = hyperparameters['growth_mode']
        if params['population_model'] == 'BH':
            if growth_mode == 'gtc':
                model = BevertonHolt_POMDP_gtc(**restrict_class(params, BevertonHolt_POMDP_gtc))
            elif growth_mode == 'gac':
                model = BevertonHolt_POMDP_gac(**restrict_class(params, BevertonHolt_POMDP_gac))
        elif params['population_model'] == 'Surplus':
            if growth_mode == 'gtc':
                model = Surplus_POMDP_gtc(**restrict_class(params, Surplus_POMDP_gtc))
            elif growth_mode == 'gac':
                model = Surplus_POMDP_gac(**restrict_class(params, Surplus_POMDP_gac))

        model.make()
        # print('actions', model.actions)
        # find both normalized and unnormalized threshold policies
        # print('***Training POMDP policy now...')
        MaxTry = hyperparameters['MaxTry']
        UpdateTime = hyperparameters['UpdateTime']

        PopulationSize = hyperparameters['PopulationSize']
        GenerationTime = hyperparameters['GenerationTime']
        p_c = hyperparameters['p_c']
        p_m = hyperparameters['p_m']
        gen_method = hyperparameters['gen_method']
        random_state = random_state

        # thresholds = train_policy(model=model, seed=seed, MaxTry=MaxTry, UpdateTime=UpdateTime,
        #                           PopulationSize=PopulationSize, GenerationTime=GenerationTime, p_c=p_c, p_m=p_m,
        #                           gen_method=gen_method, reset_mode='uniform', random_state=42)

        # policies.append(POMDPThresholdPolicy(model.A, thresholds, model.K))

        # generate pomdpx file
        model.get_pomdpx_file(pomdpx_path, params['ID'])

        models.append(model)

    M = len(models)
    # for p in range(M):
    #     print('')
    #     print('------------Threshold policy trained from %s model:' %
    #             # (model_params_list[p]['name'] if 'name' in model_params_list[p] else str(p)))
    #           (model_params_list[p]['population_model']))
    #     print(policies[p])

    # evaluate all policies on all models
    values = []
    print('')
    print('*****Evaluating the policies now...')
    for m in range(M-1, M):     #only evaluate on the GT world model
        print('***Now the world model is', model_params_list[m]['ID'])
        params = {}
        params['world'] = models[m]
        params['despot_t'] = hyperparameters['despot_t']
        params['runs'] = hyperparameters['runs']
        params['simlen'] = hyperparameters['simlen']
        params['seed'] = hyperparameters['seed']
        params['gamma'] = 0.95
        params['reset_world_mode'] = reset_world_mode
        params['pomdpx_path'] = pomdpx_path
        params['despot_path'] = despot_path
        world_ID = model_params_list[m]['ID']

        vs = []
        for p in range(M):
            print('**The simulator is', model_params_list[p]['ID'])
            print('')
            params['simulator'] = models[p]
            simulator_ID = model_params_list[p]['ID']
            params['info'] = 'w%s_s%s' % (world_ID, simulator_ID)

            # # evaluate normalized policy
            # policies[p].normalize()
            # params['policy'] = policies[p]
            # vs.append(list(evaluate(**params)))

            # evaluate unnormalized policy
            # policies[p].unnormalize()
            # params['policy'] = policies[p]
            if m == p:
                params['eval'] = True
                # tp_value = list(simulate(**restrict(params, simulate)))
                # vs.append(tp_value)

                t = hyperparameters['despot_t']
                runs = hyperparameters['runs']  # 1000
                simlen = hyperparameters['simlen']  # 100

                despot_value = model.get_despot_result(t, runs, simlen, despot_path, despot, info=simulator_ID)

                vs.append(despot_value)

                # print('evaluation by threshold policy:', tp_value)
                print('evaluation by despot:', despot_value)
            else:
                params['despot'] = despot
                tmp = evaluate2(**restrict(params, evaluate2))
                # vs.append(tmp[0])
                vs.append(tmp)
        
                # print('evaluation by threshold policy:', tmp[0])
                print('evaluation by despot:', tmp)

        values.append(vs)

    index = ['$M_{%s}$' % (model_params_list[m]['ID']) for m in range(M-1, M)]

    columns = []
    for i in range(3):
        # if i%2 == 0:
        #     columns.append('$\pi_{%s}$' % (model_params_list[int(i/2)]['ID']))
        columns.append('DESPOT (Sim:%s)' % (model_params_list[i]['ID']))
    columns = np.array(columns).flatten()

    rslt = pd.DataFrame(values, index=index, columns=columns)
    # rslt['DESPOT'] = despot_results

    # df.applymap(lambda x: [round(x[0], 2), round(x[1], 2)])

    display(rslt)

    end = timeit.default_timer()
    print('')
    print('total time used:', end - start)

    return models, policies, rslt


if __name__ == "__main__":
    pass
