import numpy as np
import functools
from tp.models.BH_POMDP import BevertonHolt_POMDP_gac_sim, BevertonHolt_POMDP_gtc_sim
from tp.models.Surplus_POMDP import Surplus_POMDP_gac_sim, Surplus_POMDP_gtc_sim
from .utilities.run_despot import run_despot_evaluate
from collections import Counter
import matplotlib.pyplot as plt

print = functools.partial(print, flush=True)


def belief_update(b, a, o, transition_model, observation_model):
    obs_prob = observation_model[:, a, o]

    sum_transition = np.matmul(b, transition_model[a, :, :])

    sum_obs = sum(obs_prob * sum_transition)

    new_belief = obs_prob * sum_transition / sum_obs

    return np.array(new_belief)


def evaluate(simulator, policy, world, pomdpx_path, despot_path, info, despot_t, despot, reset_world_mode = 'uniform',
             gamma=0.95, runs=100, simlen=90, seed=345, verbose=False):
    print('evaluating now')
    np.random.seed(seed)
    print('seed in {evaluate}:', seed)

    V = np.zeros(runs)
    actions = []
    # obss = []
    expected_ss = []
    for i in range(runs):
        b = simulator.reset()
        world.reset(mode=reset_world_mode)
        for t in range(simlen):
            # print('b',b)
            expected_s = policy.expected_biomass(b, simulator.K)
            expected_ss.append(expected_s)

            a_sim = policy.get_action(b, simulator.K)
            actions.append(a_sim)
            # print('a',a)
            if a_sim > np.max(world.A):
                a_world = np.max(world.A)
            else:
                a_world = a_sim
            r, o = world.step(a_world)
            # print('new_world.step(a)', o)

            if o > np.max(simulator.O):
                o = np.max(simulator.O)
            # obss.append(o)
            # print('for simulator to update belief', o)

            b = simulator.update_belief(b, a_sim, o)
            # print('updated belief', b)
            V[i] += gamma**t * r
        expected_ss.append(policy.expected_biomass(b, simulator.K))

    policy_eval_rslt = [round(np.mean(V), 2), round(np.std(V) / np.sqrt(len(V)), 4)]

    if not verbose:
        # world.generate_pomdpx(pomdpx_path, info)
        despot_eval_rslt = world.despot_evaluate(simulator.pomdpx_filename, despot_t, runs, simlen, despot_path, despot, info)

        return [policy_eval_rslt, despot_eval_rslt]
    else:
        return actions, expected_ss, policy_eval_rslt


# def evaluate(simulator, policy, world, pomdpx_path, despot_path, info, despot_t, despot, reset_world_mode = 'uniform',
#              gamma=0.95, runs=100, simlen=90, seed=345, verbose=False):
#     print('evaluating now')
#     np.random.seed(seed)
#
#     sim_K = simulator.K
#     sim_states_interval = sim_K / simulator.num_state
#     sim_obs_interval = sim_K / simulator.num_obs
#     # sim_num_actions = simulator.num_action
#
#     if world.population_model == 'BH':
#         if world.growth_mode == 'gac':
#             new_world = BevertonHolt_POMDP_gac_sim(world.rho, world.K, world.B0, world.q, sim_states_interval, sim_obs_interval, world.action_step,
#                                        world.N, world.seed, world.rho_noise_std)
#         elif world.growth_mode == 'gtc':
#             new_world = BevertonHolt_POMDP_gtc_sim(world.rho, world.K, world.B0, world.q, sim_states_interval, sim_obs_interval, world.action_step,
#                                        world.N, world.seed, world.rho_noise_std)
#     elif world.population_model == 'Surplus':
#         if world.growth_mode == 'gac':
#             new_world = Surplus_POMDP_gac_sim(world.rho, world.K, world.B0, world.q, sim_states_interval, sim_obs_interval, world.action_step,
#                                        world.N, world.m, world.seed, world.rho_noise_std)
#         elif world.growth_mode == 'gtc':
#             new_world = Surplus_POMDP_gtc_sim(world.rho, world.K, world.B0, world.q, sim_states_interval, sim_obs_interval, world.action_step,
#                                        world.N, world.m, world.seed, world.rho_noise_std)
#     new_world.make()
#
#     V = np.zeros(runs)
#     actions = []
#     # obss = []
#     expected_ss = []
#     for i in range(runs):
#         b = simulator.reset()
#         new_world.reset(mode=reset_world_mode)
#         for t in range(simlen):
#             # print('b',b)
#             expected_s = policy.expected_biomass(b, sim_K)
#             expected_ss.append(expected_s)
#
#             a_sim = policy.get_action(b, sim_K)
#             actions.append(a_sim)
#             # print('a',a)
#             if a_sim > np.max(new_world.A):
#                 a_world = np.max(new_world.A)
#             else:
#                 a_world = a_sim
#             r, o = new_world.step(a_world)
#             # print('new_world.step(a)', o)
#
#             if o > np.max(simulator.O):
#                 o = np.max(simulator.O)
#             # obss.append(o)
#             # print('for simulator to update belief', o)
#
#             b = simulator.update_belief(b, a_sim, o)
#             # print('updated belief', b)
#             V[i] += gamma**t * r
#         expected_ss.append(policy.expected_biomass(b, sim_K))
#
#     policy_eval_rslt = [round(np.mean(V), 2), round(np.std(V) / np.sqrt(len(V)), 4)]
#
#     if not verbose:
#         new_world.generate_pomdpx(pomdpx_path, info)
#         despot_eval_rslt = new_world.despot_evaluate(simulator.pomdpx_filename, despot_t, runs, simlen, despot_path, despot, info)
#
#         return [policy_eval_rslt, despot_eval_rslt]
#     else:
#         return actions, expected_ss, policy_eval_rslt


def simulate(world, policy, initial_belief=None, gamma=0.95, simlen=90, runs=100, seed=345, eval=False, verbose=False):

    model = world
    if initial_belief is None:
        initial_belief = model.initial_belief
    transition_model = model.transition_model
    observation_model = model.observation_model
    rewards = model.rewards

    np.random.seed(seed)
    # print('seed in {simulate}:', seed)
    V = np.array([0 for _ in range(runs)])

    actions = []
    # obss = []
    expected_ss = []
    for i in range(runs):
        b = initial_belief.copy()
        s = np.random.choice(model.S, p=b)
        for n in range(simlen):
            expected_s = policy.expected_biomass(b, model.K)
            expected_ss.append(expected_s)

            a = policy.get_action(b, model.K)
            actions.append(a)

            s_next = np.random.choice(model.S, p=transition_model[a][s])

            r = gamma ** n * rewards[s_next][a]

            V[i] = V[i] + r

            o = np.random.choice(model.O, p=observation_model[s_next][a])
            # obss.append(o)

            b = belief_update(b, a, o, transition_model, observation_model)
            s = s_next
        expected_ss.append(policy.expected_biomass(b, model.K))

    rslt = [round(np.mean(V), 2), round(np.std(V) / np.sqrt(len(V)), 4)]
    if eval:
        if verbose:
            return actions, expected_ss, rslt
        else:
            return rslt
    else:
        return np.mean(V)


def compare_policies(simulator, policy, world, despot_rslt_file, runs=500, simlen=100, seed=345):
    ## threshold policy
    if simulator is None:
        tp_actions, tp_expectes_ss, _ = simulate(world, policy, initial_belief=None, gamma=0.95, simlen=simlen,
                                                 runs=runs, seed=seed,
                                                 eval=True, verbose=True)
    else:
        tp_actions, tp_expectes_ss, _ = evaluate(simulator, policy, world, pomdpx_path=None, despot_path=None,
                                                 info=None,
                                                 despot_t=0, despot=None, reset_world_mode='uniform',
                                                 gamma=0.95, simlen=simlen, runs=runs, seed=seed, verbose=True)

    dic_tp_actions = Counter(tp_actions)

    ## despot
    with open(despot_rslt_file, 'r') as f:
        data = f.readlines()

    despot_actions = []
    despot_obs = []

    for i in range(len(data)):
        line = data[i]
        if 'Action =' in line:
            despot_actions.append(int(line.split()[-1].split('a')[-1]))
        if 'Observation =' in line:
            despot_obs.append(int(line.split()[-1].replace(']', 'o').split('o')[-2]))

    dic_despot_actions = Counter(despot_actions)

    ## histogram for actions
    dictionary_items = dic_tp_actions.items()
    sorted_tp_actions = dict(sorted(dictionary_items))

    dictionary_items = dic_despot_actions.items()
    sorted_despot_actions = dict(sorted(dictionary_items))

    plt.figure(figsize=(8, 5))

    plt.subplot(3, 2, 1)
    plt.bar(range(len(dic_tp_actions)), list(sorted_tp_actions.values()), align='center')
    plt.xticks(range(len(dic_tp_actions)), list(sorted_tp_actions.keys()))
    plt.title('threshold policy')

    plt.subplot(3, 2, 2)
    plt.bar(range(len(dic_despot_actions)), list(sorted_despot_actions.values()), align='center')
    plt.xticks(range(len(dic_despot_actions)), list(sorted_despot_actions.keys()))
    plt.title('despot')

    ## plot for one run
    length = simlen

    despot_expected_ss = []

    if simulator is None:
        simulator = world
    b0 = simulator.initial_belief
    for i in range(length):
        despot_expected_ss.append(policy.expected_biomass(b0, simulator.K))

        b1 = simulator.update_belief(b0, despot_actions[i], despot_obs[i])
        b0 = b1
    despot_expected_ss.append(policy.expected_biomass(b0, simulator.K))

    # plot

    plt.subplot(3, 2, 3)
    plt.plot(np.arange(length), tp_actions[:length], '.', label='actions')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(np.arange(length), despot_actions[:length], '.', label='actions')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(np.arange(length + 1), tp_expectes_ss[:(length + 1)], label='expected biomass')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(np.arange(length + 1), despot_expected_ss, label='expected biomass')

    plt.legend()
    plt.tight_layout()
    plt.show()

# def compare_policies(simulator, policy, world, despot_actions_log, runs=500, simlen=100, seed=345):
#     ## threshold policy
#     if simulator is None:
#         tp_actions, _, _ = simulate(world, policy, initial_belief=None, gamma=0.95, simlen=simlen, runs=runs, seed=seed,
#                                  eval=True, print_actions=True)
#     else:
#         tp_actions, _, _ = evaluate(simulator, policy, world, pomdpx_path=None, despot_path=None, info=None,
#                                  despot_t=0, despot=None, reset_world_mode = 'uniform',
#                                 gamma=0.95, simlen=simlen, runs=runs, seed=seed, print_actions=True)
#
#     tp_actions = Counter(tp_actions)
#
#     ## despot
#     with open(despot_actions_log, 'r') as f:
#         data = f.readlines()
#
#     despot_actions = {}
#     for i in range(len(data)):
#         line = data[i]
#         a = int(line.replace(':', ' ').split()[-2])
#         num = int(line.replace(':', ' ').split()[0])
#         despot_actions[a] = num
#
#     ## comparison
#     dictionary_items = tp_actions.items()
#     sorted_tp_actions = dict(sorted(dictionary_items))
#
#     plt.subplot(1, 2, 1)
#     plt.bar(range(len(tp_actions)), list(sorted_tp_actions.values()), align='center')
#     plt.xticks(range(len(tp_actions)), list(sorted_tp_actions.keys()))
#     plt.title('threshold policy')
#
#     plt.subplot(1, 2, 2)
#     dictionary_items = despot_actions.items()
#     sorted_despot_actions = dict(sorted(dictionary_items))
#
#     plt.bar(range(len(despot_actions)), list(sorted_despot_actions.values()), align='center')
#     plt.xticks(range(len(despot_actions)), list(sorted_despot_actions.keys()))
#     plt.title('despot')
#
#     plt.show()


# def plot_one_run(simulator, policy, world, despot_rslt_log, runs = 1, simlen = 100, seed = 345):
#     length = runs * simlen
#
#     ## threshold policy
#     if simulator is None:
#         tp_actions, tp_expectes_ss, _ = simulate(world, policy, initial_belief=None, gamma=0.95, simlen=simlen, runs=runs, seed=seed,
#                         eval = True, verbose = True)
#     else:
#         tp_actions, tp_expectes_ss, _ = evaluate(simulator, policy, world, pomdpx_path=None, despot_path=None, info=None,
#                     despot_t = 0, despot = None, reset_world_mode = 'uniform',
#                     gamma = 0.95, simlen = simlen, runs = runs, seed = seed, verbose = True)
#
#     ## despot
#     with open(despot_rslt_log, 'r') as f:
#         data = f.readlines()
#
#     despot_actions = []
#     despot_obs = []
#
#     for i in range(len(data)):
#         line = data[i]
#         if 'Action =' in line:
#             despot_actions.append(int(line.split()[-1].split('a')[-1]))
#         if 'Observation =' in line:
#             despot_obs.append(int(line.split()[-1].replace(']','o').split('o')[-2]))
#
#     despot_expected_ss = []
#     b0 = simulator.initial_belief
#     for i in range(length):
#         despot_expected_ss.append(policy.expected_biomass(b0, simulator.K))
#
#         b1 = simulator.update_belief(b0, despot_actions[i], despot_obs[i])
#         b0 = b1
#     despot_expected_ss.append(policy.expected_biomass(b0, simulator.K))
#
#     ## plot
#
#     f = plt.figure(figsize=(8, 5))
#     ax1 = f.add_subplot(221)
#     ax2 = f.add_subplot(222)
#     ax1.plot(np.arange(length), tp_actions, '.')
#     ax1.legend('actions')
#     ax2.plot(np.arange(length+1), tp_expectes_ss, label='expected biomass')
#     ax1.set_title('threshold policy')
#
#     ax3 = f.add_subplot(223)
#     ax4 = f.add_subplot(224)
#     ax3.plot(np.arange(length), despot_actions[:100], '.', label='actions')
#     ax4.plot(np.arange(length+1), despot_expected_ss, label='despot ')
#     ax3.set_title('despot')
#
#     plt.tight_layout()
#     plt.show()







