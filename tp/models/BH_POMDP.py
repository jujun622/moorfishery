from .FishPOMDP import *
from .FishModels import BevertonHolt_gtc, BevertonHolt_gac, catch_model, catch_from_updated_biomass, BevertonHolt
import numpy as np
import xml.dom.minidom
from ..utilities.generate_pomdpx_file import generate_pomdpx_file
import timeit

import functools
print = functools.partial(print, flush=True)


class BevertonHolt_POMDP_gac(FishPOMDP):
    def __init__(self, rho, K, B0, q, state_step, obs_step, action_step, N, seed=345, rho_noise_std=0.1):
        super(BevertonHolt_POMDP_gac, self).__init__()

        self.population_model = 'BH'
        self.growth_mode = 'gac'
        self.rho = rho
        self.K = K
        self.B0 = B0
        self.q = q
        self.action_step = action_step
        self.state_step = state_step
        self.obs_step = obs_step

        self.num_state = np.int(np.ceil(self.K / self.state_step))
        self.num_obs = np.int(np.ceil(self.K / self.obs_step))
        # self.num_state = num_state
        self.max_action = np.floor(1/q)
        self.num_action = np.int(np.ceil(self.max_action /action_step))
        # self.num_obs = num_obs

        self.S = list(range(self.num_state))
        self.A = list(range(self.num_action))
        self.O = list(range(self.num_obs))

        self.states = np.concatenate((np.array([[i, i + 1] for i in range(self.num_state - 1)]) * self.state_step,
                                      np.array([[self.state_step * (self.num_state - 1), self.K]])), axis=0)
        self.obss = np.concatenate((np.array([[i, i + 1] for i in range(self.num_obs - 1)]) * self.obs_step,
                                    np.array([[self.obs_step * (self.num_obs - 1), self.K]])), axis=0)

        # self.states = np.array([[i, i + 1] for i in range(num_state)]) * 1.0 * K / num_state
        self.actions_interval = np.array([[i, i + 1] for i in range(self.num_action)]) * action_step * self.q
        self.actions_interval[-1] = [self.actions_interval[-1][0], self.max_action * self.q]
        self.actions = [np.average(self.actions_interval[i]) for i in range(self.num_action)]
        # self.obss = np.array([[i, i + 1] for i in range(num_obs)]) * 1.0 * K / num_obs

        self.N = N
        self.seed = seed
        self.rho_noise_std = rho_noise_std

        self.belief = None
        self.state = None

        self.pomdpx_filename = None
        self.despot_eval = []

    # def discretize_transition(self):
    #     # generate discretized transition model, observation model, reward
    #     transition_model = np.zeros((self.num_action, self.num_state, self.num_state))
    #     for i in range(self.num_action):
    #         a = self.actions[i]
    #         for j in range(self.num_state):
    #             ss0 = np.random.uniform(self.states[j][0], self.states[j][1], size=self.N)
    #             rhos = np.random.normal(self.rho, self.rho_noise_std, size=self.N)
    #             ss1 = BevertonHolt_gac(ss0, self.K, rhos, a) #* np.exp(np.random.normal(0.0, 0.1, size=self.N))
    #
    #             for m in range(len(ss1)):
    #                 if ss1[m] < 0:
    #                     ss1[m] = 0
    #                 if ss1[m] > self.K:
    #                     ss1[m] = self.K
    #             for k in range(self.num_state):
    #                 sk = self.states[k]
    #                 l, u = sk
    #                 if np.abs(sk[1] - self.K) < 1E-10:
    #                     u += 1E-10
    #
    #                 transition_model[i][j][k] = ((ss1 >= l) & (ss1 < u)).mean()
    #     transition_model = transition_model * 0.9999 + 0.0001 / self.num_state
    #     return transition_model

    def discretize_transition(self):
        print('new discretising tr')

        ss0 = np.array([[np.random.uniform(self.states[j][0], self.states[j][1], size=self.N)
                         for j in range(self.num_state)] for i in range(self.num_action)])
        rhos = np.array([np.random.normal(self.rho, self.rho_noise_std, size=ss0.shape)])[0]

        ss1 = np.array([[BevertonHolt_gac(ss0[i][j], self.K, rhos[i][j], self.actions[i])
                         for j in range(len(ss0[i]))]
                        for i in range(len(ss0))])

        tr = np.array([[[((ss1[i][j] >= self.states[k][0]) & (ss1[i][j] < self.states[k][1])).mean()
                         for k in range(self.num_state)]
                        for j in range(len(ss1[i]))]
                       for i in range(len(ss1))])

        tr = tr * 0.9999 + 0.0001 / self.num_state

        return tr

    def discretize_observation(self, noise=0.1, epsilon=1e-6):       #growth/harvest at the sam time

        observation_model = np.zeros((self.num_state, self.num_action, self.num_obs))

        distribution_tuple = []

        for i in range(self.num_action):
            a = self.actions[i]
            rhos = np.random.normal(self.rho, self.rho_noise_std, size=self.N)
            print('a', i)

            for j in range(self.num_state):
                print('s', j)
                xs = np.random.uniform(self.states[j][0], self.states[j][1], size=self.N)
                xs1 = BevertonHolt_gac(xs, self.K, rhos, a)
                xs1[xs1 > self.K] = self.K
                # bins = np.linspace(0, 0.9, 10) * self.K
                state_bins = [self.states[i][0] for i in range(self.num_state)]
                discrete_xs = np.digitize(xs, state_bins, right=False) - 1
                discrete_xs1 = np.digitize(xs1, state_bins, right=False) - 1

                obs1 = catch_model(xs, a) * np.exp(np.random.normal(0.0, noise, size=self.N))
                obs1[np.where(obs1 > self.K)] = self.K
                obs_bins = [self.obss[i][0] for i in range(self.num_obs)]
                discrete_obs = np.digitize(obs1, obs_bins, right=False) - 1

                for k in range(len(xs)):
                    distribution_tuple.append([i, discrete_xs[k], discrete_xs1[k], discrete_obs[k]])

        updated_distribution_tuple = np.array([np.array(xi) for xi in distribution_tuple])

        actions_list = np.array([xi[0] for xi in updated_distribution_tuple])
        # current_state_list = np.array([xi[1] for xi in updated_distribution_tuple])
        next_state_list = np.array([xi[2] for xi in updated_distribution_tuple])
        observation_list = np.array([xi[3] for xi in updated_distribution_tuple])

        joint_distribution = np.zeros((self.num_state, self.num_action, self.num_obs))

        for i in range(self.num_state):
            for j in range(self.num_action):
                for k in range(self.num_obs):
                    joint_distribution[i, j, k] = (
                            (next_state_list == i + 1) & (actions_list == j) & (observation_list == k + 1)).mean()

        marginal_distribution = np.sum(joint_distribution, 2)

        for i in range(self.num_state):
            for j in range(self.num_action):
                if marginal_distribution[i, j] != 0:
                    for k in range(self.num_obs):
                        observation_model[i, j, k] = joint_distribution[i, j, k] / marginal_distribution[i, j]

        for i in range(self.num_state):
            for j in range(self.num_action):
                if np.sum(observation_model[i, j, :]) == 0:
                    observation_model[i, j, :] = np.ones(self.num_obs) / sum(np.ones(self.num_obs))

        # observation_model = self.noisy_matrix(observation_model, epsilon)
        observation_model = observation_model * 0.9999 + 0.0001 / self.num_obs
        return observation_model

    def discretize_tr_ob(self, noise=0.1):
        print('new discretising obs')
        start = timeit.default_timer()

        ss0 = np.array([[np.random.uniform(self.states[j][0], self.states[j][1], size=self.N)
                         for j in range(self.num_state)] for i in range(self.num_action)])
        rhos = np.array([np.random.normal(self.rho, self.rho_noise_std, size=ss0.shape)])[0]

        ss1 = np.array([[BevertonHolt_gac(ss0[i][j], self.K, rhos[i][j], self.actions[i])
                         for j in range(len(ss0[i]))]
                        for i in range(len(ss0))])
        print('finish generating ss1, used', timeit.default_timer()-start)

        s_bins = [self.states[i][0] for i in range(self.num_state)]
        # s_bins=s_bins.append(self.K)
        ss1d = np.array([np.digitize(ss1[i], s_bins, right=False) for i in range(len(ss1))]) - 1
        print('finish generating ss1d, used', timeit.default_timer() - start)

        obs = np.array([[catch_model(ss0[i][j], self.actions[i], noise) * np.exp(np.random.normal(0.0, noise, size=self.N))
                         for j in range(len(ss0[i]))]
                        for i in range(len(ss0))])
        print('finish generating obs, used', timeit.default_timer() - start)

        o_bins = [self.obss[i][0] for i in range(self.num_obs)]
        # s_bins=s_bins.append(self.K)
        obsd = np.array([np.digitize(obs[i], o_bins, right=False) for i in range(len(obs))]) - 1
        print('finish generating obsd, used', timeit.default_timer() - start)

        tupl = np.array([[j, i, ss1d[i, j, k], obsd[i, j, k]]
                         for j in range(self.num_state)
                         for i in range(self.num_action)
                         for k in range(self.N)
                         ])

        print('finish generating tupl, used', timeit.default_timer() - start)

        obm = np.zeros((self.num_state, self.num_action, self.num_obs))
        trm = np.zeros((self.num_action, self.num_state, self.num_state))

        for (i, j, m, n) in tupl:
            #     print(i,j,k)
            obm[m, j, n] += 1
            trm[j, i, m] += 1

        ob = np.array([[obm[i, j, :] / sum(obm[i, j, :]) if sum(obm[i, j, :]) != 0 else np.ones(
            self.num_obs) / self.num_obs
                   for j in range(self.num_action)]
                  for i in range(self.num_state)])

        print('finish ob', timeit.default_timer() - start)

        ob = ob * 0.9999 + 0.0001 / self.num_obs

        tr = np.array([[trm[i, j, :] / self.N
                   for j in range(self.num_state)]
                  for i in range(self.num_action)])

        tr = tr * 0.9999 + 0.0001 / self.num_state

        print('finish tr', timeit.default_timer() - start)

        end = timeit.default_timer()
        print('constructing obs and trans used', end - start)

        return tr, ob

    def discretize_reward(self, noise=0.1):
        rewards = np.zeros((self.num_state, self.num_action))

        # rhos = np.random.normal(self.rho, self.rho_noise_std, size=self.N)

        for s in range(self.num_state):
            xs = np.random.uniform(self.states[s][0], self.states[s][1], size=self.N)
            for a in range(self.num_action):
                rewards[s, a] = (catch_model(xs, self.actions[a])* np.exp(np.random.normal(0.0, noise, size=self.N))).mean()
        return rewards


class BevertonHolt_POMDP_gtc(BevertonHolt_POMDP_gac):
    def __init__(self, rho, K, B0, q, state_step, obs_step, action_step, N, seed=345, rho_noise_std=0.1):
        BevertonHolt_POMDP_gac.__init__(self, rho, K, B0, q, state_step, obs_step, action_step, N, seed, rho_noise_std)
        self.growth_mode = 'gtc'

    def discretize_transition(self):
        # generate discretized transition model, observation model, reward
        transition_model = np.zeros((self.num_action, self.num_state, self.num_state))
        for i in range(self.num_action):
            a = self.actions[i]
            # if i == 0:
            #     print('action', a)
            for j in range(self.num_state):
                ss0 = np.random.uniform(self.states[j][0], self.states[j][1], size=self.N)
                rhos = np.random.normal(self.rho, self.rho_noise_std, size=self.N)
                ss1 = BevertonHolt_gtc(ss0, self.K, rhos, a) * np.exp(np.random.normal(0.0, 0.1, size=self.N))
                # if i == 0:
                #     print('this state', ss0)
                #     print('next state', ss1)
                for m in range(len(ss1)):
                    if ss1[m] < 0:
                        ss1[m] = 0
                    if ss1[m] > self.K:
                        ss1[m] = self.K
                for k in range(self.num_state):
                    sk = self.states[k]
                    l, u = sk
                    if np.abs(sk[1] - self.K) < 1E-10:
                        u += 1E-10

                    transition_model[i][j][k] = ((ss1 >= l) & (ss1 < u)).mean()
        transition_model = transition_model * 0.9999 + 0.0001 / self.num_state
        return transition_model

    def discretize_observation(self, noise=0.1, epsilon=1e-6):       #growth/harvest at the sam time

        observation_model = np.zeros((self.num_state, self.num_action, self.num_obs))

        distribution_tuple = []

        for i in range(self.num_action):
            a = self.actions[i]
            rhos = np.random.normal(self.rho, self.rho_noise_std, size=self.N)

            for j in range(self.num_state):
                xs = np.random.uniform(self.states[j][0], self.states[j][1], size=self.N)
                xs1 = BevertonHolt_gtc(xs, self.K, rhos, a)
                xs1[xs1 > self.K] = self.K
                # bins = np.linspace(0, 0.9, 10) * self.K
                state_bins = [self.states[i][0] for i in range(self.num_state)]
                discrete_xs = np.digitize(xs, state_bins, right=False) - 1
                discrete_xs1 = np.digitize(xs1, state_bins, right=False) - 1

                obs1 = catch_from_updated_biomass(a, xs1) * np.exp(np.random.normal(0.0, noise, size=self.N))
                obs1[np.where(obs1 > self.K)] = self.K
                obs_bins = [self.obss[i][0] for i in range(self.num_obs)]
                discrete_obs = np.digitize(obs1, obs_bins, right=False) - 1

                for k in range(len(xs)):
                    distribution_tuple.append([i, discrete_xs[k], discrete_xs1[k], discrete_obs[k]])

        updated_distribution_tuple = np.array([np.array(xi) for xi in distribution_tuple])

        actions_list = np.array([xi[0] for xi in updated_distribution_tuple])
        next_state_list = np.array([xi[2] for xi in updated_distribution_tuple])
        observation_list = np.array([xi[3] for xi in updated_distribution_tuple])

        joint_distribution = np.zeros((self.num_state, self.num_action, self.num_obs))

        for i in range(self.num_state):
            for j in range(self.num_action):
                for k in range(self.num_obs):
                    joint_distribution[i, j, k] = (
                            (next_state_list == i + 1) & (actions_list == j) & (observation_list == k + 1)).mean()

        marginal_distribution = np.sum(joint_distribution, 2)

        for i in range(self.num_state):
            for j in range(self.num_action):
                if marginal_distribution[i, j] != 0:
                    for k in range(self.num_obs):
                        observation_model[i, j, k] = joint_distribution[i, j, k] / marginal_distribution[i, j]

        for i in range(self.num_state):
            for j in range(self.num_action):
                if np.sum(observation_model[i, j, :]) == 0:
                    observation_model[i, j, :] = np.ones(self.num_obs) / sum(np.ones(self.num_obs))

        observation_model = observation_model * 0.9999 + 0.0001 / self.num_obs
        return observation_model

    def discretize_reward(self, noise=0.1):
        rewards = np.zeros((self.num_state, self.num_action))

        rhos = np.random.normal(self.rho, self.rho_noise_std, size=self.N)

        for s in range(self.num_state):
            xs = np.random.uniform(self.states[s][0], self.states[s][1], size=self.N)
            for a in range(self.num_action):
                rewards[s, a] = (catch_model(BevertonHolt(xs, self.K, rhos), self.actions[a]) * np.exp(np.random.normal(0.0, noise, size=self.N))).mean()
        return rewards


class BevertonHolt_POMDP_gac_sim(BevertonHolt_POMDP_gac):
    def __init__(self, rho, K, B0, q, state_interval, obs_interval, action_step, N, seed=345, rho_noise_std=0.1):
        # super(BevertonHolt_POMDP_gac, self).__init__()

        self.rho = rho
        self.K = K
        self.B0 = B0
        self.q = q

        self.num_state = int(K / state_interval if K % state_interval == 0 else K // state_interval + 1)
        self.num_obs = int(K / obs_interval if K % obs_interval == 0 else K // obs_interval + 1)

        self.S = list(range(self.num_state))
        self.O = list(range(self.num_obs))

        self.states = np.concatenate((np.array([[i, i + 1] for i in range(self.num_state-1)]) * state_interval,
                                      np.array([[state_interval*(self.num_state-1), K]])), axis=0)
        self.obss = np.concatenate((np.array([[i, i + 1] for i in range(self.num_obs-1)]) * obs_interval,
                                      np.array([[obs_interval*(self.num_obs-1), K]])), axis=0)

        self.max_action = np.floor(1/q)
        self.num_action = np.int(np.ceil(self.max_action / action_step))
        self.A = list(range(self.num_action))

        self.actions_interval = np.array([[i, i + 1] for i in range(self.num_action)]) * action_step * self.q
        self.actions_interval[-1] = [self.actions_interval[-1][0], self.max_action * self.q]
        self.actions = [np.average(self.actions_interval[i]) for i in range(self.num_action)]

        self.N = N
        self.seed = seed
        self.rho_noise_std = rho_noise_std

        self.belief = None
        self.state = None

        self.pomdpx_filename = None
        self.despot_eval = []

        # self.match_obs_sim = False

    def match_obs_with_simulator(self, num_sim_obs):
        obs_num_gap = self.num_obs - num_sim_obs
        new_observation_model = np.zeros((self.num_state, self.num_action, num_sim_obs))
        for i in range(self.num_state):
            for j in range(self.num_action):
                for k in range(num_sim_obs):
                    new_observation_model[i, j, k] = self.observation_model[i,j,k]
                for k in range(obs_num_gap):
                    new_observation_model[i, j, num_sim_obs-1] += self.observation_model[i, j, num_sim_obs+k]
        return new_observation_model

    def match_states_with_simulator(self, num_sim_states):
        states_num_gap = self.num_state - num_sim_states
        new_transition_model = np.zeros((self.num_action, num_sim_states, num_sim_states))
        tmp = np.zeros((self.num_action, self.num_state, num_sim_states))
        for i in range(self.num_action):
            for j in range(self.num_state):
                for k in range(num_sim_states):
                    if j < num_sim_states:
                        new_transition_model[i, j, k] = self.transition_model[i,j,k]
                    tmp[i,j,k] = self.transition_model[i,j,k]
                for k in range(states_num_gap):
                    if j < num_sim_states:
                        new_transition_model[i, j, num_sim_states-1] += self.transition_model[i, j, num_sim_states+k]
                    tmp[i,j,num_sim_states-1] += self.transition_model[i,j,num_sim_states+k]
            for j in range(states_num_gap):
                new_transition_model[i,num_sim_states-1,:] += tmp[i,num_sim_states-1,:]
        return new_transition_model

    def match_b0_with_simulator(self, num_sim_states):
        states_num_gap = self.num_state - num_sim_states
        new_initial_belief = np.zeros(num_sim_states)
        for i in range(num_sim_states):
            new_initial_belief[i] = self.initial_belief[i]
            if i == num_sim_states-1:
                for k in range(states_num_gap):
                    new_initial_belief[i] += self.initial_belief[num_sim_states+k]
        return new_initial_belief

    def generate_pomdpx_new_world(self, num_sim_obs, num_sim_states, addinfo=None):
        if self.num_obs > num_sim_obs:
            new_observation_model = self.match_obs_with_simulator(num_sim_obs)
            # if self.num_state > num_sim_states:
            #     new_transition_model = self.match_states_with_simulator(num_sim_states)
            #     new_initial_belief = self.match_b0_with_simulator(num_sim_states)
            filename = generate_pomdpx_file(self.num_state, self.num_action, num_sim_obs, self.initial_belief, self.transition_model,
                                   new_observation_model, self.rewards, self.rho, self.K, self.seed, self.rho_noise_std, addinfo)
            self.pomdpx_filename_sim.append(filename)
            print('generated pomdpx file for %s' % (addinfo))
            return filename
        else:
            print('no need to construct a new pomdpx file')
            return self.pomdpx_filename


class BevertonHolt_POMDP_gtc_sim(BevertonHolt_POMDP_gtc):
    def __init__(self, rho, K, B0, q, state_interval, obs_interval, action_step, N, seed=345, rho_noise_std=0.1):
        # super(BevertonHolt_POMDP_gtc, self).__init__()

        self.rho = rho
        self.K = K
        self.B0 = B0
        self.q = q

        self.num_state = int(K / state_interval if K % state_interval == 0 else K // state_interval + 1)
        self.num_obs = int(K / obs_interval if K % obs_interval == 0 else K // obs_interval + 1)

        self.S = list(range(self.num_state))
        self.O = list(range(self.num_obs))

        self.states = np.concatenate((np.array([[i, i + 1] for i in range(self.num_state - 1)]) * state_interval,
                                      np.array([[state_interval * (self.num_state - 1), K]])), axis=0)
        self.obss = np.concatenate((np.array([[i, i + 1] for i in range(self.num_obs - 1)]) * obs_interval,
                                    np.array([[obs_interval * (self.num_obs - 1), K]])), axis=0)

        self.max_action = np.floor(1/q)
        self.num_action = np.int(np.ceil(self.max_action / action_step))
        self.A = list(range(self.num_action))

        self.actions_interval = np.array([[i, i + 1] for i in range(self.num_action)]) * action_step * self.q
        self.actions_interval[-1] = [self.actions_interval[-1][0], self.max_action * self.q]
        self.actions = [np.average(self.actions_interval[i]) for i in range(self.num_action)]

        self.N = N
        self.seed = seed
        self.rho_noise_std = rho_noise_std

        self.belief = None
        self.state = None

        self.pomdpx_filename = None
        self.despot_eval = []
