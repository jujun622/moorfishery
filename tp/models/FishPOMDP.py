import numpy as np
from ..utilities import run_despot, generate_pomdpx_file
import os
import xml.dom.minidom

import functools
print = functools.partial(print, flush=True)


class FishPOMDP:
    # def __init__(self, rho, K, B0, num_state, num_action, num_obs, N, seed=345, rho_noise_std=0.1):
    def __init__(self):
        pass

    def discretize_transition(self):
        pass

    def discretize_observation(self):
        pass

    def discretize_reward(self):
        pass

    def belief2state(self, belief):
        gap = self.K / self.num_state
        mean = []
        for i in range(self.num_state):
            mean.append(belief[i] * (gap * i + gap / 2))
        state = np.sum(mean)
        return state

    def belief2stateidx(self, belief):
        state = self.belief2state(belief)
        for k in range(self.num_state):
            sk = self.states[k]
            l, u = sk
            if l <= state < u:
                state_idx = k
                break
        return state_idx

    def reset(self, mode='uniform'):
        if mode == 'uniform':
            self.belief = np.ones(len(self.states)) / len(self.states)
            # return np.ones(len(self.states)) / len(self.states)
        elif mode == 'near extinction':
            b = np.zeros(len(self.states))
            b[0] = 1
            self.belief = b
            # return b
        # self.stateidx = self.belief2stateidx(self.belief)
        self.stateidx = np.random.choice(self.S, p=self.belief)
        return self.belief

    def make(self):
        np.random.seed(self.seed)
        print('seed in {FishPOMDP.make}:', self.seed)
        self.initial_belief = self.reset()
        self.transition_model, self.observation_model = self.discretize_tr_ob()
        print('finish trs, obs')
        self.rewards = self.discretize_reward()
        print('finish reward')

    def update_belief(self, belief, action, obs):
        obs_prob = self.observation_model[:, action, obs]
        sum_transition = np.matmul(belief, self.transition_model[action, :, :])
        sum_obs = sum(obs_prob * sum_transition)
        # if sum_obs == 0:
        #     # print('will have nan')
        #     obs_prob = 0.999 * obs_prob + 0.001 * np.random.uniform(0, 1, self.num_state)
        #     sum_obs = sum(obs_prob * sum_transition)
        new_belief = obs_prob * sum_transition / sum_obs
        return new_belief

    def step(self, action):
        # print('action', action)
        # print('state', self.stateidx)
        # np.random.seed(self.seed)
        # get the next state and update self.state
        self.stateidx = np.random.choice(self.S, p=self.transition_model[action][self.stateidx])

        # get the corresponding reward/observation
        r = self.rewards[self.stateidx][action]
        o = np.random.choice(self.O, p=self.observation_model[self.stateidx][action])

        # update self.belief synchronously
        self.belief = self.update_belief(self.belief, action, o)

        return r, o

    def generate_pomdpx(self, path, info):
        filename = generate_pomdpx_file.generate_pomdpx_file(self.num_state, self.num_action, self.num_obs,
                                        self.initial_belief, self.transition_model, self.observation_model, self.rewards,
                                        self.rho, self.K, self.seed, self.rho_noise_std, path, info)
        self.pomdpx_filename = filename
        return filename

    def get_pomdpx_file(self, path, info):
        if self.pomdpx_filename is None:
            self.generate_pomdpx(path, info)
        else:
            print('This POMDP model has been generated pomdpx file:', self.pomdpx_filename)

    def get_despot_result(self, t, runs, simlen, path, despot, info=None):
        # rel_filename = self.pomdpx_filename.split('/')[-1]
        # pomdpx_name = './pomdpx_files/'+rel_filename

        txt_name = '%s_t%.3f_runs%d_simlen%d.txt' \
                   % (info, t, runs, simlen)
        txt_name = os.path.join(path, txt_name)
        print('despot rst file name:', txt_name)
        self.despot_filename = txt_name

        despot_rslt = run_despot.run_despot(
            pomdpx=self.pomdpx_filename,
            save_name= txt_name,
            despot=despot,
            options=['-t', str(t), '--runs', str(runs), '--simlen', str(simlen)])

        return despot_rslt

    def despot_evaluate(self, simulator_pomdpx, t, runs, simlen, path, despot, info=None):
        # rel_filename = self.pomdpx_filename.split('/')[-1]  #world model
        # pomdpx_name = './pomdpx_files/'+rel_filename

        # rel_simulator_filename = simulator.split('/')[-1]
        # simulator_pomdpx = './pomdpx_files/'+rel_simulator_filename

        txt_name = '%s_t%.3f_runs%d_simlen%d.txt' \
                   % (info, t, runs, simlen)
        txt_name = os.path.join(path, txt_name)
        print('despot rst file name:', txt_name)
        self.despot_eval.append(txt_name)

        despot_rslt = run_despot.run_despot_evaluate(
            simulator_pomdpx= simulator_pomdpx,
            world_pomdpx=self.pomdpx_filename,
            save_name= txt_name,
            despot=despot,
            options=['-t', str(t), '--runs', str(runs), '--simlen', str(simlen)])

        return despot_rslt

