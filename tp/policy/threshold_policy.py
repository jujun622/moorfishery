import numpy as np
import matplotlib.pyplot as plt
# from tp.utilities.simulation import idx2state

import functools
print = functools.partial(print, flush=True)


class POMDPThresholdPolicy:
    def __init__(self, A, thresholds, K):
        self.A = A.copy()
        self.thresholds = np.array(thresholds)
        self.K = K      # here is K in the model that used to train this policy
        self.normalized = False
        # self.belief_action_dict = {}
        self.eptbio_action_dict = {}
        self.eptbioratio_action_dict = {}

    # def update(self, η=None, η_index=-1):
    #     self.ψ.clear()
    #     if η != None:
    #         self.thresholds[η_index] = η

    def __str__(self):
        return str(self.thresholds)
        # return self.thresholds

    def normalize(self):
        if not self.normalized:
            self.thresholds = self.thresholds / self.K
            self.normalized = True
            print('Now the threshold policy is normalized')

    def unnormalize(self):
        if self.normalized:
            self.thresholds = self.thresholds * self.K
            self.normalized = False
            print('Now the threshold policy is back to unnormalized')

    def expected_biomass(self, b, K):
        num_state = len(b)
        gap = K/num_state       # both training and using the policy are on the simulator models,
                                # and the states in the simulator models are split averagely
        mean = []
        for i in range(len(b)):
            mean.append(b[i] * (gap * i + gap/2))
        return np.sum(mean)

    def std(self, b, K):
        num_state = len(b)
        gap = K/num_state
        mean = []
        for i in range(len(b)):
            mean.append(b[i] * (gap * i + gap/2))

        return np.sqrt(np.sum(mean) - self.expected_biomass(b) ** 2)

    def expected_biomass_ratio(self, b, K):
        exp_bio = self.expected_biomass(b,K)
        return exp_bio / K

    def get_action(self, b, K):

        if self.normalized:
            val = self.expected_biomass_ratio(b, K)
            # print('val',val)

            if self.eptbioratio_action_dict.get(val) is None:

                for i in range(len(self.thresholds)):
                    if i == 0:
                        η = self.thresholds[0]
                        if val < η:
                            self.eptbioratio_action_dict[val] = self.A[0]
                            break

                    elif i == np.max(range(len(self.thresholds))):
                        η = self.thresholds[i]
                        if val >= η:
                            self.eptbioratio_action_dict[val] = self.A[i + 1]
                            break
                        else:
                            η1 = self.thresholds[i - 1]
                            if η1 <= val < η:
                                self.eptbioratio_action_dict[val] = self.A[i]
                            break

                    else:
                        η1 = self.thresholds[i - 1]
                        η2 = self.thresholds[i]
                        if η1 <= val < η2:
                            self.eptbioratio_action_dict[val] = self.A[i]
                            break

            return self.eptbioratio_action_dict[val]

        else:
            val = self.expected_biomass(b, K)
            # print('val', val)
            # print('thresholds', self.thresholds)

            # if self.eptbio_action_dict.get(tuple(b)) is None:
            if self.eptbio_action_dict.get(val) is None:

                for i in range(len(self.thresholds)):
                    if i == 0:
                        # print('1')
                        η = self.thresholds[0]
                        if val < η:
                            # print('2')
                            self.eptbio_action_dict[val] = self.A[0]
                        elif len(self.thresholds) == 1:
                            # print('3')
                            self.eptbio_action_dict[val] = self.A[1]

                    elif i == np.max(range(len(self.thresholds))):
                        η = self.thresholds[i]
                        if val >= η:
                            # print('4')
                            self.eptbio_action_dict[val] = self.A[i+1]
                            break
                        else:
                            # print('5')
                            η1 = self.thresholds[i-1]
                            if η1 <= val < η:
                                # print('6')
                                self.eptbio_action_dict[val] = self.A[i]
                            break

                    else:
                        # print('7')
                        η1 = self.thresholds[i - 1]
                        η2 = self.thresholds[i]
                        if η1 <= val < η2:
                            # print('8')
                            self.eptbio_action_dict[val] = self.A[i]
                            break

            return self.eptbio_action_dict[val]

    def plot_policy(self, title=None):
        thresholds = list(self.thresholds)
        if self.normalized:
            xlabel = 'ratio of expected biomass and carrying capacity'
            thresholds.append(1)
        else:
            xlabel = 'expected biomass'
            thresholds.append(self.K)
        for i in range(len(thresholds)):
            if i == 0:
                plt.hlines(i, 0, thresholds[0], linestyles='-')
            else:
                plt.hlines(i, thresholds[i-1], thresholds[i], linestyles='-')
        plt.yticks(list(range(len(self.A))))
        plt.ylabel('action')
        plt.xlabel(xlabel)
        if title is not None:
            plt.title(title)
        plt.show()

