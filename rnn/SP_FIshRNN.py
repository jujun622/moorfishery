import torch.nn as nn
import torch
from sklearn.utils import check_random_state
from .FishModels import Surplus, catch_model
from .FishRNN import FishRNN
import numpy as np
import matplotlib.pyplot as plt


class SurplusRNN(FishRNN):
    def __init__(self, rho=1.5, K=10000., B0=5000., q=0.001, m=2):
        super().__init__()

        # self.rho = nn.Parameter(torch.tensor(rho))
        # self.K = nn.Parameter(torch.tensor(K))
        # self.B0 = nn.Parameter(torch.tensor(B0))
        # self.q = nn.Parameter(torch.tensor(q))
        # self.m = m

    def forward(self, x, x2, noise_std=0.0, random_state=None):
        pass

    @staticmethod
    def plot_growth_model(rho, K, B0, ax, label='BH'):
        b0 = np.linspace(0, K, 1000)
        b1_bh = Surplus(rho,  K, b0)
        plt.plot(b0, b1_bh, label=label)
        ax.plot()

    def init_params(self, efforts, catches, normalize=True, random_init=True, random_state=None):
        self.Escale, self.Cscale, Emax, Cmax = self.get_normalize_scale(efforts, catches, normalize)

        efforts = efforts / self.Escale
        # catches = catches / (2 * self.Cscale)
        catches = catches / self.Cscale
        catches = torch.tensor(catches)

        random = check_random_state(random_state)

        if random_init:
            jitter = np.exp(random.normal(0, 0.05, 4))
        else:
            jitter = np.ones(4)

        # self.rho.data.fill_(1. * jitter[0])
        # self.K.data.fill_((Cmax / self.Cscale * jitter[1]) *2)
        # self.B0.data.fill_(Cmax / self.Cscale * jitter[2])
        # self.q.data.fill_((self.Escale / Emax * jitter[3]) /2)
        self.rho.data.fill_(jitter[0])
        self.K.data.fill_(Cmax / self.Cscale * jitter[1])
        self.B0.data.fill_(Cmax / self.Cscale * jitter[2])
        self.q.data.fill_(self.Escale / Emax * jitter[3])

        return efforts, catches

    def get_params(self, normalized=True, param_index=None):

        params = {'rho': self.rho, 'K': self.K, 'B0': self.B0, 'q': self.q}

        rho = self.rho.data.item()
        K = self.K.data.item() * self.Cscale #* 2
        B0 = self.B0.data.item() * self.Cscale #* 2
        q = self.q.data.item() / self.Escale

        unnormalized_params = {'rho': rho, 'K': K, 'B0': B0, 'q': q}

        if normalized:
            retParams = params
        else:
            retParams = unnormalized_params

        if param_index is not None:
            return list(retParams.values())[param_index]
        return retParams


class SurplusRNN_gac(SurplusRNN):
    def __init__(self, rho=1.5, K=10000., B0=5000., q=0.001, m=2):
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(rho))
        self.K = nn.Parameter(torch.tensor(K))
        self.B0 = nn.Parameter(torch.tensor(B0))
        self.q = nn.Parameter(torch.tensor(q))
        self.m = m

    # grow/catch at the same time
    def forward(self, x1, x2, noise_std=0.0, random_state=None):
        catches = torch.zeros(len(x1))
        b = self.B0

        random = check_random_state(random_state)
        for i in range(len(x1)):
            # print('b', b)
            catches[i] = catch_model(self.q, x1[i], b)
            b = (Surplus(self.rho, self.K, b, self.m) - catches[i]) * np.exp(random.normal(0.0, noise_std))
        return catches

    @staticmethod
    def sample_data(rho, K, B0, q, c, num, m=2, noise_std=0.0, random_state=None):
        '''haven't modified to generate variable efforts'''
        r = rho

        biomasses = []
        efforts = []
        catches = []

        random = check_random_state(random_state)
        b = B0
        for i in range(num):
            biomasses.append(b)

            effort = c + random.normal(0.0, 3)
            efforts.append(effort)

            catch = catch_model(q, effort, b)
            catches.append(catch)

            b = (Surplus(r, K, b, m) - catch) * np.exp(random.normal(0.0, noise_std))

        return np.array(biomasses), np.array(efforts), np.array(catches)

    @staticmethod
    def predict(rho, K, B0, q, efforts, nsample=1, noise_std=0.0, random_state=None, m=2):
        random = check_random_state(random_state)

        shape = (nsample, len(efforts))

        catches = np.zeros(shape)
        biomasses = np.zeros(shape)

        b = B0*np.ones(len(catches))
        for i in range(len(efforts)):
            biomasses[:, i] = b
            catches[:, i] = catch_model(q, efforts[i], b)
            b = (Surplus(rho, K, b, m) - catches[:, i]) * np.exp(random.normal(0.0, noise_std, len(catches[:, i])))

        return catches, biomasses

    @staticmethod
    def expected_mse(rho, K, B0, q, efforts, catches, nsample=1, noise_std=0.0, random_state=None, mask=None,
                      retStd=False, m=2):

        output, _ = SurplusRNN_gac().predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                                                 random_state=random_state, m=m)

        if mask is not None:
            mse = np.mean(([output[i][tuple(mask)] for i in range(len(output))] - catches[tuple(mask)]) ** 2, axis=1)
        else:
            mse = np.mean((output - catches) ** 2, axis=1)

        if retStd:
            return np.mean(mse), np.std(mse) / np.sqrt(nsample)
        return np.mean(mse)

    @staticmethod
    def plot_learnt_data(rho, K, B0, q, catches, biomasses, efforts, nsample=1, noise_std=0.0, random_state=None, m=2):
        new_catches, new_biomasses = SurplusRNN_gac.predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                            random_state=random_state, m=m)
        plt.plot(np.arange(1, len(efforts)+1), efforts, label = 'efforts')
        plt.plot(np.arange(1, len(efforts)+1), catches, label = 'true catches')
        plt.plot(np.arange(1, len(efforts)+1), biomasses, label='true biomasses')
        plt.plot(np.arange(1, len(efforts)+1), new_catches[0], label='learned catches')
        plt.plot(np.arange(1, len(efforts)+1), new_biomasses[0], label='learned biomasses')
        plt.legend()
        plt.show()


class SurplusRNN_gtc(SurplusRNN):
    def __init__(self, rho=1.5, K=10000., B0=5000., q=0.001, m=2):
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(rho))
        self.K = nn.Parameter(torch.tensor(K))
        self.B0 = nn.Parameter(torch.tensor(B0))
        self.q = nn.Parameter(torch.tensor(q))
        self.m = m

    # first grow then catch
    def forward(self, x, x2, noise_std=0.0, random_state=None):
        catches = torch.zeros(len(x))
        b = self.B0

        random = check_random_state(random_state)
        catches[0] = 0
        for i in range(len(x)-1):
            tmp_b = Surplus(self.rho, self.K, b, self.m)
            catches[i+1] = catch_model(self.q, x[i], tmp_b)
            b = (tmp_b - catches[i+1]) * np.exp(random.normal(0.0, noise_std))

        return catches

    @staticmethod
    def sample_data(rho, K, B0, q, c, num, m=2, noise_std=0.0, random_state=None):
        r = rho
        biomasses = []
        efforts = []
        catches = []

        random = check_random_state(random_state)
        b = B0
        catches.append(0)
        biomasses.append(b)
        for i in range(num):

            effort = c + random.normal(0.0, 3)
            efforts.append(effort)

            if (i + 1) < num:
                tmp_b = Surplus(r, K, b, m)
                catch = catch_model(q, effort, tmp_b)
                catches.append(catch)

                b = (tmp_b - catch) * np.exp(random.normal(0.0, noise_std))
                biomasses.append(b)

        return np.array(biomasses), np.array(efforts), np.array(catches)

    @staticmethod
    def predict(rho, K, B0, q, efforts, nsample=1, noise_std=0.0, random_state=None, m=2):
        random = check_random_state(random_state)

        shape = (nsample, len(efforts))
        catches = np.zeros(shape)
        biomasses = np.zeros(shape)

        b = B0 * np.ones(len(catches))
        catches[:, 0] = 0
        biomasses[:, 0] = b
        for i in range(len(efforts) - 1):
            tmp_b = Surplus(rho, K, b, m)
            catches[:, i + 1] = catch_model(q, efforts[i], tmp_b)
            biomasses[:, i + 1] = (tmp_b - catches[:, i + 1]) * np.exp(random.normal(0, noise_std, len(catches[:, i + 1])))
            b = biomasses[:, i + 1]
        return catches, biomasses

    @staticmethod
    def expected_mse(rho, K, B0, q, efforts, catches, nsample=1, noise_std=0.0, random_state=None,
                     retStd=False, m=2):

        output, _ = SurplusRNN_gtc.predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                            random_state=random_state, m=m)

        mse = np.mean((output - catches) ** 2, axis=1)

        if retStd:
            return np.mean(mse), np.std(mse) / np.sqrt(nsample)
        return np.mean(mse)

    @staticmethod
    def plot_learnt_data(rho, K, B0, q, catches, biomasses, efforts, nsample=1, noise_std=0.0, random_state=None, m=2):
        new_catches, new_biomasses = SurplusRNN_gtc.predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                            random_state=random_state, m=m)
        plt.plot(np.arange(1, len(efforts)+1), efforts, label = 'efforts')
        plt.plot(np.arange(1, len(efforts)+1), catches, label = 'true catches')
        plt.plot(np.arange(1, len(efforts)+1), biomasses, label='true biomasses')
        plt.plot(np.arange(1, len(efforts)+1), new_catches[0], label='learned catches')
        plt.plot(np.arange(1, len(efforts)+1), new_biomasses[0], label='learned biomasses')
        plt.legend()
        plt.show()
