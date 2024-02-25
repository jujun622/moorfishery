import torch.nn as nn
import torch
from sklearn.utils import check_random_state
from .FishModels import BevertonHolt, catch_model, simulate_effort
from .FishRNN import FishRNN
import numpy as np
import matplotlib.pyplot as plt
import random as random0


class BevertonHoltRNN(FishRNN):
    def __init__(self, rho=1.5, K=10000., B0=5000., q=0.001):
        super().__init__()

        # self.rho = nn.Parameter(torch.tensor(rho))
        # self.K = nn.Parameter(torch.tensor(K))
        # self.B0 = nn.Parameter(torch.tensor(B0))
        # self.q = nn.Parameter(torch.tensor(q))

    def forward(self, x, x2, noise_std=0.0, random_state=None):    #first grow then catch
        pass

    @staticmethod
    def plot_growth_model(rho, K, B0, ax, label='BH'):
        b0 = np.linspace(0, K, 1000)
        b1_bh = BevertonHolt(rho,  K, b0)
        plt.plot(b0, b1_bh, label=label)
        ax.plot()


    def init_params(self, efforts, catches, normalize=True, random_init=True, random_state=None):
        self.Escale, self.Cscale, Emax, Cmax = self.get_normalize_scale(efforts, catches, normalize)

        efforts = efforts / self.Escale
        # catches = catches / (3 * self.Cscale)
        catches = catches / self.Cscale
        catches = torch.tensor(catches)

        random = check_random_state(random_state)

        if random_init:
            jitter = np.exp(random.normal(0, 0.05, 4))
        else:
            jitter = np.ones(4)

        # self.rho.data.fill_(2. * jitter[0])
        # self.K.data.fill_(Cmax / self.Cscale * jitter[1] *2)
        # self.B0.data.fill_(Cmax / self.Cscale * jitter[2])
        # self.q.data.fill_((self.Escale / Emax * jitter[3])/3)
        self.rho.data.fill_(jitter[0])
        self.K.data.fill_(Cmax / self.Cscale * jitter[1])
        self.B0.data.fill_(Cmax / self.Cscale * jitter[2])
        self.q.data.fill_(self.Escale / Emax * jitter[3])

        return efforts, catches

    def get_params(self, normalized=True, param_index=None):

        params = {'rho': self.rho, 'K': self.K, 'B0': self.B0, 'q': self.q}

        rho = self.rho.data.item()
        K = self.K.data.item() * self.Cscale #*3
        B0 = self.B0.data.item() * self.Cscale #*3
        q = self.q.data.item() / self.Escale

        unnormalized_params = {'rho': rho, 'K': K, 'B0': B0, 'q': q}

        if normalized:
            retParams = params
        else:
            retParams = unnormalized_params

        if param_index is not None:
            return list(retParams.values())[param_index]
        return retParams

    # @staticmethod
    # def predict(rho, K, B0, q, efforts, nsample=1, noise_std=0.0, random_state=None, m=2):
    #     pass


class BevertonHoltRNN_gac(BevertonHoltRNN):
    def __init__(self, rho=1.5, K=10000., B0=5000., q=0.001):
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(rho))
        self.K = nn.Parameter(torch.tensor(K))
        self.B0 = nn.Parameter(torch.tensor(B0))
        self.q = nn.Parameter(torch.tensor(q))

    def forward(self, x1, x2, noise_std=0.0, random_state=None):         # grow/catch at the same time
        catches = torch.zeros(len(x1))
        b = self.B0

        random = check_random_state(random_state)
        for i in range(len(x1)):
            catches[i] = catch_model(self.q, x1[i], b)
            b = (BevertonHolt(self.rho, self.K, b) - catches[i]) * np.exp(random.normal(0.0, noise_std))
        return catches

    @staticmethod
    def sample_data(rho, K, B0, q, c=10, num=50, noise_std=0.0, random_state=None, num_missing_years=0, effort_mode='constant'):

        dic = {}
        dic['num missing years'] = num_missing_years
        dic['complete data'] = {}

        biomasses = []
        efforts = []
        catches = []

        random = check_random_state(random_state)
        print('effort mode in sample_data', effort_mode)
        
        efforts = simulate_effort(t=num, turning_t1=int(0.7*num), turning_t2=int(0.8*num), init_x=c, max_x=0.9*1/q, 
                              mode=effort_mode, noise=0.05, random=random, c=c)

        b = B0
        for i in range(num):
            biomasses.append(b)

            # effort = c + random.normal(0.0, 3)
            # efforts.append(effort)

            catch = catch_model(q, efforts[i], b)
            catches.append(catch)

            b = (BevertonHolt(rho, K, b) - catch) * np.exp(random.normal(0.0, noise_std))

        dic['complete data']['biomasses'] = np.array(biomasses)
        dic['complete data']['efforts'] = np.array(efforts)
        dic['complete data']['catches'] = np.array(catches)

        if num_missing_years == 0:
            dic['for model training'] = dic['complete data']
            return dic
        else:
            sample_missing_years = random.choice(num, num_missing_years)

            print('missing year:',sample_missing_years)
            dic['missing years'] = sample_missing_years
            dic['incomplete data'] = {}

            b1 = biomasses.copy()
            e1 = efforts.copy()
            c1 = catches.copy()

            for year in sample_missing_years:
                b1[year] = float("NaN")
                e1[year] = float('NaN')
                c1[year] = float('NaN')

            dic['incomplete data']['biomasses'] = np.array(b1)
            dic['incomplete data']['efforts'] = np.array(e1)
            dic['incomplete data']['catches'] = np.array(c1)

            dic['for model training'] = dic['incomplete data']
            return dic

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
            b = (BevertonHolt(rho, K, b) - catches[:, i]) * np.exp(random.normal(0.0, noise_std, len(catches[:,i])))

        return catches, biomasses

    @staticmethod
    def expected_mse(rho, K, B0, q, efforts, catches, nsample=1, noise_std=0.0, random_state=None, mask=None,
                    retStd=False, m=2):

        output, _ = BevertonHoltRNN_gac().predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
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
        new_catches, new_biomasses = BevertonHoltRNN_gac.predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                            random_state=random_state, m=m)
        plt.plot(np.arange(1, len(efforts)+1), efforts, label = 'efforts')
        plt.plot(np.arange(1, len(efforts)+1), catches, label = 'true catches')
        plt.plot(np.arange(1, len(efforts)+1), biomasses, label='true biomasses')
        plt.plot(np.arange(1, len(efforts)+1), new_catches[0], label='learned catches')
        plt.plot(np.arange(1, len(efforts)+1), new_biomasses[0], label='learned biomasses')
        plt.legend()
        plt.show()


class BevertonHoltRNN_gtc(BevertonHoltRNN):
    def __init__(self, rho=1.5, K=10000., B0=5000., q=0.001):
        super().__init__()

        self.rho = nn.Parameter(torch.tensor(rho))
        self.K = nn.Parameter(torch.tensor(K))
        self.B0 = nn.Parameter(torch.tensor(B0))
        self.q = nn.Parameter(torch.tensor(q))

    def forward(self, x, x2, noise_std=0.0, random_state=None):    #first grow then catch
        catches = torch.zeros(len(x))
        b = self.B0

        random = check_random_state(random_state)
        catches[0] = 0
        for i in range(len(x)-1):
            # print('biomass',b)
            tmp_b = BevertonHolt(self.rho, self.K, b)
            catches[i+1] = catch_model(self.q, x[i], tmp_b)
            b = (tmp_b - catches[i+1]) * np.exp(random.normal(0.0, noise_std))

        return catches

    @staticmethod
    def sample_data(rho=7.625, K=10000, B0=5000, q=0.005, c=10, num=50, noise_std=0.0, random_state=None):
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
                tmp_b = BevertonHolt(rho, K, b)
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
            tmp_b = BevertonHolt(rho, K, b)
            catches[:, i + 1] = catch_model(q, efforts[i], tmp_b)
            biomasses[:, i + 1] = (tmp_b - catches[:, i + 1]) * np.exp(random.normal(0, noise_std, len(catches[:, i + 1])))
            b = biomasses[:, i + 1]
        return catches, biomasses

    @staticmethod
    def expected_mse(rho, K, B0, q, efforts, catches, nsample=1, noise_std=0.0, random_state=None,
                     retStd=False, m=2):

        output, _ = BevertonHoltRNN_gtc.predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                            random_state=random_state, m=m)

        mse = np.mean((output - catches) ** 2, axis=1)

        if retStd:
            return np.mean(mse), np.std(mse) / np.sqrt(nsample)
        return np.mean(mse)

    @staticmethod
    def plot_learnt_data(rho, K, B0, q, catches, biomasses, efforts, nsample=1, noise_std=0.0, random_state=None, m=2):
        new_catches, new_biomasses = BevertonHoltRNN_gtc.predict(rho, K, B0, q, efforts, nsample=nsample, noise_std=noise_std,
                            random_state=random_state, m=m)
        plt.plot(np.arange(1, len(efforts)+1), efforts, label = 'efforts')
        plt.plot(np.arange(1, len(efforts)+1), catches, label = 'true catches')
        plt.plot(np.arange(1, len(efforts)+1), biomasses, label='true biomasses')
        plt.plot(np.arange(1, len(efforts)+1), new_catches[0], label='learned catches')
        plt.plot(np.arange(1, len(efforts)+1), new_biomasses[0], label='learned biomasses')
        plt.legend()
        plt.show()




