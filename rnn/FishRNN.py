import torch.nn as nn
import torch
from sklearn.utils import check_random_state
from .FishModels import BevertonHolt, Surplus, catch_model
import numpy as np
import json


class FishRNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, noise=False):
        pass

    def init_params(self, efforts, catches, normalize=True, random_init=True, random_state=None):
        pass

    def get_params(self, normalized=True, param_index=None):
        pass

    def print_gradient(self):
        params = self.get_params(normalized=True)
        uparams = self.get_params(normalized=False)
        for name in params:
            # print('%5s: raw_val=%10.3g; grad=%10.3g; norm_val=%10.3g;' %
            #       (name, params[name], params[name].grad, uparams[name]))
            # print full precision
            print('%5s: raw_val=%s; grad=%s; norm_val=%s;' %
                  (name, json.dumps(params[name].item()), json.dumps(params[name].grad.item()), json.dumps(uparams[name])))

    def print_loss(self, epoch, loss):
        #print('[%4d] loss=%.2f' % (epoch + 1, loss * self.Cscale ** 2))
        # print full precision
        print('[%4d] loss=%s' % (epoch + 1, json.dumps((loss * self.Cscale ** 2).item())))

        # params = self.get_params(normalized=True)
        # uparams = self.get_params(normalized=False)
        # for name in params:
        #     print('%5s: raw_val=%10.3g; grad=%10.3g; norm_val=%10.3g;' %
        #           (name, params[name], params[name].grad, uparams[name]))

    @staticmethod
    def get_normalize_scale(efforts, catches, normalize=True):
        Emax, Cmax = np.max(efforts), np.max(catches)
        if normalize:
            Escale = Emax
            Cscale = Cmax
        else:
            Escale, Cscale = 1, 1
        return Escale, Cscale, Emax, Cmax

'''
class BevertonHoltRNN(FishRNN):
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
            b = (tmp_b - catches[i]) * np.exp(random.normal(0.0, noise_std))

        return catches
    # def forward(self, x1, x2, noise_std=None, random_state=None):         # grow/catch at the same time
    #     catches = torch.zeros(len(x1))
    #     b = self.B0
    #
    #     random = check_random_state(random_state)
    #     for i in range(len(x1)):
    #         # print('b', b)
    #         catches[i] = catch_model(self.q, x1[i], b)
    #         if noise_std is not None:
    #             b = (BevertonHolt(self.rho, self.K, b) - catches[i]) * np.exp(random.normal(0.0, noise_std))
    #         else:
    #             b = (BevertonHolt(self.rho, self.K, b) - catches[i])
    #     return catches

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
        self.K.data.fill_(2 * Cmax / self.Cscale * jitter[1])
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


class SurplusRNN(FishRNN):
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
            b = (tmp_b - catches[i]) * np.exp(random.normal(0.0, noise_std))

        return catches

    # grow/catch at the same time
    # def forward(self, x1, x2, noise_std=0.0, random_state=None):
    #     catches = torch.zeros(len(x1))
    #     b = self.B0
    #
    #     random = check_random_state(random_state)
    #     for i in range(len(x1)):
    #         # print('b', b)
    #         catches[i] = catch_model(self.q, x1[i], b)
    #         b = (Surplus(self.rho, self.K, b, self.m) - catches[i]) * np.exp(random.normal(0.0, noise_std))
    #         # if noise_std is not None:
    #         #     b = (Surplus(self.rho, self.K, b, self.m) - catches[i]) * np.exp(random.normal(0.0, noise_std))
    #         # else:
    #         #     b = (Surplus(self.rho, self.K, b, self.m) - catches[i])
    #     return catches

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
'''
