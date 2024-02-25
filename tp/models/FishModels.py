from sklearn.utils import check_random_state
import numpy as np

# def BevertonHolt(x, K, rho, a, noise=0):
#     return (rho * K * (1.0 - a) * x) / (K + (rho - 1.0) * x)

def BevertonHolt(x, K, rho):
    return rho * x * K /((rho-1)*x + K)


def Surplus(x, K, rho, m=2):
    return x + rho/(m-1)*x*(1-(x/K)**(m-1))


def catch_model(a, x, noise=0.0):   # x can be the current biomass/next biomass
    return a*x


def BevertonHolt_gac(x, K, rho, a):
    return BevertonHolt(x, K, rho) - catch_model(a,x)      # growth/harvest at the same time


def Surplus_gac(x, K, rho, a, m=2):
    return Surplus(x, K, rho, m) - catch_model(a,x)


def BevertonHolt_gtc(x,K,rho,a):
    x1 = BevertonHolt(x, K, rho)
    return x1 - catch_model(a,x1)


def Surplus_gtc(x, K, rho, a, m=2):
    x1 = Surplus(x, K, rho, m)
    return  x1 - catch_model(a,x1)


def catch_from_updated_biomass(a, x):
    return a * x / (1 - a)



#
# def catch_from_updated_biomass(a, x):
#     return a * x / (1 - a)
#
#
# def catch_for_bhm(x, a, noise=0.1):    # x is current biomass
#     return x * np.exp(np.random.normal(0, noise, len(x))) * a         # growth/harvest at the same time
#     # return BevertonHolt(x, K, rho) * a      # growth first then harvest
#     # return a * (rho * K * x) / (K + (rho - 1.0) * x)      #Zhihao's code (growth first then harvest)
#
#
# def catch_for_surplus(x, a, noise=0.1):
#     # return Surplus(rho, K, x, m) * a      #growth first then harvest
#     # noise_term = np.exp(np.random.normal(0, noise))       # noise term for growth/harvest at the same time
#     return x * np.exp(np.random.normal(0, noise, len(x))) * a       # growth/harvest at the same time


def find_surplus_max_bio(r, K, m=2):
    numerator = 1 + r / (m - 1)
    denonimator = m * r / (m - 1) * (1 / K) ** (m - 1)
    x = (numerator / denonimator) ** (1 / (m - 1))

    max_bio = Surplus(r, K, x, m)
    return max_bio


def index2state(idx, num_state, K):
    return np.array([idx, idx+1])*1.0*K/num_state


def index2action(idx, num_action, K):
    return np.array([idx, idx+1])*1.0*K/num_action


def index2obs(idx, num_obs, K):
    return np.array([idx, idx+1])*1.0*K/num_obs
