from sklearn.utils import check_random_state
import numpy as np


def BevertonHolt(rho, K, B):
    return rho*B*K/((rho-1)*B + K)


def catch_model(q, e, B):
    return q*e*B


def simulate_effort(random, t=50, turning_t1=30, turning_t2=40, init_x=10, max_x=200, noise=0.05):
    
    e1 = [(max_x - init_x)/turning_t1 * i + init_x for i in range(turning_t1)]
    e2 = [max_x for i in range(t-turning_t1)]
    e = e1 + e2

    effort = [e[j] * np.exp(random.normal(0.0, noise)) for j in range(t)]
    effort = [effort[j] if effort[j]<=max_x else max_x for j in range(t)]
    
    return effort


def sample_data(rho, K, B0, q, num=50, noise_std=0.0, random_state=None, num_missing_years=0):

    dic = {}
    dic['num missing years'] = num_missing_years
    dic['complete data'] = {}

    biomasses = []
    efforts = []
    catches = []

    random = check_random_state(random_state)
    print('effort mode in sample_data', effort_mode)
    
    efforts = simulate_effort(t=num, turning_t1=int(0.7*num), turning_t2=int(0.8*num), init_x=c, max_x=0.9*1/q, 
                          noise=0.05, random=random)

    b = B0
    for i in range(num):
        biomasses.append(b)

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


if __name__ == '__main__':
	# generate stochastic complete data when rho=2.0
	rho, K, B0, q = 2.0, 10000, 5000, 0.05
	num=50
	noise_std=0.1
	random_state=42 
	num_missing_years=0

	data_dic = sample_data(rho, K, B0, q, noise_std=noise_std, random_state=random_state, num_missing_years=num_missing_years)