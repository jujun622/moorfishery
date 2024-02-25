from sklearn.utils import check_random_state
import numpy as np

def BevertonHolt(rho, K, B):
    return rho*B*K/((rho-1)*B + K)


def Surplus(rho, K, B, m=2):
    return B + rho/(m-1)*B*(1-(B/K)**(m-1))


def catch_model(q, e, B):
    return q*e*B


def simulate_effort(random, t=50, turning_t1=30, turning_t2=40, init_x=10, max_x=200, mode='constant', noise=0.05, c=10):
    
    if mode == 'no decrease':
        print('qwerty')
        e1 = [(max_x - init_x)/turning_t1 * i + init_x for i in range(turning_t1)]
        e2 = [max_x for i in range(t-turning_t1)]
        e = e1 + e2

        effort = [e[j] * np.exp(random.normal(0.0, noise)) for j in range(t)]
        effort = [effort[j] if effort[j]<=max_x else max_x for j in range(t)]
        
    elif mode == 'with decrease':
        
        e1 = [(max_x - init_x)/turning_t1 * i + init_x for i in range(turning_t1)]
        e2 = [max_x for i in range(turning_t2-turning_t1)]
        e3 = [(- max_x)/(t - turning_t2) * i + (max_x / (t - turning_t2) * t) for i in range(turning_t2, t)]
        e = e1 + e2 + e3
        
        effort = [e[j] * np.exp(random.normal(0.0, noise)) for j in range(t)]
        effort = [effort[j] if effort[j]<=max_x else max_x for j in range(t)]

    elif mode == 'constant':
        
        effort = []
        for i in range(t):
            e = c + random.normal(0.0, 3)
            effort.append(e)
    
    return effort



