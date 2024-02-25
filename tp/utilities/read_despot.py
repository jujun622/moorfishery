import numpy as np
import matplotlib.pyplot as plt
from ..models import belief, transition, obs_probs, mean_catches


with open('out1_bigt.txt', 'r') as f:
    data = f.readlines()[17:]

# print(type(data[0]))
# print(data[1][10])
# #
# for i in range(len(data[3])):
#     print(i, data[3][i])

# put the lines of actions into a list
actions = []

observations = []
states = []
for i in range(len(data)):
    line = data[i]
    if 'Action' in line:
        actions.append(int(line[11]))
    if 'Observation' in line:
        observations.append(int(line[24]))
    if 'state_1' in line:
        states.append(int(line[10]))

print(actions[0:100])
# rewards = 0
# for i in range(100):
#     r = 0.95**i * mean_catches[states[i]][actions[i]]
#     rewards += r
#     print(rewards)

from collections import Counter
act_nums = Counter(actions)
print(act_nums)


act_rounds = []
obs_rounds = []
for j in range(0,len(actions),100):
    a = actions[j:j+100]
    o = observations[j:j+100]
    act_rounds.append(a)
    obs_rounds.append(o)


b0 = belief
trans_probs = transition
obs_probs = obs_probs


def belief_update(b, a, o, trans_probs, obs_probs):
    obs_prob = obs_probs[:, a, o]
    sum_transition = np.matmul(b, trans_probs[a, :, :])
    sum_obs = sum(obs_prob * sum_transition)
    new_belief = obs_prob * sum_transition / sum_obs
    return np.array(new_belief)


def expected_biomass(b):
    mean = []
    for i in range(len(b)):
        mean.append(b[i] * (1000 * i + 500))
    return np.sum(mean)


# init_m = expected_biomass(b0)
# beliefs = []
# biomass = []
# for j in range(1000):
#     bs = [b0]
#     ms = [init_m]
#     for i in range(99):
#         b1 = belief_update(b0, act_rounds[j][i], obs_rounds[j][i], trans_probs, obs_probs)
#         m1 = expected_biomass(b1)
#         bs.append(b1)
#         ms.append(m1)
#         b0 = b1
#     beliefs.append(bs)
#     biomass.append(ms)

# for i in range(1000):
#     plt.plot(biomass[i], act_rounds[i], 'o')
#     plt.show()

init_m = expected_biomass(b0)
beliefs = []
biomass = []
for j in range(1000):
    b = b0
    beliefs.append(b)
    biomass.append(init_m)
    for i in range(99):
        b1 = belief_update(b, actions[j*100+i], observations[j*100+i], trans_probs, obs_probs)
        m1 = expected_biomass(b1)
        # print('b1',b1)
        # print('m1',m1)
        beliefs.append(b1)
        biomass.append(m1)
        b = b1

# print(actions[200:300])
# print(observations[200:300])
# print(biomass[0:100])




# plt.plot(list(range(100)), biomass[0:100])
# plt.show()

rho1 = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.00495112e-02,
        1.71729816e+02, 7.88820719e+03, 8.66502175e+03, 9.90084956e+03]
rho2 = [ 933.4624406,  2514.77526712, 6124.84743178, 6865.86294778,
         7778.71886025, 8155.20247128, 8267.25379654, 9239.82065146, 9680.55129286]
rho3 = [  146.78808803,  1322.17975668,  2072.39037289,  2091.01367911,
          7210.78290949,  7493.63017743,  9331.27247091, 10000., 10000. ]
rho4 = [ 958.64095962, 1613.02696414, 2098.82731525, 2423.18786592, 4279.9850754, 5310.90275255,
         7745.88016364, 8818.79177103, 9122.60994876]

rho1_new = [ 1714.64951905,  1804.48233283,  2265.05777639,  3395.39936848,
  3598.47593255,  7695.34973986,  8257.10908898,  9834.3357825, 10000.]
# mean: 10379.469
# std: 34.940473051162314

x = list(range(10))
x.reverse()

# plt.plot(rho1_new + [10000], x, '.-', label='opposite threshold policy')

plt.plot(biomass, actions, 'o', label = 'DESPOT', color='orange')

for i in range(len(rho1_new)):
    if i == len(rho1_new)-1:
        plt.hlines(x[i], rho1_new[i], 10000, linestyles='-', label='opposite threshold policy')
    else:
        plt.hlines(x[i], rho1_new[i], rho1_new[i+1], linestyles='-')


plt.yticks(list(range(10)))
plt.xlabel('expected biomass')
plt.ylabel('action')
plt.legend(loc = 'upper right')
plt.title('POMDP model with rho=1')
plt.show()
