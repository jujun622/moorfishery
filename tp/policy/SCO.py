import numpy as np
import timeit

import functools
print = functools.partial(print, flush=True)


def SCO(f, ς, w, thd_num, K, num_sample_thds, MaxTry=5, UpdateTime=10):
    """ Application of SCO for solving a threshold policy with multiple thresholds. """
    time_start = timeit.default_timer()

    B1 = [np.array([0 for i in range(thd_num)]), np.array([K for i in range(thd_num)])]

    B1_samples = [np.random.uniform(B1[0], B1[1]) for _ in range(num_sample_thds)]
    B1_samples_sorted = [np.sort(B1_samples[i]) for i in range(num_sample_thds)]

    Y_SET = {0: np.array([[B1_samples_sorted[i][j] for j in range(len(B1_samples_sorted[i]))] for i in range(num_sample_thds)])}
    # print(Y_SET)

    # X_best: all learnt parameters, V: all best values
    X_SET = {}
    X_best = {}
    V = {}
    t = 0
    N_elite = int(np.ceil(num_sample_thds * ς))

    R = np.zeros(N_elite)
    σ = {}
    I_SET = np.arange(N_elite)

    B_i = np.concatenate((np.ones(num_sample_thds - N_elite), np.zeros(2 * N_elite - num_sample_thds)))

    best = (None, -np.inf)
    while True:
        print(f"ITERATION {t}...")
        S_X = np.array([f(X) for X in Y_SET[t]])
        # print('S_X', S_X)
        idx = np.argsort(S_X)[::-1][:N_elite]
        S_X = S_X[idx]
        X = [Y_SET[t][i] for i in idx]
        X_SET[t + 1] = X
        V[t + 1] = S_X[0]
        X_best[t + 1] = X[0].copy()

        np.random.shuffle(B_i)

        for i in range(N_elite):
            # R[i] = int(np.floor(N / N_elite) + B_i[i])  # random splitting factor
            R[i] = 2    #use a constant splitting factor
            Y = X[i].copy()
            Y_dash = Y.copy()

            for j in range(int(R[i])):
                I = np.random.choice(I_SET[I_SET != i])
                σ[i] = w * np.abs(X[i] - X[I])
                μ = np.random.permutation(thd_num)
                # print('μ:', μ)

                for k in range(thd_num):
                    for Try in range(MaxTry):
                        Z = np.random.normal()

                        # updating one value in the thresholds
                        Y_dash[μ[k]] = max(0, min(K, Y[μ[k]] + σ[i][μ[k]] * Z))

                        # -->> resort the thresholds here
                        Y_dash = np.sort(Y_dash)

                        if f(Y_dash) > f(Y):
                            Y = Y_dash.copy()
                            break

                if Y_SET.get(t + 1) == None:
                    Y_SET[t + 1] = []
                Y_SET[t + 1].append(Y.copy())

        t = t + 1
        if V[t] > best[1]:
            best = (X_best[t], V[t])
            print(f"Best threshold:  {best[0]}")
            print(f"Best value:      {best[1]}")

        if t == UpdateTime:
            break

    time_stop = timeit.default_timer()
    print('Finish training tp using SCO algorithm:', time_stop-time_start)


    return best
