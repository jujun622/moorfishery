import numpy as np
import timeit
import functools
from sklearn.utils import check_random_state

print = functools.partial(print, flush=True)


def CA(f, thd_num, K, random_state=42, GenerationTime=10, search_time=1):
    '''
    Coordinate ascent with uniform initialisation (left to right).
    :param f: evaluation function
    :param thd_num: threshold dimensions
    :param K: carrying capacity
    :param GenerationTime: number of generations

    :return: optimal thresholds and corresponding evaluation values
    '''
    start = timeit.default_timer()
    gen_time = GenerationTime

    # initial samples for the first generation
    init = [i * (K / (thd_num + 2)) for i in range(thd_num + 2)]
    #     init[0] = 0
    init_value = f(init[1:-1])

    best_values = []
    best_thds = []

    random = check_random_state(random_state)
    seeds = []
    for i in range(gen_time):
        thd_best = init.copy()
        v_best = init_value
        #         print('init', i, thd_best, v_best)

        seed = random.randint(0, 10000)
        seeds.append(seed)
        random = check_random_state(seed)

        for j in range(1, thd_num + 2 - 2):

            for k in range(search_time):
                new_thd = thd_best.copy()

                new_item = random.uniform(thd_best[j - 1], thd_best[j])
                new_thd[j] = new_item

                new_value = f(new_thd[1:-1])

                #                 print('')
                #                 print(i,j,k,new_thd,new_value)
                if new_value >= v_best:
                    thd_best = new_thd
                    v_best = new_value
            #                 print('best',thd_best,v_best)

            best_thds.append(thd_best)
            best_values.append(v_best)

        print(i, '-th generation:', thd_best, v_best)

    print('seeds:', seeds)
    v = np.max(best_values)
    thd = best_thds[np.argmax(best_values)]

    end = timeit.default_timer()
    print('Finish training tp using Coodinate Ascent algorithm:', end - start)


    return [thd[1:-1], v]



# oldest version for coordinate ascent (right to left)
def GA2(f, thd_num, K, random_state=42, GenerationTime=10):
    '''

    :param f: evaluation function
    :param thd_num: threshold dimensions
    :param K: carrying capacity
    :param GenerationTime: number of generations

    :return: optimal thresholds and corresponding evaluation values
    '''

    gen_time = GenerationTime

    # initial samples for the first generation
    init = [0 for _ in range(thd_num)]
    init[-1] = K
    init_value = f(init)

    best_values = []
    best_thds = []

    random = check_random_state(random_state)
    seeds = []
    for i in range(gen_time):
        thd_best = init.copy()
        v_best = init_value
        #         print('init', i, thd_best, v_best)

        seed = random.randint(0, 10000)
        seeds.append(seed)
        random = check_random_state(seed)

        for j in range(thd_num - 2, 0, -1):
            new_thd = thd_best

            new_item = random.uniform(thd_best[j], thd_best[j + 1])
            new_thd[j] = new_item

            new_value = f(new_thd)

            if new_value >= v_best:
                thd_best = new_thd
                v_best = new_value
            else:
                break

        best_thds.append(thd_best)
        best_values.append(v_best)

        print(i, '-th generation:', thd_best, v_best)

    print('seeds:', seeds)
    v = np.max(best_values)
    thd = best_thds[np.argmax(best_values)]

    return [thd, v]
