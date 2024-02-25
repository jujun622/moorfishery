import torch
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
# from adahessian.image_classification.optim_adahessian import Adahessian

from .BH_FishRNN import BevertonHoltRNN_gac, BevertonHoltRNN_gtc
from .SP_FIshRNN import SurplusRNN_gac, SurplusRNN_gtc
from .datasets import check_negative

import functools
print = functools.partial(print, flush=True)


def train(net, optimizer, mask, efforts, catches, num_epochs, batch_size, restart=True, random_init=False,
           normalize=True, noise_std=0.0, weight_loss=False, verbose=False, random_state=None, l2_lambda=0.01, alpha=0.0):
    # random = check_random_state(random_state)
    # print('original catches:', catches)
    c = catches.copy()
    c[np.where(np.isnan(c))] = np.mean(c[tuple(mask)])

    if restart:
        efforts, catches = net.init_params(efforts, c, normalize=normalize, random_init=random_init,
                                           random_state=random_state)
    else:
        efforts = efforts / net.Escale
        catches = catches / net.Cscale
        catches = torch.tensor(catches)

    print(net.get_params())

    # print('scaled catches:', catches)
    # print('masked catches:', catches[tuple(mask)])

    if weight_loss:
        print('weighted loss')
        weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
        for j in range(len(efforts) - 7):
            weight.append(0.4 / (len(efforts) - 7))
        weight = torch.tensor(weight)
    else:
        weight = torch.ones(len(efforts))

    best_loss = np.inf
    best_solution = net.get_params(normalized=False)

    if isinstance(optimizer, torch.optim.LBFGS):

        for i in range(num_epochs):
            # print('epoch', i)
            def closure():
                optimizer.zero_grad()
                loss = 0
                for j in range(batch_size):
                    output = net(efforts, catches, noise_std=noise_std, random_state=random_state)
                    loss += torch.mean((output[tuple(mask)] - catches[tuple(mask)]) ** 2)

                   # l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
                    # l2_norm = net.rho.data.item()**2 + (net.K.data.item() * net.Cscale)**2 \
                    #     + (net.B0.data.item() * net.Cscale)**2 + (net.q.data.item() / net.Escale)**2
                        
                    loss = loss #+ l2_lambda * l2_norm

                loss /= batch_size  # +

                # print('using l2 reg, loss=', loss)

                loss.backward()

                if verbose:
                    net.print_gradient()
                    net.print_loss(i, loss)

                return loss

            # if verbose:
            #     net.print_gradient()

            loss = optimizer.step(closure)

            if verbose:
                print("gradient")
                net.print_gradient()
                print('loss')
                net.print_loss(i, loss)
                print('')

            if loss < best_loss and not np.isnan(net.K.data):
                best_loss = loss
                best_solution = net.get_params(normalized=False)

    else:
        for i in range(num_epochs):
            optimizer.zero_grad()
            loss = 0
            for j in range(batch_size):
                output = net(efforts, catches, noise_std=noise_std, random_state=random_state)
                loss += torch.mean(weight * (output[tuple(mask)] - catches[tuple(mask)]) ** 2)

            loss /= batch_size
            loss.backward()

            print('before step')
            print(net.get_params())
            for param_group in optimizer.param_groups:
                a = param_group['lr']

            if verbose:
                net.print_loss(i, loss)
                net.print_gradient()

            # optimizer.step()
            for param in net.parameters():
                param.data.add_(- a * param.grad.data)

            print('after step')
            print(net.get_params())

            # if verbose:
            #     net.print_loss(i, loss)

            if loss < best_loss and not np.isnan(net.K.data):
                best_loss = loss
                best_solution = net.get_params(normalized=False)

    return best_solution


# use BevertonHoltRNN_gac/gtc, SurplusRNN_gac/gtc classes
def find_sol(net, optimizer, data_dic, num_epochs, batch_size, mse_threshold=0.001, restart=True,
             random_init=True, noise_std=0.0, verbose=False, normalize=True, weight_loss=False,
             random_state=None, trials=5, alpha=0, m=2, l2_lambda=0.001):

    random = check_random_state(random_state)
    best_mse = np.inf
    best_sol = []

    seeds = []
    # print('train by', model, 'model')

    mask = [~np.isnan(data_dic['for model training']['efforts'])]
    # print('found missing years:', np.where(~mask[0]))

    e = data_dic['for model training']['efforts'].copy()
    # e[np.where(np.isnan(e))]=np.mean(e[tuple(mask)])

    mys = np.where(~mask[0])
    print('found missing years:', mys)

    for my in mys[0]:
        # print(np.nanmean(e[my-2:my+3]))
        imp = np.nanmean(e[my-2:my+3])
        if np.isnan(imp):
            e[my] = np.mean(e[tuple(mask)])
        else:
            e[my] = imp
    print('imputed effort:', e)

    for i in range(trials):
        print('***', i, '-th trial***')
        seed = random.randint(0, 10000)
        seeds.append(seed)

        best_solution = train(net, optimizer, mask=mask, efforts = e,
                              catches = data_dic['for model training']['catches'], num_epochs=num_epochs, batch_size=batch_size,
                              restart=restart, random_init=random_init, normalize=normalize, noise_std=noise_std,
                              weight_loss=weight_loss, verbose=verbose, random_state=seed, alpha=alpha, l2_lambda=l2_lambda)

        best_params_values = list(best_solution.values())

        mse = net.expected_mse(rho=best_params_values[0], K=best_params_values[1], B0=best_params_values[2],
                               q=best_params_values[3], mask=mask,
                           efforts=e, catches=data_dic['complete data']['catches'],
                               noise_std=noise_std, nsample=1000, retStd=False, m=m)
        # print('result:', best_solution, mse)

        if mse < best_mse:
            best_mse = mse
            best_sol = best_solution
            best_net = net

            print('A better solution found!')
            print(net.get_params())
            print('mse:', mse)
        if best_mse < mse_threshold:
            break
    print('used seeds', seeds)
    return best_sol, best_mse, best_net


if __name__ == "__main__":
    pass




