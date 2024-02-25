import subprocess
from subprocess import Popen, PIPE
import os
import sys

import functools
print = functools.partial(print, flush=True)


# replace the despot argument with command for running DESPOT
def run_despot(pomdpx, save_name, despot='despot', options=[]):
    p = subprocess.Popen([despot, '-m', pomdpx] + options, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    output, err = p.communicate()
    if p.returncode != 0:
        print('Error encountered:', err)

    summary = output.decode("utf-8").split('\n')

    file = open(save_name, 'w')
    for line in summary:
        file.writelines(line+'\n')
    file.close()

    line = summary[-4]
    print('despot result:', line)

    toks = line.replace('(', '').replace(')', '').split('=')[-1].strip().split(' ')

    return [float(toks[0]), float(toks[1])]


def run_despot_evaluate(simulator_pomdpx, world_pomdpx, save_name, despot='despot', options=[]):
    p = subprocess.Popen([despot, '-m', simulator_pomdpx, '-w', world_pomdpx] + options, stdin=PIPE, stdout=PIPE, stderr=PIPE)

    output, err = p.communicate()
    if p.returncode != 0:
        print('Error encountered:', err)

    summary = output.decode("utf-8").split('\n')

    file = open(save_name, 'w')
    for line in summary:
        file.writelines(line+'\n')
    file.close()

    line = summary[-4]
    print('despot result:', line)

    toks = line.replace('(', '').replace(')', '').split('=')[-1].strip().split(' ')

    return [float(toks[0]), float(toks[1])]

#
if __name__ == "__main__":
    t = 0.001
    runs = 1 #1000
    simlen = 1 #100

    for pomdpx in os.listdir('./pomdpx_files'):
        if not 'DS' in pomdpx:
            print('')
            print(pomdpx)
            run_despot(
                pomdpx='/Users/junju/Desktop/PhD/DP2020/rnn+tp/pomdpx_files/'+pomdpx,
                save_name= './despot_results/%s__t%.3f_runs%d_simlen%d' % (pomdpx[:-7], t, runs, simlen),
                despot='/Users/junju/Desktop/PhD/experiments/despot/examples/pomdpx_models/pomdpx',
                options=['-t', str(t), '--runs', str(runs), '--simlen', str(simlen)])
