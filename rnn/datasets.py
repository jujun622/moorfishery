import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.utils import check_random_state
from .FishModels import BevertonHolt, Surplus, catch_model


def check_negative(biomasses, efforts, catches):
    negative = 0

    if not all(biomasses[i] >= 0 for i in range(len(biomasses))):
        negative += 1
        print('negative values in biomasses data')
    if not all(efforts[i] >= 0 for i in range(len(efforts))):
        negative += 1
        print('negative values in efforts data')
    if not all(catches[i] >= 0 for i in range(len(catches))):
        negative += 1
        print('negative values in catches data')

    if negative == 0:
        return False
    else:
        warnings.warn('Negative values in sampled data!')
        return True


if __name__ == '__main__':
    pass


