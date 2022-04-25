import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
# import NOTEARS
import nt_linear as nt
from sklearn.preprocessing import scale
import multiprocessing as mp
import os

# def experiment(experiment_nums):
def experiment(grid, experiments):
    pid = os.getpid()
    print("Process {} starts experiment".format(pid))
    expected_adj = np.array([[0, 0, 0],
                            [1, 0, 0], 
                            [0, 1, 0]])
    adjacencies = []
    metadata = []
    # for exp_num in experiment_nums:
    for exp_num, scales in zip(experiments, grid):
        #scales = np.random.uniform(-10, 10, 2)
        # generate data according to a causal graph X0 -> X1 -> X2
        X0 = np.random.normal(0, 1, 1000)
        X1 = scales[0]*X0 + np.random.normal(0, 1, 1000)
        X2 = scales[1]*X1 + np.random.normal(0, 1, 1000)
        # scale data s.t. we lose varsortability information
        gdata = np.array([X0, X1, X2]).T
        gdata = scale(gdata)
        # scale X0
        gdata[:, 0] *= 2
        gradient_log_dir = './experiments/flip-experiment-grid/gradients_{}/'.format(exp_num)
        # os.mkdir(gradient_log_dir)
        nt_pred = nt.notears_linear(gdata, 0.05, 'l2', log_gradients=False, gradient_log_dir=gradient_log_dir, rand_init=False)
        record = list(scales)
        # append experiment number, process id and 1 for success=true as default
        record.append(exp_num)
        record.append(pid)
        record.append(1)
        nt_pred_copy = nt_pred.copy()
        # NOTEARS estimates coefficients, set all to 1 to indicate an edge was found
        nt_pred[nt_pred != 0] = 1
        if not np.all(expected_adj == nt_pred):
            # set success to 0 if there's a mismatch
            record[-1] = 0
        metadata.append(record)
        adjacencies.append(nt_pred_copy)

    metadata_df = pd.DataFrame(data=metadata, columns=['s_1', 's_2', 'Exp', 'pid', 'success'])
    metadata_df.to_csv('./experiments/flip-experiment-grid/metadata_{}.csv'.format(pid))
    os.mkdir('./experiments/flip-experiment-grid/adjacencies_{}/'.format(pid))
    for idx, adj in enumerate(adjacencies):
        adj_df = pd.DataFrame(adj, columns=list(range(0, len(adj))), index=list(range(0, len(adj))))
        adj_df.to_csv('./experiments/flip-experiment-grid/adjacencies_{}/adj_{}.csv'.format(pid, idx))

if __name__ == '__main__':
    processes = []
    g = np.arange(-10, 10, 0.05)
    grid = np.array(list(itertools.product(g, g)))
    grid_splits = np.array_split(grid, mp.cpu_count())
    exp_nums = np.arange(0, len(grid))
    exp_num_splits = np.array_split(exp_nums, mp.cpu_count())
    for idx in range(0, mp.cpu_count()):
        p = mp.Process(target=experiment, args=[grid_splits[idx], exp_num_splits[idx]])
        processes.append(p)
        p.start()

    for p in processes:
        p.join()