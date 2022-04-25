import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
# import NOTEARS
import nt_linear as nt
from sklearn.preprocessing import scale
import multiprocessing as mp
import os
import pdb

def experiment(grid, experiments, attack_scale, nt_regularization, experiment_prefix):
    pid = os.getpid()
    print("Process {} starts experiment".format(pid))
    expected_adj = np.array([[0, 0, 0],
                            [1, 0, 0], 
                            [0, 1, 0]])
    adjacencies = []
    metadata = []
    noise_df = pd.read_csv('./experiments/flip-experiment-fixed-data-wiggle/noise/noise.csv', index_col=0)
    N0, N1, N2 = noise_df['N0'].to_numpy(), noise_df['N1'].to_numpy(), noise_df['N2'].to_numpy()
    gdata_prev = None
    #for exp_num in experiment_nums:
    reseeds = int(len(grid) / 10000)
    for _ in range(reseeds):
        np.random.seed()
        exp_idx = np.random.randint(0, len(grid), 10000)
        current_experiments = experiments[exp_idx]
        current_grid = grid[exp_idx]
        for exp_num, scales in zip(current_experiments, current_grid):
            # scales = np.random.uniform(-10, 10, 2)
            # generate data according to a causal graph X0 -> X1 -> X2
            s1 = scales[0] + np.random.normal(0, 1)
            s2 = scales[1] + np.random.normal(0, 1)
            X0 = N0
            X1 = s1*X0 + N1
            X2 = s2*X1 + N2
            # scale data s.t. we lose varsortability information
            gdata = np.array([X0, X1, X2]).T
            gdata = scale(gdata)
            # scale X0
            gdata[:, 0] *= attack_scale
            gradient_log_dir = experiment_prefix + '/gradients_{}/'.format(exp_num)
            # os.mkdir(gradient_log_dir)
            nt_pred = nt.notears_linear(gdata, nt_regularization, 'l2', log_gradients=False, gradient_log_dir=gradient_log_dir, rand_init=False)
            record = list(scales)
            # append experiment number, process id and 1 for success=true as default
            record.append(exp_num)
            record.append(pid)
            record.append(1)
            nt_pred_copy = nt_pred.copy()
            # NOTEARS estimates coefficients, set all to 1 to indicate an edge was found
            nt_pred[nt_pred != 0] = 1
            if not np.allclose(expected_adj, nt_pred):
                # set success to 0 if there's a mismatch
                record[-1] = 0
            metadata.append(record)
            adjacencies.append(nt_pred_copy)

    metadata_df = pd.DataFrame(data=metadata, columns=['s_1', 's_2', 'Exp', 'pid', 'success'])
    metadata_df.to_csv(experiment_prefix + '/metadata_{}.csv'.format(pid))
    os.mkdir(experiment_prefix + '/adjacencies_{}/'.format(pid))
    for idx, adj in enumerate(adjacencies):
        adj_df = pd.DataFrame(adj, columns=list(range(0, len(adj))), index=list(range(0, len(adj))))
        adj_df.to_csv(experiment_prefix + '/adjacencies_{}/adj_{}.csv'.format(pid, idx))

if __name__ == '__main__':
    g = np.arange(-10, 10, 0.05)
    grid = np.array(list(itertools.product(g, g)))
    grid_splits = np.array_split(grid, mp.cpu_count())
    exp_nums = np.arange(0, len(grid))
    exp_num_splits = np.array_split(exp_nums, mp.cpu_count())
    #attack_scales = [2, 4, 8, 10]
    #nt_regularization = [0, 0.01, 0.1, 1]
    exp_param_pairs = list(itertools.product([0, 0.01, 0.1, 1], [8, 10]))
    exp_param_pairs.append((1, 4))
    for reg, attack_scale in exp_param_pairs:
        print("Starting experiment with: ")
        print((reg, attack_scale))
        processes = []
        experiment_prefix = './experiments/flip-experiment-fixed-data-as{}-reg{}'.format(attack_scale, reg)
        os.mkdir(experiment_prefix)
        for idx in range(0, mp.cpu_count()):
            p = mp.Process(target=experiment, args=[grid_splits[idx], exp_num_splits[idx], attack_scale, reg, experiment_prefix])
            processes.append(p)
            p.start()       

        for p in processes:
            p.join()