import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, 
                   w_threshold=0.3, substract_variances=False, log_gradients=False, 
                   gradient_log_dir: str=None, rand_init=False):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    # ensure log-file is passed if gradients are logged
    assert (not log_gradients) or (log_gradients and gradient_log_dir is not None)
    assert gradient_log_dir is None or (gradient_log_dir is not None and gradient_log_dir.endswith('/'))

    loss_gradients = []
    losses = []
    objectives = []
    acyclicity_losses = []
    acyclicity_gradients = []

    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        loss_gradients.append(G_loss)
        losses.append(loss)
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        acyclicity_losses.append(h)
        acyclicity_gradients.append(G_h)
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        W_c = np.copy(W)
        if substract_variances:
            parentless = [i for i in range(W.shape[0]) if np.all(W[:, i] < w_threshold)]
            for p in parentless:
                W_c[p, p] = 1
        loss, G_loss = _loss(W_c)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        objectives.append(obj)
        return obj, g_obj

    n, d = X.shape
    if rand_init:
        w_est, rho, alpha, h = np.random.uniform(-3, 3, 2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    else:
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    print(W_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    # log metadata
    if log_gradients:
        log_loss(losses, gradient_log_dir + 'losses.csv')
        log_loss(acyclicity_losses, gradient_log_dir + 'acyclicity_losses.csv')
        log_loss(objectives, gradient_log_dir + 'objectives.csv')
        log_gradient_array(loss_gradients, gradient_log_dir + 'loss_gradients.csv')
        log_gradient_array(acyclicity_gradients, gradient_log_dir + 'acyclicity_gradients.csv')
    return W_est

def log_loss(losses, file_name):
    loss_df = pd.DataFrame(data=losses, columns=['loss'])
    loss_df.to_csv(file_name)

def log_gradient_array(gradients, file_name):
    gradient_df = build_gradient_df(gradients)
    gradient_df.to_csv(file_name)

def build_gradient_df(gradients):
    df = pd.DataFrame()
    t = 0
    for mat in gradients:
        variables = np.arange(0, len(mat))
        mat_df = pd.DataFrame(data=mat, columns=list(range(0, len(mat))))
        mat_df['vars'] = variables
        mat_df['t'] = t
        t += 1
        df = pd.concat((df, mat_df))
    return df


if __name__ == '__main__':
    from notears import utils
    utils.set_random_seed(1)

    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert utils.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = utils.count_accuracy(B_true, W_est != 0)
    print(acc)

