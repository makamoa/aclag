import jax.numpy as jnp
import numpy as np
import time
import compression_operator
from compression_operator import Top_k
# TODO: CLAG_step function needs a compression operator to be a Top_k operator.
# Resolve this issue.


def LAG_step(x_k, g_k, comm, oracle_container,
             grad_k_prev, zeta, stepsize):
    x_k -= stepsize * g_k.mean(axis=0)
    grad_k = oracle_container.compute_grad(x_k)
    trigger_rhs = zeta * \
        np.linalg.norm(grad_k - grad_k_prev, ord=2, axis=1) ** 2
    trigger_lhs = np.linalg.norm(g_k - grad_k, ord=2, axis=1) ** 2
    trigger = jnp.expand_dims(jnp.array(trigger_lhs > trigger_rhs), 1)
    g_k = jnp.multiply(grad_k, trigger) + jnp.multiply(g_k, 1 - trigger)
    comm += len(x_k) * trigger.sum()
    return x_k, g_k, grad_k, comm


def LAG(x_0, oracle_container, stepsize, zeta, max_comm):
    g_k = oracle_container.compute_grad(x_0)
    grad_k_prev = jnp.array(np.copy(g_k))
    x_k = jnp.array(np.copy(x_0))
    history = [np.linalg.norm(g_k.mean(axis=0))]
    comm = oracle_container.num_clients() * len(x_0)
    history_comm = [0]
    while(history_comm[-1] < max_comm):
        x_k, g_k, grad_k_prev, comm = LAG_step(
            x_k, g_k, comm, oracle_container, grad_k_prev, zeta, stepsize
        )
        history.append(np.linalg.norm(grad_k_prev.mean(axis=0)))
        history_comm.append(comm)
        print('Currently communicated {} float numbers'.format(comm), end='\r')
    return history, history_comm


def GD(x_0, oracle_container, stepsize, max_iter):
    g_k = None
    x_k = jnp.array(np.copy(x_0))
    history = []
    history_comm = []
    comm = 0
    num_clients = oracle_container.num_clients()
    for i in range(max_iter):
        g_k = oracle_container.compute_grad(x_k)
        print('Iteration {} / {}'.format(i + 1, max_iter), end='\r')
        grad_ = g_k.mean(axis=0)
        history.append(np.linalg.norm(grad_))
        history_comm.append(comm)
        x_k -= stepsize * grad_
        comm += num_clients * len(x_0)
    return history, history_comm


def CLAG_step(x_k, g_k, comm, oracle_container, compression_operator,
              trigger_beta, grad_k_prev, stepsize, g_k_tilde=None):
    if g_k_tilde is None:
        x_k -= stepsize * g_k.mean(axis=0)
    else:
        x_k -= stepsize * g_k_tilde
    grad_k = oracle_container.compute_grad(x_k)
    trigger_rhs = trigger_beta * np.linalg.norm(
        grad_k - grad_k_prev, ord=2, axis=1) ** 2
    trigger_lhs = np.linalg.norm(g_k - grad_k, ord=2, axis=1) ** 2
    trigger = jnp.expand_dims(jnp.array(trigger_lhs > trigger_rhs), 1)
    compressed = jnp.vstack([compression_operator.compress(grad_k[i] - g_k[i])
                            for i in range(len(g_k))])
    g_k += jnp.multiply(compressed, trigger)
    comm += trigger.sum() * compression_operator.k
    return x_k, g_k, grad_k, comm, trigger.astype(np.int).flatten() * compression_operator.k

def adaptive_step(x_k_prev, g_k_prev, comm, oracle_container, compression_operator,
              trigger_beta, grad_k_prev, stepsize_k_prev, theta_k_prev, alpha):
    x_k = x_k_prev - stepsize_k_prev * g_k_prev.mean(axis=0)
    grad_k = oracle_container.compute_grad(x_k)
    trigger_rhs = trigger_beta * np.linalg.norm(
        grad_k - grad_k_prev, ord=2, axis=1) ** 2
    trigger_lhs = np.linalg.norm(g_k_prev - grad_k, ord=2, axis=1) ** 2
    trigger = jnp.expand_dims(jnp.array(trigger_lhs > trigger_rhs), 1)
    compressed = jnp.vstack([compression_operator.compress(grad_k[i] - g_k_prev[i])
                            for i in range(len(g_k_prev))])
    g_k = g_k_prev + jnp.multiply(compressed, trigger)
    ## adaptive stepsize part
    den = np.linalg.norm(g_k - g_k_prev, ord=2, axis=1)
    num = np.linalg.norm(x_k - x_k_prev, ord=2)
    stepsize = np.min(([[np.sqrt(1 + theta_k_prev)*stepsize_k_prev] * len(g_k), num/den*alpha]), axis=0).min()
    theta_k = stepsize / stepsize_k_prev
    ##
    comm += trigger.sum() * compression_operator.k
    return x_k, g_k, grad_k, comm, trigger.astype(np.int).flatten() * compression_operator.k, stepsize, theta_k

def adaptive(x_0, oracle_container, compression_operator,
         trigger_beta, max_comm, time_limit=None, alpha=0.5):
    assert trigger_beta >= 0
    g_k = oracle_container.compute_grad(x_0)
    grad_k_prev = jnp.array(np.copy(g_k))
    x_k = jnp.array(np.copy(x_0))
    history = [np.linalg.norm(g_k.mean(axis=0))]
    history_comm = [0]
    compressors = [np.zeros(g_k.shape[0], dtype=np.int)]
    comm = len(x_0) * oracle_container.num_clients()
    theta_k = 1e9
    stepsize_k = 0.01
    if time_limit is None:
        time_limit = np.float('inf')
    begin_time = time.time()
    while(history_comm[-1] < max_comm):
        print('Currently communicated {} float numbers'.format(comm), end='\r')
        history_comm.append(int(comm))
        x_k, g_k, grad_k_prev, comm, compres, stepsize_k, theta_k = adaptive_step(
            x_k, g_k, comm, oracle_container, compression_operator,
            trigger_beta, grad_k_prev, stepsize_k, theta_k, alpha=alpha
        )
        history.append(np.linalg.norm(grad_k_prev.mean(axis=0)))
        compressors.append(compres)
        if history[-1] > 1e5 or time.time() - begin_time > time_limit:
            break
    return history, history_comm, compressors

def aCLAG_step(x_k, g_k, comm, oracle_container, compression_operator,
              trigger_beta, grad_k_prev, stepsize, min_compressor=1, max_compressor=None, g_k_tilde=None):
    if g_k_tilde is None:
        x_k -= stepsize * g_k.mean(axis=0)
    else:
        x_k -= stepsize * g_k_tilde
    n, d = g_k.shape
    if max_compressor is None:
        max_compressor = d // 2
    triggered = np.array([False] * n)
    compressions = np.zeros(n,dtype=np.int)
    g_k_opt = np.zeros(g_k.shape)
    grad_k = oracle_container.compute_grad(x_k)
    for i in range(len(g_k)):
        for k in range(min_compressor, max_compressor, 5):
            compression_operator = Top_k(k)
            g_k_hat = g_k[i] + compression_operator.compress(grad_k[i] - g_k[i])
            trigger_rhs = trigger_beta * np.linalg.norm(
            grad_k[i] - grad_k_prev[i], ord=2) ** 2
            trigger_lhs = np.linalg.norm(g_k_hat[i] - grad_k[i], ord=2) ** 2
            if trigger_lhs < trigger_rhs:
                compressions[i] = k
                g_k_opt[i] = g_k_hat
                triggered[i] = True
                break
        if not triggered[i]:
            compression_operator = Top_k(max_compressor)
            trigger_rhs = trigger_beta * np.linalg.norm(grad_k[i] - grad_k_prev[i], ord=2) ** 2
            trigger_lhs = np.linalg.norm(g_k[i] - grad_k[i], ord=2) ** 2
            if trigger_lhs > trigger_rhs:
                g_k_opt[i] = g_k[i] + compression_operator.compress(grad_k[i] - g_k[i])
                compressions[i] = max_compressor
            else:
                g_k_opt[i] = g_k[i]
                compressions[i] = 0
            #g_k_opt[i] = g_k[i] + compression_operator.compress(grad_k[i] - g_k[i])
            compressions[i] = max_compressor
            triggered[i] = True
    g_k = jnp.array(g_k_opt)
    comm += compressions.sum()
    return x_k, g_k, grad_k, comm, compressions

def new_CLAG_step(x_k_prev, g_k_prev, comm, oracle_container, compression_operator,
              trigger_beta, grad_k_prev, stepsize):
    x_k = x_k_prev - stepsize * g_k_prev.mean(axis=0)
    grad_k = oracle_container.compute_grad(x_k)
    trigger_rhs = trigger_beta * np.linalg.norm(
        grad_k - grad_k_prev, ord=2, axis=1) ** 2
    trigger_lhs = np.linalg.norm(g_k_prev - grad_k, ord=2, axis=1) ** 2
    trigger = jnp.expand_dims(jnp.array(trigger_lhs > trigger_rhs), 1)
    compressed = jnp.vstack([compression_operator.compress(grad_k[i] - g_k_prev[i])
                            for i in range(len(g_k_prev))])
    g_k = g_k_prev + jnp.multiply(compressed, trigger)
    comm += trigger.sum() * compression_operator.k
    return x_k, g_k, grad_k, comm, trigger.astype(np.int).flatten() * compression_operator.k

def new_CLAG(x_0, oracle_container, compression_operator, scale,
         trigger_beta, max_comm, L, L_tilde, time_limit=None):
    assert trigger_beta >= 0
    d = len(x_0)
    beta = compression_operator.beta(d)
    theta = compression_operator.theta(d)
    B = max(trigger_beta, beta)
    L_ak = L_tilde
    g_k = oracle_container.compute_grad(x_0)
    grad_k_prev = jnp.array(np.copy(g_k))
    G_k_prev = (np.linalg.norm(grad_k_prev - g_k, ord=2, axis=1) ** 2).mean()
    x_k_prev = jnp.array(np.copy(x_0))
    history = [np.linalg.norm(g_k.mean(axis=0))]
    history_comm = [0]
    compressors = [np.zeros(g_k.shape[0], dtype=np.int)]
    comm = len(x_0) * oracle_container.num_clients()
    if time_limit is None:
        time_limit = np.float('inf')
    begin_time = time.time()
    while(history_comm[-1] < max_comm):
        print('Currently communicated {} float numbers'.format(comm), end='\r')
        history_comm.append(int(comm))
        theoretical_stepsize = 1. / (L_ak + L_ak * np.sqrt(
            B / theta))
        stepsize = theoretical_stepsize * scale
        x_k, g_k, grad_k, comm, compres = new_CLAG_step(
            x_k_prev, g_k, comm, oracle_container, compression_operator,
            trigger_beta, grad_k_prev, stepsize
        )
        G_k = (np.linalg.norm(grad_k - g_k, ord=2, axis=1) ** 2).mean()
        L_est = np.sqrt(np.abs(G_k - (1-theta)*G_k_prev)) / np.sqrt(B) / np.linalg.norm(x_k - x_k_prev, ord=2)
        # if not L_est < L_tilde:
        #     print('not triggered', np.linalg.norm(grad_k_prev.mean(axis=0)), L_est)
        L_ak = np.min([L_est, L_tilde])
        x_k_prev = x_k
        grad_k_prev = grad_k
        G_k_prev = G_k
        history.append(np.linalg.norm(grad_k_prev.mean(axis=0)))
        compressors.append(compres)
        if history[-1] > 1e5 or time.time() - begin_time > time_limit:
            break
    return history, history_comm, compressors

def CLAG(x_0, oracle_container, compression_operator, stepsize,
         trigger_beta, max_comm, time_limit=None, adaptive=False):
    assert trigger_beta >= 0
    g_k = oracle_container.compute_grad(x_0)
    grad_k_prev = jnp.array(np.copy(g_k))
    x_k = jnp.array(np.copy(x_0))
    history = [np.linalg.norm(g_k.mean(axis=0))]
    history_comm = [0]
    compressors = [np.zeros(g_k.shape[0], dtype=np.int)]
    comm = len(x_0) * oracle_container.num_clients()
    if adaptive == True:
        CLAG_s = aCLAG_step
    else:
        CLAG_s = CLAG_step
    if time_limit is None:
        time_limit = np.float('inf')
    begin_time = time.time()
    while(history_comm[-1] < max_comm):
        print('Currently communicated {} float numbers'.format(comm), end='\r')
        history_comm.append(int(comm))
        x_k, g_k, grad_k_prev, comm, compres = CLAG_s(
            x_k, g_k, comm, oracle_container, compression_operator,
            trigger_beta, grad_k_prev, stepsize
        )
        history.append(np.linalg.norm(grad_k_prev.mean(axis=0)))
        compressors.append(compres)
        if history[-1] > 1e5 or time.time() - begin_time > time_limit:
            break
    return history, history_comm, compressors

def biCLAG(x_0, oracle_container, compression_operator, stepsize,
         trigger_beta, max_comm, time_limit=None, adaptive=False, master_c=None):
    assert trigger_beta >= 0
    g_k = oracle_container.compute_grad(x_0)
    g_k_tilde = g_k.copy().mean(axis=0)
    n, _ = g_k.shape
    d = len(x_0)
    if master_c is None:
        master_c = d
    master_compressor = Top_k(master_c)
    grad_k_prev = jnp.array(np.copy(g_k))
    x_k = jnp.array(np.copy(x_0))
    history = [np.linalg.norm(g_k.mean(axis=0))]
    history_comm = [0]
    compressors = [np.zeros(g_k.shape[0], dtype=np.int)]
    comm = len(x_0) * oracle_container.num_clients()
    if adaptive == True:
        CLAG_s = aCLAG_step
    else:
        CLAG_s = CLAG_step
    if time_limit is None:
        time_limit = np.float('inf')
    begin_time = time.time()
    while(history_comm[-1] < max_comm):
        print('Currently communicated {} float numbers'.format(comm), end='\r')
        history_comm.append(int(comm))
        x_k, g_k, grad_k_prev, comm, compres = CLAG_s(
            x_k, g_k, comm, oracle_container, compression_operator,
            trigger_beta, grad_k_prev, stepsize, g_k_tilde=g_k_tilde
        )
        g_k_tilde = g_k_tilde + master_compressor.compress(g_k.mean(axis=0) - g_k_tilde)
        compres += master_c * n
        history.append(np.linalg.norm(grad_k_prev.mean(axis=0)))
        compressors.append(compres)
        if history[-1] > 1e5 or time.time() - begin_time > time_limit:
            break
    return history, history_comm, compressors

def CLAG_it(x_0, oracle_container, compression_operator, trigger_beta,
            tol, stepsize, time_budget, adaptive=False):
    assert trigger_beta >= 0
    g_k = oracle_container.compute_grad(x_0)
    grad_k_prev = jnp.array(np.copy(g_k))
    x_k = jnp.array(np.copy(x_0))
    comm = len(x_0) * oracle_container.num_clients()
    if adaptive == True:
        CLAG_s = aCLAG_step
    else:
        CLAG_s = CLAG_step
    begin_time = time.time()
    success_flag = True
    while np.linalg.norm(grad_k_prev.mean(axis=0)) > tol:
        x_k, g_k, grad_k_prev, comm, compress = CLAG_s(
            x_k, g_k, comm, oracle_container, compression_operator,
            trigger_beta, grad_k_prev, stepsize
        )
        print('Tolerance = ', np.linalg.norm(grad_k_prev.mean(axis=0)),
              end='\r')
        if time.time() - begin_time > time_budget:
            success_flag = False
            break
    print('')
    if np.isnan(np.array(x_k)).any() or np.isnan(np.array(g_k)).any():
        success_flag = False
    return comm, success_flag


def heatmap_CLAG(x_0,
                 oracle_container,
                 ks,
                 trigger_betas,
                 tol,
                 stepsize_coefs,
                 time_budget,
                 file):
    heatmap = jnp.zeros(shape=(len(ks), len(trigger_betas)))
    d = len(x_0)
    L = oracle_container.compute_smoothness()
    L_tilde = oracle_container.compute_distributed_smoothness()
    assert trigger_betas[0] == 0
    for k_id, k in enumerate(ks):
        print('k = ', k)
        top_k = compression_operator.Top_k(k)
        # choosing the best stepsize for EF21
        beta = top_k.beta(d)
        theta = top_k.theta(d)
        theoretical_stepsize = 1. / (L + L_tilde * np.sqrt(beta / theta))
        best_comm = float('inf')
        best_stepsize = 0.
        for coef in stepsize_coefs:
            stepsize = coef * theoretical_stepsize
            comm, flag = CLAG_it(x_0, oracle_container, top_k, 0,
                                 tol, stepsize, time_budget)
            print(comm, flag)
            if flag and comm < best_comm:
                best_comm = comm
                best_stepsize = stepsize
        print(best_stepsize)
        for beta_id, trigger_beta in enumerate(trigger_betas):
            print('trigger_beta = ', trigger_beta)
            comm, flag = CLAG_it(x_0, oracle_container, top_k, trigger_beta,
                                 tol, best_stepsize, time_budget)
            print(comm, flag)
            if flag:
                heatmap = heatmap.at[k_id, beta_id].set(comm)
            else:
                heatmap = heatmap.at[k_id, beta_id].set(-1)
            jnp.save(file, heatmap)
    return heatmap

def heatmap_aCLAG(x_0,
                      oracle_container,
                      ks,
                      trigger_betas,
                      tol,
                      stepsize_coefs,
                      time_budget,
                      file):
    heatmap = jnp.zeros(shape=(len(ks), len(trigger_betas)))
    d = len(x_0)
    L = oracle_container.compute_smoothness()
    L_tilde = oracle_container.compute_distributed_smoothness()
    for k_id, k in enumerate(ks[:1]):
        print('k = ', k)
        top_k = compression_operator.Top_k(k)
        beta = top_k.beta(d)
        theta = top_k.theta(d)
        for beta_id, trigger_beta in enumerate(trigger_betas):
            print('trigger_beta = ', trigger_beta)
            # choosing the best stepsize for aCLAG
            theoretical_stepsize = 1. / \
                (L + L_tilde * np.sqrt(max(beta, trigger_beta) / theta))
            best_comm = float('inf')
            for coef in stepsize_coefs:
                stepsize = coef * theoretical_stepsize
                comm, flag = CLAG_it(x_0, oracle_container, top_k,
                                     trigger_beta, tol, stepsize, time_budget, adaptive=True)
                print(comm, flag)
                if flag and comm < best_comm:
                    best_comm = comm
            print('Best communication is {}'.format(best_comm))
            if best_comm != float('inf'):
                heatmap = heatmap.at[k_id, beta_id].set(best_comm)
            else:
                heatmap = heatmap.at[k_id, beta_id].set(-1)
            jnp.save(file, heatmap)
    return heatmap

def heatmap_CLAG_full(x_0,
                      oracle_container,
                      ks,
                      trigger_betas,
                      tol,
                      stepsize_coefs,
                      time_budget,
                      file):
    heatmap = jnp.zeros(shape=(len(ks), len(trigger_betas)))
    d = len(x_0)
    L = oracle_container.compute_smoothness()
    L_tilde = oracle_container.compute_distributed_smoothness()
    for k_id, k in enumerate(ks):
        print('k = ', k)
        top_k = compression_operator.Top_k(k)
        beta = top_k.beta(d)
        theta = top_k.theta(d)
        for beta_id, trigger_beta in enumerate(trigger_betas):
            print('trigger_beta = ', trigger_beta)
            # choosing the best stepsize for CLAG
            theoretical_stepsize = 1. / \
                (L + L_tilde * np.sqrt(max(beta, trigger_beta) / theta))
            best_comm = float('inf')
            for coef in stepsize_coefs:
                stepsize = coef * theoretical_stepsize
                comm, flag = CLAG_it(x_0, oracle_container, top_k,
                                     trigger_beta, tol, stepsize, time_budget)
                print(comm, flag)
                if flag and comm < best_comm:
                    best_comm = comm
            print('Best communication is {}'.format(best_comm))
            if best_comm != float('inf'):
                heatmap = heatmap.at[k_id, beta_id].set(best_comm)
            else:
                heatmap = heatmap.at[k_id, beta_id].set(-1)
            jnp.save(file, heatmap)
    return heatmap