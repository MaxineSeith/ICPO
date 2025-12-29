import numpy as np
import matplotlib.pyplot as plt
from cec2014 import CEC2014
import time
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

res_path = "results data FGO"

def init_pop(N, D, u, l):
    u = np.array(u).flatten()
    l = np.array(l).flatten()
    if len(u) == 1:
        pos = np.random.rand(N, D) * (u[0] - l[0]) + l[0]
    else:
        pos = np.zeros((N, D))
        for i in range(D):
            pos[:, i] = np.random.rand(N) * (u[i] - l[i]) + l[i]
    return pos


def get_bound(fid):
    D = 10
    if fid == 11:
        l = -600 * np.ones(D)
        u = 600 * np.ones(D)
    else:
        l = -100 * np.ones(D)
        u = 100 * np.ones(D)
    return l, u, D


def save_csv(fid, curves, runs):
    if not curves:
        return
    mlen = max(len(c) for c in curves)
    p_curves = []
    for c in curves:
        if len(c) < mlen:
            pad = np.pad(c, (0, mlen - len(c)), 'edge')
            p_curves.append(pad)
        else:
            p_curves.append(c)

    dat = np.array(p_curves).T
    cols = [f'Run_{i + 1}' for i in range(runs)]
    df = pd.DataFrame(dat, columns=cols)
    df.index.name = 'Gen'
    fname = os.path.join(res_path, f'FGO_F{fid:02d}_Raw.csv')
    df.to_csv(fname)
    print(f"Saved: {fname}")


def fgo_algo(N, max_e, u, l, D, obj_f, cec):
    u = np.array(u).flatten()
    l = np.array(l).flatten()
    if len(u) == 1: u = np.full(D, u[0])
    if len(l) == 1: l = np.full(D, l[0])

    g_best = np.zeros(D)
    g_val = np.inf
    curve = []

    X = init_pop(N, D, u, l)
    fit = np.zeros(N)

    evals = 0
    for i in range(N):
        fit[i] = cec.evaluate(X[i, :], obj_f)
        evals += 1
        if fit[i] < g_val:
            g_val = fit[i]
            g_best = X[i, :].copy()

    curve.append(g_val)

    m_iter = max_e / N
    curr = 0
    Ep = 0.7

    while evals < max_e:
        curr += 1
        E = 1 - (curr / m_iter)
        X_new = X.copy()

        for i in range(N):
            idxs = list(range(N))
            idxs.remove(i)
            if len(idxs) >= 3:
                a, b, c = np.random.choice(idxs, 3, replace=False)
            else:
                a, b, c = idxs[0], idxs[0], idxs[0]

            for j in range(D):
                sig = 1 if np.random.rand() < 0.5 else -1
                mu = sig * np.random.rand() * E

                if np.random.rand() < Ep:
                    X_new[i, j] = X[b, j] + mu * (X[c, j] - X[a, j])
                else:
                    w = curr / m_iter
                    t1 = (w * g_best[j] + (1 - w) * X[a, j])
                    t2 = X[b, j]
                    val_m = (t1 + t2) / 2.0
                    val_d = mu * np.abs((X[c, j] + X[a, j] + X[b, j]) / 3.0 - X[i, j])
                    X_new[i, j] = val_m + val_d

            X_new[i] = np.clip(X_new[i], l, u)

            if evals < max_e:
                n_fit = cec.evaluate(X_new[i, :], obj_f)
                evals += 1

                if n_fit < fit[i]:
                    fit[i] = n_fit
                    X[i, :] = X_new[i, :].copy()
                    if n_fit < g_val:
                        g_val = n_fit
                        g_best = X[i, :].copy()
            else:
                break

        curve.append(g_val)

    return g_val, g_best, np.array(curve)


def p_task(rid, pop, m_fes, u, l, D, fid):
    c_inst = CEC2014(D)
    s, p, c = fgo_algo(pop, m_fes, u, l, D, fid, c_inst)
    return rid, s, c


def run_exp(fid, N=50, M_FES=25050, R=20, D=10, plot_flg=True):
    t0 = time.time()
    l, u, D = get_bound(fid)
    opt = fid * 100

    res = [None] * R

    with ProcessPoolExecutor() as ex:
        futs = []
        for j in range(R):
            futs.append(ex.submit(p_task, j, N, M_FES, u, l, D, fid))

        for ft in as_completed(futs):
            try:
                idx, sc, cr = ft.result()
                res[idx] = (sc, cr)
                print(f"  F{fid} Run {idx + 1} finished. Val: {sc:.4e}")
            except Exception as e:
                print(f"  Error in run: {e}")

    fits = np.zeros(R)
    curves = []

    for k in range(R):
        if res[k]:
            fits[k] = res[k][0]
            curves.append(res[k][1])
        else:
            fits[k] = np.inf

    save_csv(fid, curves, R)

    valid = fits[fits != np.inf]
    t_end = time.time() - t0

    if len(valid) > 0:
        avg_v = np.mean(valid)
        std_v = np.std(valid)
        best_v = np.min(valid)
        print(f"F{fid} Stats  Avg: {avg_v:.4e}, Std: {std_v:.4e}, Best: {best_v:.4e}, Time: {t_end:.2f}s")
    else:
        print(f"F{fid} Failed.")

    if plot_flg and curves:
        plt.figure(figsize=(10, 6))
        ml = min(len(x) for x in curves)
        t_curves = [x[:ml] for x in curves]
        x_ax = np.arange(1, ml + 1)

        for tc in t_curves:
            plt.semilogy(x_ax, tc, 'lightgray', alpha=0.5, linewidth=0.5)

        avg_c = np.mean(t_curves, axis=0)
        plt.semilogy(x_ax, avg_c, 'b-', linewidth=2, label='Mean')

        bst_idx = np.argmin(fits)
        plt.semilogy(x_ax, t_curves[bst_idx], 'r--', linewidth=1.5, label='Best')

        plt.title(f'Convergence F{fid}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        pname = os.path.join(res_path, f'FGO F{fid:02d} Plot.png')
        plt.savefig(pname, dpi=1200)
        plt.close()


if __name__ == "__main__":
    ids = list(range(1, 31))
    for i in ids:
        run_exp(i, N=50, M_FES=25050, R=20, D=10)
