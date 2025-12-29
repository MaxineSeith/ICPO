import numpy as np
import matplotlib.pyplot as plt
from cec2014 import CEC2014
import time
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_PATH = "results_data_FLA"

def init_x(n, d, u, l):
    u = np.array(u).flatten()
    l = np.array(l).flatten()
    if len(u) == 1:
        return np.random.rand(n, d) * (u[0] - l[0]) + l[0]
    pos = np.zeros((n, d))
    for i in range(d):
        pos[:, i] = np.random.rand(n) * (u[i] - l[i]) + l[i]
    return pos


def get_lims(fid):
    d = 10
    v = 600 if fid == 11 else 100
    return -v * np.ones(d), v * np.ones(d), d


def export_res(fid, data, opt):
    data.sort(key=lambda x: x[0])
    sc = [x[1] for x in data]
    cv = [x[2] for x in data]

    if cv:
        ml = max(len(c) for c in cv)
        tmp = []
        for c in cv:
            pad = np.pad(c, (0, ml - len(c)), 'edge') if len(c) < ml else c
            tmp.append(pad)

        df = pd.DataFrame(np.array(tmp).T, columns=[f'Run_{i + 1}' for i in range(len(sc))])
        df.to_csv(os.path.join(OUT_PATH, f'FLA_F{fid:02d}_Raw.csv'), index_label='Iter')

    sc = np.array(sc)
    v_sc = sc[sc != np.inf]

    if len(v_sc) > 0:
        bst = np.min(v_sc) - opt
        avg = np.mean(v_sc) - opt
        std = np.std(v_sc)
        print(f"F{fid:02d} | Best: {bst:.4e} | Avg: {avg:.4e} | Std: {std:.4e}")
    else:
        print(f"F{fid:02d} | Failed")

    if cv:
        plt.figure(figsize=(10, 6))
        ml = min(len(c) for c in cv)
        t_cv = [c[:ml] for c in cv]
        x_ax = np.arange(1, ml + 1)

        for c in t_cv:
            plt.semilogy(x_ax, c, 'lightgray', alpha=0.5, lw=0.5)

        plt.semilogy(x_ax, np.mean(t_cv, axis=0), 'b-', lw=2)
        plt.semilogy(x_ax, t_cv[np.argmin(sc)], 'r--', lw=1.5)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_PATH, f'FLA_F{fid:02d}_Plot.png'), dpi=300)
        plt.close()


def run_fla(pop, mit, ub, lb, dim, fid, cec):
    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()
    if len(ub) == 1: ub = np.full(dim, ub[0])
    if len(lb) == 1: lb = np.full(dim, lb[0])

    gbest = np.zeros(dim)
    gfit = np.inf
    curve = np.zeros(mit)
    Ne = 5

    X = init_x(pop, dim, ub, lb)
    fit = np.zeros(pop)

    for i in range(pop):
        fit[i] = cec.evaluate(X[i, :], fid)
        if fit[i] < gfit:
            gfit = fit[i]
            gbest = X[i, :].copy()

    for it in range(mit):
        t = it + 1
        pk = ((np.sqrt(mit * (t ** 2) + 1) + (1.2 / t)) ** (-2 / 3)) * (1.2 / t)

        Xnew = X.copy()
        fmin = np.min(fit)
        fmax = np.max(fit)
        denom = fmax - fmin + np.finfo(float).eps

        for i in range(pop):
            idx = np.random.randint(0, pop)
            while idx == i:
                idx = np.random.randint(0, pop)

            pe = ((fit[i] - fmin) / denom) ** 2
            r1 = np.random.rand()
            r2 = np.random.rand()

            if fit[i] < fit[idx]:
                dn = X[i] - X[idx]
                db = X[i] - gbest
                Xnew[i] = X[i] + r1 * dn + r2 * pk * db
            else:
                dn = X[idx] - X[i]
                db = gbest - X[i]
                Xnew[i] = X[i] + r1 * dn + r2 * pk * db

            Xnew[i] = np.clip(Xnew[i], lb, ub)
            nf = cec.evaluate(Xnew[i, :], fid)

            if nf < fit[i]:
                fit[i] = nf
                X[i, :] = Xnew[i, :].copy()
                if nf < gfit:
                    gfit = nf
                    gbest = X[i, :].copy()

        pt = np.abs(np.sin(np.random.rand() / t))

        if np.random.rand() < pt:
            s_idx = np.argsort(fit)
            w_idx = s_idx[-Ne:]

            for k in w_idx:
                rv = lb + np.random.rand(dim) * (ub - lb)
                X[k, :] = gbest + np.random.rand() * 0.1 * rv
                X[k, :] = np.clip(X[k, :], lb, ub)

                fit[k] = cec.evaluate(X[k, :], fid)
                if fit[k] < gfit:
                    gfit = fit[k]
                    gbest = X[k, :].copy()

        curve[it] = gfit

    return gfit, gbest, curve


def task(fid, rid, p, m, u, l, d):
    try:
        c = CEC2014(d)
        s, _, cr = run_fla(p, m, u, l, d, fid, c)
        return fid, rid, s, cr
    except Exception as e:
        print(e)
        return fid, rid, np.inf, []


def main():
    dim = 10
    pop = 50
    iters = 500
    runs = 20

    full_test = True
    f_list = list(range(1, 31)) if full_test else [1]

    print(f"Running FLA. Mode: {'Full' if full_test else 'Single'}")
    t0 = time.time()

    db = {f: [] for f in f_list}

    with ProcessPoolExecutor() as ex:
        jobs = []
        for f in f_list:
            l, u, _ = get_lims(f)
            for r in range(runs):
                jobs.append(ex.submit(task, f, r, pop, iters, u, l, dim))

        cnt = 0
        tot = len(jobs)
        for j in as_completed(jobs):
            f, r, s, c = j.result()
            db[f].append((r, s, c))
            cnt += 1
            if cnt % 20 == 0:
                print(f"Done: {cnt}/{tot}")

    for f in f_list:
        if db[f]:
            export_res(f, db[f], f * 100)

    print(f"Total Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
