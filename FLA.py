import numpy as np
import matplotlib.pyplot as plt
from cec2014 import CEC2014
import time
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = "results_data_FLA"


def initialization(search_agents_no, dim, ub, lb):

    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()
    boundary_no = len(ub)

    if boundary_no == 1:
        positions = np.random.rand(search_agents_no, dim) * (ub[0] - lb[0]) + lb[0]
    else:
        positions = np.zeros((search_agents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = np.random.rand(search_agents_no) * (ub_i - lb_i) + lb_i
    return positions


def get_functions_details_cec(func_id):
    """
    获取CEC2014函数的边界信息
    """
    dim = 10
    if func_id == 11:
        lb = -600 * np.ones(dim)
        ub = 600 * np.ones(dim)
    else:
        lb = -100 * np.ones(dim)
        ub = 100 * np.ones(dim)
    return lb, ub, dim


def save_data_and_plot(func_id, results_list, optimal_val):

    # 提取数据并按 Run 编号排序
    results_list.sort(key=lambda x: x[0])

    fitness_vals = []
    convergence_curves = []

    for _, score, curve in results_list:
        fitness_vals.append(score)
        convergence_curves.append(curve)

    run_no = len(fitness_vals)
    fitness_vals = np.array(fitness_vals)

    if convergence_curves:
        # 填充数据以防长度不一
        max_len = max(len(c) for c in convergence_curves)
        padded_curves = []
        for curve in convergence_curves:
            if len(curve) < max_len:
                padded = np.pad(curve, (0, max_len - len(curve)), 'edge')
                padded_curves.append(padded)
            else:
                padded_curves.append(curve)

        data = np.array(padded_curves).T
        columns = [f'Run_{i + 1}' for i in range(run_no)]
        df = pd.DataFrame(data, columns=columns)
        df.index.name = 'Iteration'

        csv_path = os.path.join(DATA_DIR, f'FLA_F{func_id:02d}_Raw_Data.csv')
        df.to_csv(csv_path)

    # 统计指标
    valid_fitness = fitness_vals[fitness_vals != np.inf]

    if len(valid_fitness) > 0:
        best_val = np.min(valid_fitness)
        worst_val = np.max(valid_fitness)
        avg_val = np.mean(valid_fitness)
        std_val = np.std(valid_fitness)
        avg_error = avg_val - optimal_val
        std_error = std_val
        best_error = best_val - optimal_val
    else:
        best_val = worst_val = avg_val = std_val = avg_error = std_error = best_error = np.inf

    print(f"F{func_id:02d}")
    print(f"  最优误差: {best_error:.4e}")
    print(f"  平均误差:  {avg_error:.4e}")
    print(f"  标准差:  {std_error:.4e}")

    # 绘图
    if convergence_curves:
        plt.figure(figsize=(10, 6))

        # 统一长度用于绘图
        min_len = min(len(c) for c in convergence_curves)
        trimmed_curves = [c[:min_len] for c in convergence_curves]
        iterations = np.arange(1, min_len + 1)

        # 绘制灰色背景线
        for curve in trimmed_curves:
            plt.semilogy(iterations, curve, 'lightgray', alpha=0.5, linewidth=0.5)

        # 绘制平均线
        mean_curve = np.mean(trimmed_curves, axis=0)
        plt.semilogy(iterations, mean_curve, 'b-', linewidth=2, label='Mean Fitness')

        # 绘制最佳线
        best_run_idx = np.argmin(fitness_vals)
        plt.semilogy(iterations, trimmed_curves[best_run_idx], 'r--', linewidth=1.5, label='Best Run')

        plt.xlabel('Iterations')
        plt.ylabel('Fitness (Log Scale)')
        plt.title(f'FLA Convergence - F{func_id}')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        img_path = os.path.join(DATA_DIR, f'FLA_F{func_id:02d}_Plot.png')
        plt.savefig(img_path, dpi=1200)
        plt.close()


# FLA
def FLA(pop_size, max_iter, ub, lb, dim, fobj, cec_instance):

    # 边界处理
    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()
    if len(ub) == 1: ub = np.full(dim, ub[0])
    if len(lb) == 1: lb = np.full(dim, lb[0])

    # 初始化
    gb_sol = np.zeros(dim)
    gb_fit = np.inf
    conv_curve = np.zeros(max_iter)

    Ne = 5  # 洪水阶段移除的差解数量

    X = initialization(pop_size, dim, ub, lb)
    fitness = np.zeros(pop_size)

    # 初始评估
    for i in range(pop_size):
        fitness[i] = cec_instance.evaluate(X[i, :], fobj)
        if fitness[i] < gb_fit:
            gb_fit = fitness[i]
            gb_sol = X[i, :].copy()

    # 主循环
    for it in range(max_iter):
        t = it + 1
        PK = ((np.sqrt(max_iter * (t ** 2) + 1) + (1.2 / t)) ** (-2 / 3)) * (1.2 / t)

        X_new = X.copy()
        min_fit = np.min(fitness)
        max_fit = np.max(fitness)
        denominator = max_fit - min_fit + np.finfo(float).eps

        # Phase I: Regular Movement
        for i in range(pop_size):
            ind = np.random.randint(0, pop_size)
            while ind == i:
                ind = np.random.randint(0, pop_size)

            Pe = ((fitness[i] - min_fit) / denominator) ** 2
            r1 = np.random.rand()
            r2 = np.random.rand()

            if fitness[i] < fitness[ind]:
                dist_neighbor = X[i] - X[ind]
                dist_best = X[i] - gb_sol
                X_new[i] = X[i] + r1 * dist_neighbor + r2 * PK * dist_best
            else:
                dist_neighbor = X[ind] - X[i]
                dist_best = gb_sol - X[i]
                X_new[i] = X[i] + r1 * dist_neighbor + r2 * PK * dist_best

            X_new[i] = np.clip(X_new[i], lb, ub)
            new_fit = cec_instance.evaluate(X_new[i, :], fobj)

            if new_fit < fitness[i]:
                fitness[i] = new_fit
                X[i, :] = X_new[i, :].copy()
                if new_fit < gb_fit:
                    gb_fit = new_fit
                    gb_sol = X[i, :].copy()

        # Phase II: Flooding
        Pt = np.abs(np.sin(np.random.rand() / t))

        if np.random.rand() < Pt:
            sorted_indices = np.argsort(fitness)
            worst_indices = sorted_indices[-Ne:]

            for idx in worst_indices:
                rand_vec = lb + np.random.rand(dim) * (ub - lb)
                X[idx, :] = gb_sol + np.random.rand() * 0.1 * rand_vec
                X[idx, :] = np.clip(X[idx, :], lb, ub)

                fitness[idx] = cec_instance.evaluate(X[idx, :], fobj)
                if fitness[idx] < gb_fit:
                    gb_fit = fitness[idx]
                    gb_sol = X[idx, :].copy()

        conv_curve[it] = gb_fit

    return gb_fit, gb_sol, conv_curve


# 并行 Worker 函数
def worker_fla(func_id, run_idx, pop_size, max_iter, ub, lb, dim):

    # 局部实例化，防止进程冲突
    cec_local = CEC2014(dim)

    # 运行算法
    best_score, best_pos, conv_curve = FLA(pop_size, max_iter, ub, lb, dim, func_id, cec_local)

    return func_id, run_idx, best_score, conv_curve



def main():

    DIM = 10
    POP_SIZE = 50
    MAX_ITER = 500
    N_RUNS = 20

    # 模式选择: "single" 跑单个, "full" 跑 F1-F30
    TEST_MODE = "full"
    SINGLE_FUNC_ID = 1

    # 确定要跑的函数列表
    if TEST_MODE == "full":
        func_ids = list(range(1, 31))
    else:
        func_ids = [SINGLE_FUNC_ID]

    start_time_total = time.time()

    all_results = {fid: [] for fid in func_ids}

    total_tasks = len(func_ids) * N_RUNS
    finished_count = 0

    with ProcessPoolExecutor() as executor:
        futures = []

        # 遍历所有函数和所有运行
        for fid in func_ids:
            lb, ub, _ = get_functions_details_cec(fid)
            for r in range(N_RUNS):
                # 提交任务
                futures.append(executor.submit(worker_fla, fid, r, POP_SIZE, MAX_ITER, ub, lb, DIM))


        # 收集结果
        for future in as_completed(futures):
            # 获取返回值
            fid, r_idx, score, curve = future.result()

            # 存入对应函数的列表中
            all_results[fid].append((r_idx, score, curve))

    # 所有数据都在 all_results 字典里
    for fid in func_ids:
        results = all_results[fid]

        optimal = fid * 100
        save_data_and_plot(fid, results, optimal)

    total_time = time.time() - start_time_total
    print(f"耗时: {total_time:.2f}s")

if __name__ == "__main__":
    main()