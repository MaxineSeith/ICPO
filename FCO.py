import numpy as np
import matplotlib.pyplot as plt
from cec2014 import CEC2014
import time
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = "results_data_FGO"

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


def save_data_to_csv(func_id, convergence_curves, run_no):

    # 找出最大长度
    if not convergence_curves:
        return

    max_len = max(len(c) for c in convergence_curves)

    # 填充数据
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

    df.index.name = 'Generation'

    # 保存文件
    filename = os.path.join(DATA_DIR, f'FGO_F{func_id:02d}_Raw_Data.csv')
    df.to_csv(filename)


# FGO
def FGO(pop_size, max_fes, ub, lb, dim, fobj, cec_instance):

    # 边界处理
    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()
    if len(ub) == 1: ub = np.full(dim, ub[0])
    if len(lb) == 1: lb = np.full(dim, lb[0])

    # 初始化
    gb_sol = np.zeros(dim)
    gb_fit = np.inf
    conv_curve = []

    Ep = 0.7  # 环境概率

    X = initialization(pop_size, dim, ub, lb)
    fitness = np.zeros(pop_size)

    # 初始评估
    evaluations = 0
    for i in range(pop_size):
        fitness[i] = cec_instance.evaluate(X[i, :], fobj)
        evaluations += 1
        if fitness[i] < gb_fit:
            gb_fit = fitness[i]
            gb_sol = X[i, :].copy()

    # 记录初始状态
    conv_curve.append(gb_fit)

    # 计算最大迭代次数
    max_iter = max_fes / pop_size
    current_iter = 0

    # 主循环
    while evaluations < max_fes:
        current_iter += 1

        # 环境因子 E
        E = 1 - (current_iter / max_iter)

        X_new = X.copy()

        for i in range(pop_size):
            # 随机选择 a, b, c
            candidates = list(range(pop_size))
            candidates.remove(i)
            # 确保样本足够
            if len(candidates) >= 3:
                a, b, c = np.random.choice(candidates, 3, replace=False)
            else:
                a, b, c = candidates[0], candidates[0], candidates[0]

            for j in range(dim):
                sig = 1 if np.random.rand() < 0.5 else -1
                mu = sig * np.random.rand() * E

                if np.random.rand() < Ep:
                    # Exploration
                    X_new[i, j] = X[b, j] + mu * (X[c, j] - X[a, j])
                else:
                    # Exploitation
                    weight = current_iter / max_iter
                    term1 = (weight * gb_sol[j] + (1 - weight) * X[a, j])
                    term2 = X[b, j]
                    mean_val = (term1 + term2) / 2.0
                    diff_val = mu * np.abs((X[c, j] + X[a, j] + X[b, j]) / 3.0 - X[i, j])
                    X_new[i, j] = mean_val + diff_val

            # 边界检查
            X_new[i] = np.clip(X_new[i], lb, ub)

            # 评估与更新
            if evaluations < max_fes:
                new_fit = cec_instance.evaluate(X_new[i, :], fobj)
                evaluations += 1

                if new_fit < fitness[i]:
                    fitness[i] = new_fit
                    X[i, :] = X_new[i, :].copy()

                    if new_fit < gb_fit:
                        gb_fit = new_fit
                        gb_sol = X[i, :].copy()
            else:
                break

        # 每代记录一次收敛值
        conv_curve.append(gb_fit)

    return gb_fit, gb_sol, np.array(conv_curve)


#并行 Worker
def worker_fgo(run_idx, cp_no, max_fes, ub, lb, dim, func_id):

    local_cec = CEC2014(dim)

    # 运行算法
    best_score, best_pos, conv_curve = FGO(cp_no, max_fes, ub, lb, dim, func_id, local_cec)

    return run_idx, best_score, conv_curve


#运行控制与测试代码
def run_single_function(func_id, CP_no=50, MaxFEs=25050, RUN_NO=20, dim=10, plot=True):
    """
    单函数测试
    """
    start_time = time.time()
    lb, ub, dim = get_functions_details_cec(func_id)
    optimal = func_id * 100

    # 容器初始化
    results_list = [None] * RUN_NO  # 用于按顺序存储结果

    # 并行执行
    with ProcessPoolExecutor() as executor:
        futures = []
        for j in range(RUN_NO):
            futures.append(executor.submit(worker_fgo, j, CP_no, MaxFEs, ub, lb, dim, func_id))

        # 结果
        for future in as_completed(futures):
            r_idx, r_score, r_curve = future.result()
            results_list[r_idx] = (r_score, r_curve)

            error = r_score - optimal
            print(f"  Run {r_idx + 1:2d}/{RUN_NO} Completed. Fitness = {r_score:.4e}")

    # 整理数据
    fitness = np.zeros(RUN_NO)
    convergence_curves = []

    for i in range(RUN_NO):
        if results_list[i] is not None:
            fitness[i] = results_list[i][0]
            convergence_curves.append(results_list[i][1])
        else:
            fitness[i] = np.inf  # 标记失败

    # 导出数据到 CSV
    save_data_to_csv(func_id, convergence_curves, RUN_NO)

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / RUN_NO

    # 结果统计
    valid_fitness = fitness[fitness != np.inf]

    # 初始化统计变量
    avg_error = np.inf
    std_error = np.inf
    best_error_val = np.inf

    if len(valid_fitness) > 0:
        best_fitness = np.min(valid_fitness)
        worst_fitness = np.max(valid_fitness)
        avg_fitness = np.mean(valid_fitness)
        std_fitness = np.std(valid_fitness)

        avg_error = avg_fitness - optimal
        std_error = std_fitness
        best_error_val = best_fitness - optimal
    else:
        # 如果全部失败
        avg_error = np.inf
        std_error = np.inf
        best_fitness = np.inf
        best_error_val = np.inf

    print(f"函数 F{func_id} 统计:")
    print(f"  平均误差: {avg_error:.4e}")
    print(f"  标准差:   {std_error:.4e}")
    print(f"  最优误差: {best_error_val:.4e}")
    print(f"  总耗时:   {elapsed_time:.2f}s")

    # 绘图并保存
    if plot and convergence_curves:
        plt.figure(figsize=(10, 6))

        min_len = min(len(c) for c in convergence_curves)
        trimmed_curves = [c[:min_len] for c in convergence_curves]
        iterations = np.arange(1, min_len + 1)

        # 绘制所有运行的灰色细线
        for curve in trimmed_curves:
            plt.semilogy(iterations, curve, 'lightgray', alpha=0.5, linewidth=0.5)

        # 绘制平均线
        mean_curve = np.mean(trimmed_curves, axis=0)
        plt.semilogy(iterations, mean_curve, 'b-', linewidth=2, label='Mean Fitness')

        # 绘制最佳运行线
        best_run_idx = np.argmin(fitness)
        plt.semilogy(iterations, trimmed_curves[best_run_idx], 'r--', linewidth=1.5, label='Best Run')

        plt.xlabel('Generations')
        plt.ylabel('Fitness (Log Scale)')
        plt.title(f'FGO Convergence - F{func_id}')
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        # 保存图片
        img_name = os.path.join(DATA_DIR, f'FGO_F{func_id:02d}_Plot.png')
        plt.savefig(img_name, dpi=1200)
        plt.close()


def main():

    FUNC_NUM = 1  # 单次测试的函数ID
    DIM = 10  # 维度
    POP_SIZE = 50  # 种群大小
    MAX_FES = 25050  # 最大评估次数
    N_RUNS = 20  # 独立运行次数

    # "single" 跑单个, "full" 跑 F1-F30
    TEST_MODE = "full"

    if TEST_MODE == "single":
        run_single_function(
            func_id=FUNC_NUM,
            CP_no=POP_SIZE,
            MaxFEs=MAX_FES,
            RUN_NO=N_RUNS,
            dim=DIM,
            plot=True
        )
    elif TEST_MODE == "full":
        Fun_id = list(range(1, 31))
        for fid in Fun_id:
            run_single_function(fid, POP_SIZE, MAX_FES, N_RUNS, DIM, plot=True)

if __name__ == "__main__":
    main()