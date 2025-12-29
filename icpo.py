import numpy as np
import matplotlib.pyplot as plt
from cec2014 import CEC2014
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def initialization(search_agents_no, dim, ub, lb):

    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()

    boundary_no = len(ub)

    # 如果所有变量的边界相同
    if boundary_no == 1:
        positions = np.random.rand(search_agents_no, dim) * (ub[0] - lb[0]) + lb[0]
    else:
        # 如果每个变量有不同的边界
        positions = np.zeros((search_agents_no, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = np.random.rand(search_agents_no) * (ub_i - lb_i) + lb_i

    return positions


def get_functions_details_cec(func_id):

    dim = 10

    if func_id == 11:
        lb = -600 * np.ones(dim)
        ub = 600 * np.ones(dim)
    else:
        lb = -100 * np.ones(dim)
        ub = 100 * np.ones(dim)

    return lb, ub, dim


def Improved_CPO(pop_size, t_max, ub, lb, dim, fobj, cec_instance):
    """
    Improved Crested Porcupine Optimizer
    """
    # 边界处理
    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()
    if len(ub) == 1:
        ub = np.full(dim, ub[0])
    if len(lb) == 1:
        lb = np.full(dim, lb[0])

    # 初始化
    gb_sol = np.zeros(dim)
    gb_fit = np.inf
    conv_curve = np.zeros(t_max)

    # 参数设置
    N = pop_size
    N_min = max(4, int(0.1 * pop_size))  # 最小种群大小
    T = 2  # 种群缩减周期
    alpha = 0.5
    Tf = 0.8

    # 策略2: 外部存档初始化
    archive = []
    max_archive_size = int(1.4 * pop_size)  # 与 LSHADE 一致

    # 策略4: 参数记忆初始化 (SHADE-style)
    H = 5  # 记忆大小
    memory_F = [0.5] * H
    memory_CR = [0.5] * H
    memory_pos = 0

    # 初始化种群
    X = initialization(pop_size, dim, ub, lb)
    fitness = np.zeros(pop_size)

    for i in range(pop_size):
        fitness[i] = cec_instance.evaluate(X[i, :], fobj)

    # 初始排序 (为了 P-best)
    sorted_idx = np.argsort(fitness)
    X = X[sorted_idx]
    fitness = fitness[sorted_idx]

    gb_fit = fitness[0]
    gb_sol = X[0, :].copy()
    Xp = X.copy()  # 个人最佳位置

    t = 0
    active_pop_size = pop_size
    opt = fobj * 100  # CEC2014 最优值

    # 主循环
    while t < t_max and gb_fit != opt:
        r2 = np.random.rand()

        # 策略3: 精英保留的循环种群缩减
        rem_value = t % (t_max / T)
        cycle_progress = rem_value / (t_max / T)

        # 计算当前种群大小
        active_pop_size = int(N_min + (N - N_min) * (1 - cycle_progress))
        active_pop_size = max(N_min, min(active_pop_size, N))

        # 关键: 排序确保缩减时移除差解
        sorted_idx = np.argsort(fitness[:pop_size])
        X[:pop_size] = X[sorted_idx]
        fitness[:pop_size] = fitness[sorted_idx]
        Xp[:pop_size] = Xp[sorted_idx]

        # 策略1: P-best 参数 (前 15% 的优良个体)
        p_best_size = max(2, int(0.15 * active_pop_size))

        # 策略4: 生成自适应参数
        F_list = []
        CR_list = []
        for _ in range(active_pop_size):
            ri = np.random.randint(0, H)
            mu_F = memory_F[ri]
            mu_CR = memory_CR[ri]

            # 生成 F (Cauchy 分布)
            F = -1
            while F <= 0:
                F = np.random.standard_cauchy() * 0.1 + mu_F
            F = min(F, 1.0)

            # 生成 CR (Normal 分布)
            CR = np.random.normal(mu_CR, 0.1)
            CR = np.clip(CR, 0, 1)

            F_list.append(F)
            CR_list.append(CR)

        # 记录成功的参数
        S_F = []
        S_CR = []
        delta_f = []

        for i in range(active_pop_size):
            old_pos = X[i, :].copy()
            old_fitness = fitness[i]

            U1 = (np.random.rand(dim) > np.random.rand()).astype(float)

            if np.random.rand() < np.random.rand():  # 探索阶段 (保持原版多样性)
                if np.random.rand() < np.random.rand():  # 第一防御机制
                    rand_idx = np.random.randint(active_pop_size)
                    y = (X[i, :] + X[rand_idx, :]) / 2
                    X[i, :] = X[i, :] + np.random.randn() * np.abs(2 * np.random.rand() * gb_sol - y)
                else:  # 第二防御机制
                    rand_idx = np.random.randint(active_pop_size)
                    y = (X[i, :] + X[rand_idx, :]) / 2
                    rand_idx1 = np.random.randint(active_pop_size)
                    rand_idx2 = np.random.randint(active_pop_size)
                    X[i, :] = U1 * X[i, :] + (1 - U1) * (y + np.random.rand() * (X[rand_idx1, :] - X[rand_idx2, :]))
            else:  # 开发阶段
                Yt = 2 * np.random.rand() * (1 - t / t_max) ** (t / t_max)
                U2 = (np.random.rand(dim) < 0.5) * 2 - 1
                S = np.random.rand() * U2

                if np.random.rand() < Tf:  # 策略1 & 2: 改进的第三防御机制
                    St = np.exp(fitness[i] / (np.sum(fitness[:active_pop_size]) + 1e-20))
                    S = S * Yt * St

                    # 1. 选择 P-best (精英引导)
                    p_best_idx = np.random.randint(0, p_best_size)
                    x_pbest = X[p_best_idx, :]

                    # 2. 从 [种群 + 存档] 中选择个体 (多样性)
                    if len(archive) > 0:
                        cand_pool = np.vstack((X[:active_pop_size], np.array(archive)))
                    else:
                        cand_pool = X[:active_pop_size]

                    r1 = np.random.randint(active_pop_size)
                    r2_idx = np.random.randint(len(cand_pool))
                    x_r2 = cand_pool[r2_idx]

                    # 改进公式: current-to-pbest with archive
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (x_pbest + F_list[i] * St * (X[r1, :] - x_r2) - S)

                else:  # 第四防御机制 (物理攻击) - 保持原版
                    Mt = np.exp(fitness[i] / (np.sum(fitness[:active_pop_size]) + 1e-20))
                    vt = X[i, :]
                    rand_idx = np.random.randint(active_pop_size)
                    Vtp = X[rand_idx, :]
                    Ft = np.random.rand(dim) * (Mt * (-vt + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = (gb_sol + (alpha * (1 - r2) + r2) * (U2 * gb_sol - X[i, :])) - S

            # 边界处理
            X[i, :] = np.clip(X[i, :], lb, ub)

            # 评估
            nF = cec_instance.evaluate(X[i, :], fobj)

            # 策略2: 更新存档 & 贪婪选择
            if nF < fitness[i]:  # 发现更好的解
                # 将旧解存入存档
                if len(archive) < max_archive_size:
                    archive.append(old_pos)
                else:
                    k = np.random.randint(max_archive_size)
                    archive[k] = old_pos

                # 更新个人状态
                Xp[i, :] = X[i, :].copy()
                fitness[i] = nF

                # 记录成功参数
                S_F.append(F_list[i])
                S_CR.append(CR_list[i])
                delta_f.append(abs(old_fitness - nF))

                if nF <= gb_fit:
                    gb_sol = X[i, :].copy()
                    gb_fit = nF
            else:  # 新解更差，回退
                X[i, :] = Xp[i, :].copy()

            # 记录收敛曲线
            if t < t_max:
                conv_curve[t] = gb_fit

            t += 1
            if t >= t_max:
                break

        # 策略4: 更新参数记忆
        if len(S_F) > 0 and np.sum(delta_f) > 0:
            weights = np.array(delta_f) / np.sum(delta_f)

            # Lehmer mean for F
            sum_wF = np.sum(weights * np.array(S_F))
            if sum_wF > 1e-10:
                memory_F[memory_pos] = np.sum(weights * np.array(S_F) ** 2) / sum_wF

            # Lehmer mean for CR
            sum_wCR = np.sum(weights * np.array(S_CR))
            if sum_wCR > 1e-10:
                memory_CR[memory_pos] = np.sum(weights * np.array(S_CR) ** 2) / sum_wCR

            memory_pos = (memory_pos + 1) % H

    # 确保收敛曲线完整填充
    conv_curve[t:] = gb_fit

    return gb_fit, gb_sol, conv_curve


def CPO(pop_size, t_max, ub, lb, dim, fobj, cec_instance):
    """
    Crested Porcupine Optimizer
    """
    # 确保边界是数组
    ub = np.array(ub).flatten()
    lb = np.array(lb).flatten()
    if len(ub) == 1:
        ub = np.full(dim, ub[0])
    if len(lb) == 1:
        lb = np.full(dim, lb[0])

    # 初始化
    gb_sol = np.zeros(dim)  # 最佳解向量
    gb_fit = np.inf  # 最佳适应度值
    conv_curve = np.zeros(t_max)

    # 控制参数
    N = pop_size  # 初始种群大小
    N_min = 10  # 最小种群大小
    T = 2  # 周期数
    alpha = 0.5  # 收敛率
    Tf = 0.8  # 第三和第四防御机制之间的权衡百分比

    # 初始化种群位置
    X = initialization(pop_size, dim, ub, lb)
    t = 0  # 函数评估计数器

    # 评估初始种群
    fitness = np.zeros(pop_size)
    for i in range(pop_size):
        fitness[i] = cec_instance.evaluate(X[i, :], fobj)

    # 更新最佳解
    best_idx = np.argmin(fitness)
    gb_fit = fitness[best_idx]
    gb_sol = X[best_idx, :].copy()

    # 记录初始种群的适应度到收敛曲线
    for i in range(pop_size):
        if t < t_max:
            conv_curve[t] = gb_fit
        t += 1

    # 存储个人最佳位置
    Xp = X.copy()

    # 已知最优适应度
    opt = fobj * 100  # CEC2014的最优值

    # 种群大小管理
    active_pop_size = pop_size  # 当前活跃的种群大小

    # 优化过程
    while t <= t_max and gb_fit != opt:
        r2 = np.random.rand()

        for i in range(active_pop_size):
            U1 = (np.random.rand(dim) > np.random.rand()).astype(float)

            if np.random.rand() < np.random.rand():  # 探索阶段
                if np.random.rand() < np.random.rand():  # 第一防御机制
                    # 计算 y_t
                    rand_idx = np.random.randint(active_pop_size)
                    y = (X[i, :] + X[rand_idx, :]) / 2
                    X[i, :] = X[i, :] + np.random.randn() * np.abs(2 * np.random.rand() * gb_sol - y)
                else:  # 第二防御机制
                    rand_idx = np.random.randint(active_pop_size)
                    y = (X[i, :] + X[rand_idx, :]) / 2
                    rand_idx1 = np.random.randint(active_pop_size)
                    rand_idx2 = np.random.randint(active_pop_size)
                    X[i, :] = U1 * X[i, :] + (1 - U1) * (y + np.random.rand() * (X[rand_idx1, :] - X[rand_idx2, :]))
                    # X[i, :] = (1 - U1) * X[i, :] + U1 * (y + np.random.rand() * (X[rand_idx1, :] - X[rand_idx2, :]))
            else:
                Yt = 2 * np.random.rand() * (1 - t / t_max) ** (t / t_max)
                U2 = (np.random.rand(dim) < 0.5) * 2 - 1
                S = np.random.rand() * U2

                if np.random.rand() < Tf:  # 第三防御机制
                    St = np.exp(fitness[i] / (np.sum(fitness[:active_pop_size]) + np.finfo(float).eps))
                    S = S * Yt * St
                    rand_idx1 = np.random.randint(active_pop_size)
                    rand_idx2 = np.random.randint(active_pop_size)
                    rand_idx3 = np.random.randint(active_pop_size)
                    X[i, :] = (1 - U1) * X[i, :] + U1 * (X[rand_idx1, :] + St * (X[rand_idx2, :] - X[rand_idx3, :]) - S)
                else:  # 第四防御机制
                    Mt = np.exp(fitness[i] / (np.sum(fitness[:active_pop_size]) + np.finfo(float).eps))
                    vt = X[i, :]
                    rand_idx = np.random.randint(active_pop_size)
                    Vtp = X[rand_idx, :]
                    Ft = np.random.rand(dim) * (Mt * (-vt + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = (gb_sol + (alpha * (1 - r2) + r2) * (U2 * gb_sol - X[i, :])) - S

            # 边界处理
            for j in range(dim):
                if X[i, j] > ub[j]:
                    X[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])
                elif X[i, j] < lb[j]:
                    X[i, j] = lb[j] + np.random.rand() * (ub[j] - lb[j])

            # 计算新解的适应度值
            nF = cec_instance.evaluate(X[i, :], fobj)

            # 更新全局和个人最佳解
            if fitness[i] < nF:
                X[i, :] = Xp[i, :].copy()  # 保留个人最佳
            else:
                Xp[i, :] = X[i, :].copy()
                fitness[i] = nF
                # 更新全局最佳
                if fitness[i] <= gb_fit:
                    gb_sol = X[i, :].copy()
                    gb_fit = fitness[i]

            # 记录收敛曲线
            if t < t_max:
                conv_curve[t] = gb_fit

            t += 1
            if t > t_max:
                break

        # 更新种群大小
        if t <= t_max:
            rem_value = t % (t_max / T)

            active_pop_size = int(N_min + (N - N_min) * (1 - (rem_value / (t_max / T))))
            active_pop_size = max(active_pop_size, N_min)
            active_pop_size = min(active_pop_size, N)  # 不能超过初始种群大小

    # 确保收敛曲线完整填充
    conv_curve[t:] = gb_fit

    return gb_fit, gb_sol, conv_curve


def compare_CPO_vs_ICPO(func_id, CP_no=50, Tmax=25000, RUN_NO=30, dim=10):
    """
    比较原版 CPO 和改进版 ICPO 的性能
    """
    print(f"性能对比测试: CPO vs ICPO - F{func_id}")
    print(f"种群大小: {CP_no} | 最大FES: {Tmax} | 运行次数: {RUN_NO} | 维度: {dim}")

    # 初始化CEC2014
    cec_instance = CEC2014(dim)
    lb, ub, dim = get_functions_details_cec(func_id)
    optimal = func_id * 100

    # 存储结果
    cpo_fitness = np.zeros(RUN_NO)
    icpo_fitness = np.zeros(RUN_NO)
    cpo_curves = []
    icpo_curves = []

    print("CPO")
    cpo_start = time.time()
    for j in range(RUN_NO):
        best_score, _, conv_curve = CPO(CP_no, Tmax, ub, lb, dim, func_id, cec_instance)
        cpo_fitness[j] = best_score
        cpo_curves.append(conv_curve)
        print(f"  Run {j + 1:2d}/{RUN_NO}: {best_score:.6e} (误差: {best_score - optimal:.6e})")
    cpo_time = time.time() - cpo_start

    # 导出CPO数据
    cpo_data = np.array(cpo_curves)
    np.savetxt(f'CPO_F{func_id:02d}_Data.csv', cpo_data, delimiter=',')

    print("ICPO")
    icpo_start = time.time()
    for j in range(RUN_NO):
        best_score, _, conv_curve = Improved_CPO(CP_no, Tmax, ub, lb, dim, func_id, cec_instance)
        icpo_fitness[j] = best_score
        icpo_curves.append(conv_curve)
        print(f"  Run {j + 1:2d}/{RUN_NO}: {best_score:.6e} (误差: {best_score - optimal:.6e})")
    icpo_time = time.time() - icpo_start

    # 导出ICPO数据
    icpo_data = np.array(icpo_curves)
    np.savetxt(f'ICPO_F{func_id:02d}_Data.csv', icpo_data, delimiter=',')

    # 统计分析
    print(f"{'指标':<15} | {'CPO':<20} | {'ICPO':<20} | {'改进率':<15}")

    cpo_mean = np.mean(cpo_fitness)
    icpo_mean = np.mean(icpo_fitness)
    mean_improve = ((cpo_mean - icpo_mean) / abs(cpo_mean - optimal)) * 100 if abs(cpo_mean - optimal) > 1e-10 else 0
    print(f"{'平均值':<15} | {cpo_mean:<20.6e} | {icpo_mean:<20.6e} | {mean_improve:>+14.2f}%")

    cpo_std = np.std(cpo_fitness)
    icpo_std = np.std(icpo_fitness)
    std_improve = ((cpo_std - icpo_std) / cpo_std) * 100 if cpo_std > 1e-10 else 0
    print(f"{'标准差':<15} | {cpo_std:<20.6e} | {icpo_std:<20.6e} | {std_improve:>+14.2f}%")

    cpo_best = np.min(cpo_fitness)
    icpo_best = np.min(icpo_fitness)
    best_improve = ((cpo_best - icpo_best) / abs(cpo_best - optimal)) * 100 if abs(cpo_best - optimal) > 1e-10 else 0
    print(f"{'最佳值':<15} | {cpo_best:<20.6e} | {icpo_best:<20.6e} | {best_improve:>+14.2f}%")

    cpo_worst = np.max(cpo_fitness)
    icpo_worst = np.max(icpo_fitness)
    print(f"{'最差值':<15} | {cpo_worst:<20.6e} | {icpo_worst:<20.6e} | {'-':<15}")

    print(f"{'平均时间(s)':<15} | {cpo_time / RUN_NO:<20.2f} | {icpo_time / RUN_NO:<20.2f} | {'-':<15}")

    # 绘制对比收敛曲线
    plt.figure(figsize=(12, 6))

    # 计算平均收敛曲线
    cpo_mean_curve = np.mean(cpo_curves, axis=0)
    icpo_mean_curve = np.mean(icpo_curves, axis=0)

    # 绘制所有运行的曲线
    for curve in cpo_curves:
        plt.semilogy(curve, 'lightcoral', alpha=0.2, linewidth=0.5)
    for curve in icpo_curves:
        plt.semilogy(curve, 'lightblue', alpha=0.2, linewidth=0.5)

    # 绘制平均曲线
    plt.semilogy(cpo_mean_curve, 'r-', linewidth=2.5, label='CPO')
    plt.semilogy(icpo_mean_curve, 'b-', linewidth=2.5, label='ICPO')

    # 绘制最优值参考线
    plt.axhline(y=optimal, color='g', linestyle='--', linewidth=1.5, alpha=0.7, label=f'optimal value ({optimal})')

    plt.xlabel('Function Evaluations', fontsize=12)
    plt.ylabel('Best-so-far Fitness (log scale)', fontsize=12)
    plt.title(f'CPO vs ICPO on F{func_id}: {cec_instance.get_name(func_id)}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    filename = f'CPO_vs_ICPO_F{func_id:02d}_comparison.png'
    plt.savefig(filename, dpi=1200, bbox_inches='tight')
    plt.show()

    return {
        'cpo': {'mean': cpo_mean, 'std': cpo_std, 'best': cpo_best, 'worst': cpo_worst},
        'icpo': {'mean': icpo_mean, 'std': icpo_std, 'best': icpo_best, 'worst': icpo_worst}
    }


if __name__ == "__main__":

    DIM = 10  # 问题维度
    MAX_FES = 25050  # 最大函数评估次数
    POP_SIZE = 50  # 种群大小
    N_RUNS = 20  # 运行次数

    FUNC_IDS = range(1,30)

    print(f"维度: {DIM}, 种群: {POP_SIZE}, 最大FES: {MAX_FES}, 运行次数: {N_RUNS}")

    # 存储所有统计结果
    all_final_stats = []

    for f_id in FUNC_IDS:

        # 运行对比测试
        stats = compare_CPO_vs_ICPO(
            func_id=f_id,
            CP_no=POP_SIZE,
            Tmax=MAX_FES,
            RUN_NO=N_RUNS,
            dim=DIM
        )
        stats['func_id'] = f_id
        all_final_stats.append(stats)

        plt.close('all')


    # 最终汇总打印
    print(f"{'F_ID':<5} | {'CPO Mean':<20} | {'ICPO Mean':<20} | {'Winner':<10}")

    for s in all_final_stats:
        f_id = s['func_id']
        c_mean = s['cpo']['mean']
        i_mean = s['icpo']['mean']

        if i_mean < c_mean:
            winner = "ICPO"
        elif c_mean < i_mean:
            winner = "CPO"
        else:
            winner = "Draw"

        print(f"F{f_id:<02d}   | {c_mean:<20.6e} | {i_mean:<20.6e} | {winner:<10}")