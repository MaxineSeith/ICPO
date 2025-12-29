import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os

# ==========================================
#   1. 全局科研绘图样式配置
# ==========================================
# 加载基础样式
base_style = 'seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot'
plt.style.use(base_style)

# 强制更新参数 (字体、线宽、图例等)
rcParams.update({
    # --- 字体设置 ---
    'font.family': 'sans-serif',
    'font.sans-serif': ['Times New Roman', 'Arial', 'SimHei', 'Microsoft YaHei', 'sans-serif'],
    'axes.unicode_minus': False,

    # --- 尺寸与线条 ---
    'figure.figsize': (10, 6),
    'lines.linewidth': 2.5,       # 您的要求: 2.5
    'axes.linewidth': 2.0,        # 您的要求: 2.0
    'axes.edgecolor': 'black',

    # --- 字体大小 ---
    'axes.labelsize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',

    # --- 图例设置 ---
    'legend.fontsize': 18,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
})

# ==========================================
#   2. 配色方案 (Facecolor + Edgecolor)
# ==========================================
# 蓝色系 (用于 CPO)
style_cpo = {
    'color': '#1E90FF',       # 线条颜色
    'edgecolor': '#0000CD',   # Marker 边框色
    'facecolor': '#87CEFA',   # Marker 填充色
    'marker': 'o'             # Marker 形状
}

# 红色系 (用于 ICPO)
style_icpo = {
    'color': '#FF4500',       # 线条颜色
    'edgecolor': '#8B0000',   # Marker 边框色
    'facecolor': '#FFA07A',   # Marker 填充色
    'marker': '^'             # Marker 形状
}

# ==========================================
#   3. 准备工作
# ==========================================
figure_save_path = "Comparison_Curves_Final_Fixed"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)
    print(f"创建图片保存目录: {figure_save_path}")

# ★★★ 关键修正：关闭交互模式，防止PDF空白 ★★★
plt.ioff()

def get_optimal_value(func_id):
    return func_id * 100

# ==========================================
#   4. 批量绘图主程序
# ==========================================
print(f"{'='*60}")
print(f"开始绘制: CPO vs ICPO (修复空白问题 + 新配色)")
print(f"{'='*60}")

# 根据您的文件数量修改 range，例如 range(1, 31)
for i in range(1, 31):
    try:
        # --- 4.1 文件名构造 ---
        func_str = f"F{i:02d}"
        cpo_file = f"CPO_{func_str}_Data.csv"
        icpo_file = f"ICPO_{func_str}_Data.csv"

        # 检查文件
        if not os.path.exists(cpo_file) or not os.path.exists(icpo_file):
            print(f"[跳过] 缺少文件: {func_str}")
            continue

        print(f"正在处理 {func_str} ...")

        # --- 4.2 读取数据 ---
        # 使用 loadtxt 保证读取原始数值
        cpo_raw = np.loadtxt(cpo_file, delimiter=',')
        icpo_raw = np.loadtxt(icpo_file, delimiter=',')

        # 维度兼容性处理
        if cpo_raw.ndim == 1: cpo_raw = cpo_raw.reshape(1, -1)
        if icpo_raw.ndim == 1: icpo_raw = icpo_raw.reshape(1, -1)

        # 计算均值
        mean_cpo = np.mean(cpo_raw, axis=0)
        mean_icpo = np.mean(icpo_raw, axis=0)

        # --- 4.3 确定数据长度 (25050) ---
        data_len = len(mean_cpo)
        iterations = np.arange(1, data_len + 1)

        # 智能计算 Marker 间隔 (防止太密集)
        mark_every = max(1, data_len // 15)

        # --- 4.4 创建画布 ---
        fig, ax = plt.subplots()

        # [A] 绘制基准线
        optimal_val = get_optimal_value(i)
        ax.axhline(y=optimal_val, color='green', linestyle='--', linewidth=2,
                   label=f'Optimal ({optimal_val})', alpha=0.8, zorder=1)

        # [B] 绘制 CPO (蓝色系)
        ax.plot(
            iterations, mean_cpo,
            linestyle='-',
            linewidth=rcParams['lines.linewidth'],
            marker=style_cpo['marker'],
            markersize=7,
            markevery=mark_every,

            # 配色应用
            color=style_cpo['color'],
            markeredgecolor=style_cpo['edgecolor'],
            markerfacecolor=style_cpo['facecolor'],
            markeredgewidth=1.5,

            label='CPO', alpha=0.9,
            zorder=10, clip_on=False  # 防止遮挡
        )

        # [C] 绘制 ICPO (红色系)
        ax.plot(
            iterations, mean_icpo,
            linestyle='-',
            linewidth=rcParams['lines.linewidth'],
            marker=style_icpo['marker'],
            markersize=8,
            markevery=mark_every,

            # 配色应用
            color=style_icpo['color'],
            markeredgecolor=style_icpo['edgecolor'],
            markerfacecolor=style_icpo['facecolor'],
            markeredgewidth=1.5,

            label='ICPO', alpha=0.9,
            zorder=10, clip_on=False # 防止遮挡
        )

        # --- 4.5 坐标轴逻辑 (完美边缘) ---
        ax.set_title(f'{func_str} Convergence Analysis', pad=15)
        ax.set_xlabel('Function Evaluations (FES)')
        ax.set_ylabel('Mean Fitness (Log Scale)')
        ax.set_yscale('log')

        # 1. 设置 X 轴范围 (留出 2% 白边)
        x_max_padding = data_len * 1.02
        ax.set_xlim(0, x_max_padding)

        # 2. 强制刻度包含终点 (例如 25050)
        step = 5000
        ticks = list(range(0, data_len + 1, step))
        # 如果最后一个刻度不是终点，且距离不近，则添加终点
        if ticks[-1] != data_len:
            if data_len - ticks[-1] < step * 0.2:
                ticks[-1] = data_len
            else:
                ticks.append(data_len)
        ax.set_xticks(ticks)

        # 网格
        ax.grid(True, which='major', linestyle='-', linewidth=0.75, color='gray', alpha=0.3, zorder=0)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.2, zorder=0)

        # 图例
        legend = ax.legend(loc='best')
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('black')

        plt.tight_layout()

        # --- 4.6 保存逻辑 (修复空白问题的核心) ---

        pdf_name = os.path.join(figure_save_path, f"Compare_{func_str}.pdf")
        png_name = os.path.join(figure_save_path, f"Compare_{func_str}.png")

        # 使用 fig.savefig 显式保存当前对象
        fig.savefig(pdf_name, dpi=300, bbox_inches='tight')
        fig.savefig(png_name, dpi=150, bbox_inches='tight')

        # 显式关闭，释放内存
        plt.close(fig)

    except Exception as e:
        print(f"!! 处理 {func_str} 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        continue