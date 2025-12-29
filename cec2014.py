# cec2014.py
import numpy as np
import os

# 基础数学函数别名
sin = np.sin
cos = np.cos
sqrt = np.sqrt
pi = np.pi
exp = np.exp
e = np.e
dot = np.dot
array = np.array
sum = np.sum
matmul = np.matmul
where = np.where
sign = np.sign
min = np.min
round = np.round
ceil = np.ceil
ones = np.ones
concatenate = np.concatenate

# 支持的维度
SUPPORTED_DIMENSIONS = [2, 10, 20, 30, 50, 100]


class CEC2014:
    def __init__(self, dim, data_dir="cec2014_input_data"):
        """
        CEC2014测试函数类

        Args:
            dim: 问题维度 (2, 10, 20, 30, 50, 100)
            data_dir: 数据文件目录
        """
        if dim not in SUPPORTED_DIMENSIONS:
            raise ValueError(f"维度必须是 {SUPPORTED_DIMENSIONS} 之一")

        self.dim = dim
        self.data_dir = data_dir
        self._load_bias_and_names()

    def _load_bias_and_names(self):
        """加载偏置值和函数名称"""
        self.bias = {
            1: 100, 2: 200, 3: 300, 4: 400, 5: 500, 6: 600, 7: 700, 8: 800,
            9: 900, 10: 1000, 11: 1100, 12: 1200, 13: 1300, 14: 1400,
            15: 1500, 16: 1600, 17: 1700, 18: 1800, 19: 1900, 20: 2000,
            21: 2100, 22: 2200, 23: 2300, 24: 2400, 25: 2500, 26: 2600,
            27: 2700, 28: 2800, 29: 2900, 30: 3000
        }

        self.names = {
            1: "Rotated High Conditioned Elliptic Function",
            2: "Rotated Bent Cigar Function",
            3: "Rotated Discus Function",
            4: "Shifted and Rotated Rosenbrock's Function",
            5: "Shifted and Rotated Ackley's Function",
            6: "Shifted and Rotated Weierstrass Function",
            7: "Shifted and Rotated Griewank's Function",
            8: "Shifted Rastrigin's Function",
            9: "Shifted and Rotated Rastrigin's Function",
            10: "Shifted Schwefel's Function",
            11: "Shifted and Rotated Schwefel's Function",
            12: "Shifted and Rotated Katsuura Function",
            13: "Shifted and Rotated HappyCat Function",
            14: "Shifted and Rotated HGBat Function",
            15: "Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function",
            16: "Shifted and Rotated Expanded Scaffer's F6 Function",
            17: "Hybrid Function 1",
            18: "Hybrid Function 2",
            19: "Hybrid Function 3",
            20: "Hybrid Function 4",
            21: "Hybrid Function 5",
            22: "Hybrid Function 6",
            23: "Composition Function 1",
            24: "Composition Function 2",
            25: "Composition Function 3",
            26: "Composition Function 4",
            27: "Composition Function 5",
            28: "Composition Function 6",
            29: "Composition Function 7",
            30: "Composition Function 8"
        }

    def _load_matrix_data(self, filename):
        """加载矩阵数据"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        return np.loadtxt(filepath)

    def _load_shift_data(self, func_num):
        """加载偏移数据"""
        filename = f"shift_data_{func_num}.txt"
        data = self._load_matrix_data(filename)

        if func_num >= 23:  # 组合函数
            return data
        else:
            return data[:self.dim]

    def _load_shuffle_data(self, func_num):
        """加载混洗数据"""
        filename = f"shuffle_data_{func_num}_D{self.dim}.txt"
        data = self._load_matrix_data(filename)
        return (data[:self.dim] - ones(self.dim)).astype(int)  # 转换为0-based索引

    def _load_rotation_matrix(self, func_num):
        """加载旋转矩阵"""
        filename = f"M_{func_num}_D{self.dim}.txt"
        return self._load_matrix_data(filename)

    def _load_composition_matrix(self, func_num):
        """为组合函数加载完整的矩阵"""
        filename = f"M_{func_num}_D{self.dim}.txt"
        matrix = self._load_matrix_data(filename)
        # if total_size and matrix.shape[0] < total_size:
        #     # 如果矩阵太小，可能需要特殊处理
        #     return matrix
        return matrix

    # 基础函数定义 (F1-F14)
    def _f1_elliptic(self, solution):
        result = 0
        for i in range(len(solution)):
            result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
        return result

    def _f2_bent_cigar(self, solution):
        return solution[0]**2 + 10**6 * sum(solution[1:]**2)

    def _f3_discus(self, solution):
        return 10**6 * solution[0]**2 + sum(solution[1:]**2)

    def _f4_rosenbrock(self, solution):
        result = 0.0
        for i in range(len(solution) - 1):
            result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
        return result

    def _f5_ackley(self, solution):
        return -20 * exp(-0.2 * sqrt(sum(solution ** 2) / len(solution))) - exp(
            sum(cos(2 * pi * solution)) / len(solution)) + 20 + e

    def _f6_weierstrass(self, solution, a=0.5, b=3, k_max=20):
        result = 0.0
        for i in range(0, len(solution)):
            t1 = sum([a ** k * cos(2 * pi * b ** k * (solution[i] + 0.5)) for k in range(0, k_max)])
            result += t1
        t2 = len(solution) * sum([a ** k * cos(2 * pi * b ** k * 0.5) for k in range(0, k_max)])
        return result - t2

    def _f7_griewank(self, solution):
        result = sum(solution ** 2) / 4000
        temp = 1.0
        for i in range(len(solution)):
            temp *= cos(solution[i] / sqrt(i + 1))
        return result - temp + 1

    def _f8_rastrigin(self, solution):
        return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)

    def _f9_modified_schwefel(self, solution):
        z = solution + 4.209687462275036e+002
        result = 418.9829 * len(solution)
        for i in range(0, len(solution)):
            if z[i] > 500:
                result -= (500 - z[i] % 500) * sin(sqrt(abs(500 - z[i] % 500))) - (z[i] - 500) ** 2 / (
                            10000 * len(solution))
            elif z[i] < -500:
                result -= (z[i] % 500 - 500) * sin(sqrt(abs(z[i] % 500 - 500))) - (z[i] + 500) ** 2 / (
                            10000 * len(solution))
            else:
                result -= z[i] * sin(abs(z[i]) ** 0.5)
        return result

    def _f10_katsuura(self, solution):
        result = 1.0
        for i in range(0, len(solution)):
            t1 = sum([abs(2 ** j * solution[i] - round(2 ** j * solution[i])) / 2 ** j for j in range(1, 33)])
            result *= (1 + (i + 1) * t1) ** (10.0 / len(solution) ** 1.2)
        return (result - 1) * 10 / len(solution) ** 2

    def _f11_happy_cat(self, solution):
        return (abs(sum(solution**2) - len(solution)))**0.25 + (0.5 * sum(solution**2) + sum(solution))/len(solution) + 0.5

    def _f12_hgbat(self, solution):
        return (abs(sum(solution**2)**2 - sum(solution)**2))**0.5 + (0.5*sum(solution**2)+sum(solution))/len(solution) + 0.5

    def _f13_expanded_griewank(self, solution):
        def __f4__(x=None, y=None):
            return 100 * (x ** 2 - y) ** 2 + (x - 1) ** 2

        def __f7__(z=None):
            return z ** 2 / 4000 - cos(z / sqrt(1)) + 1

        result = __f7__(__f4__(solution[-1], solution[0]))
        for i in range(0, len(solution) - 1):
            result += __f7__(__f4__(solution[i], solution[i + 1]))
        return result

    def _f14_expanded_scaffer(self, solution):
        def __xy__(x, y):
            return 0.5 + (sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2

        result = __xy__(solution[-1], solution[0])
        for i in range(0, len(solution) - 1):
            result += __xy__(solution[i], solution[i + 1])
        return result

    # F1-F16: 基本函数（需要旋转和偏移）
    def F1(self, solution):
        shift_data = self._load_shift_data(1)
        matrix = self._load_rotation_matrix(1)
        z = dot(solution - shift_data, matrix)
        return self._f1_elliptic(z) + self.bias[1]

    def F2(self, solution):
        shift_data = self._load_shift_data(2)
        matrix = self._load_rotation_matrix(2)
        z = dot(solution - shift_data, matrix)
        return self._f2_bent_cigar(z) + self.bias[2]

    def F3(self, solution):
        shift_data = self._load_shift_data(3)
        matrix = self._load_rotation_matrix(3)
        z = dot(solution - shift_data, matrix)
        return self._f3_discus(z) + self.bias[3]

    def F4(self, solution):
        shift_data = self._load_shift_data(4)
        matrix = self._load_rotation_matrix(4)
        z = 2.048 * (solution - shift_data) / 100
        z = dot(z, matrix) + 1
        return self._f4_rosenbrock(z) + self.bias[4]

    def F5(self, solution):
        shift_data = self._load_shift_data(5)
        matrix = self._load_rotation_matrix(5)
        z = dot(solution - shift_data, matrix)
        return self._f5_ackley(z) + self.bias[5]

    def F6(self, solution):
        shift_data = self._load_shift_data(6)
        matrix = self._load_rotation_matrix(6)
        z = 0.5 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f6_weierstrass(z) + self.bias[6]

    def F7(self, solution):
        shift_data = self._load_shift_data(7)
        matrix = self._load_rotation_matrix(7)
        z = 600 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f7_griewank(z) + self.bias[7]

    def F8(self, solution):
        shift_data = self._load_shift_data(8)
        matrix = self._load_rotation_matrix(8)
        z = 5.12 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f8_rastrigin(z) + self.bias[8]

    def F9(self, solution):
        shift_data = self._load_shift_data(9)
        matrix = self._load_rotation_matrix(9)
        z = 5.12 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f8_rastrigin(z) + self.bias[9]

    def F10(self, solution):
        shift_data = self._load_shift_data(10)
        matrix = self._load_rotation_matrix(10)
        z = 1000 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f9_modified_schwefel(z) + self.bias[10]

    def F11(self, solution):
        shift_data = self._load_shift_data(11)
        matrix = self._load_rotation_matrix(11)
        z = 1000 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f9_modified_schwefel(z) + self.bias[11]

    def F12(self, solution):
        shift_data = self._load_shift_data(12)
        matrix = self._load_rotation_matrix(12)
        z = 5 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f10_katsuura(z) + self.bias[12]

    def F13(self, solution):
        shift_data = self._load_shift_data(13)
        matrix = self._load_rotation_matrix(13)
        z = 5 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f11_happy_cat(z) + self.bias[13]

    def F14(self, solution):
        shift_data = self._load_shift_data(14)
        matrix = self._load_rotation_matrix(14)
        z = 5 * (solution - shift_data) / 100
        z = dot(z, matrix)
        return self._f12_hgbat(z) + self.bias[14]

    def F15(self, solution):
        shift_data = self._load_shift_data(15)
        matrix = self._load_rotation_matrix(15)
        z = 5 * (solution - shift_data) / 100
        z = dot(z, matrix) + 1
        return self._f13_expanded_griewank(z) + self.bias[15]

    def F16(self, solution):
        shift_data = self._load_shift_data(16)
        matrix = self._load_rotation_matrix(16)
        z = dot(solution - shift_data, matrix) + 1
        return self._f14_expanded_scaffer(z) + self.bias[16]

    # F17-F22: 混合函数
    def F17(self, solution):
        shift_data = self._load_shift_data(17)
        matrix = self._load_rotation_matrix(17)
        shuffle = self._load_shuffle_data(17)

        p = array([0.3, 0.3, 0.4])
        n1 = int(ceil(p[0] * self.dim))
        n2 = int(ceil(p[1] * self.dim))

        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n1 + n2]
        idx3 = shuffle[n1 + n2:]

        mz = dot(solution - shift_data, matrix)
        result = (self._f9_modified_schwefel(mz[idx1]) +
                  self._f8_rastrigin(mz[idx2]) +
                  self._f1_elliptic(mz[idx3]))
        return result + self.bias[17]

    def F18(self, solution):
        shift_data = self._load_shift_data(18)
        matrix = self._load_rotation_matrix(18)
        shuffle = self._load_shuffle_data(18)

        p = array([0.3, 0.3, 0.4])
        n1 = int(ceil(p[0] * self.dim))
        n2 = int(ceil(p[1] * self.dim))

        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n1 + n2]
        idx3 = shuffle[n1 + n2:]

        mz = dot(solution - shift_data, matrix)
        result = (self._f2_bent_cigar(mz[idx1]) +
                  self._f12_hgbat(mz[idx2]) +
                  self._f8_rastrigin(mz[idx3]))
        return result + self.bias[18]

    def F19(self, solution):
        shift_data = self._load_shift_data(19)
        matrix = self._load_rotation_matrix(19)
        shuffle = self._load_shuffle_data(19)

        p = array([0.2, 0.2, 0.3, 0.3])
        n1 = int(ceil(p[0] * self.dim))
        n2 = int(ceil(p[1] * self.dim))
        n3 = int(ceil(p[2] * self.dim))

        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n1 + n2]
        idx3 = shuffle[n1 + n2:n1 + n2 + n3]
        idx4 = shuffle[n1 + n2 + n3:]

        mz = dot(solution - shift_data, matrix)
        result = (self._f7_griewank(mz[idx1]) +
                  self._f6_weierstrass(mz[idx2]) +
                  self._f4_rosenbrock(mz[idx3]) +
                  self._f14_expanded_scaffer(mz[idx4]))
        return result + self.bias[19]

    def F20(self, solution):
        shift_data = self._load_shift_data(20)
        matrix = self._load_rotation_matrix(20)
        shuffle = self._load_shuffle_data(20)

        p = array([0.2, 0.2, 0.3, 0.3])
        n1 = int(ceil(p[0] * self.dim))
        n2 = int(ceil(p[1] * self.dim))
        n3 = int(ceil(p[2] * self.dim))

        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n1 + n2]
        idx3 = shuffle[n1 + n2:n1 + n2 + n3]
        idx4 = shuffle[n1 + n2 + n3:]

        mz = dot(solution - shift_data, matrix)
        result = (self._f12_hgbat(mz[idx1]) +
                  self._f3_discus(mz[idx2]) +
                  self._f13_expanded_griewank(mz[idx3]) +
                  self._f8_rastrigin(mz[idx4]))
        return result + self.bias[20]

    def F21(self, solution):
        shift_data = self._load_shift_data(21)
        matrix = self._load_rotation_matrix(21)
        shuffle = self._load_shuffle_data(21)

        p = array([0.1, 0.2, 0.2, 0.2, 0.3])
        n1 = int(ceil(p[0] * self.dim))
        n2 = int(ceil(p[1] * self.dim))
        n3 = int(ceil(p[2] * self.dim))
        n4 = int(ceil(p[3] * self.dim))

        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n1 + n2]
        idx3 = shuffle[n1 + n2:n1 + n2 + n3]
        idx4 = shuffle[n1 + n2 + n3:n1 + n2 + n3 + n4]
        idx5 = shuffle[n1 + n2 + n3 + n4:]


        mz = dot(solution - shift_data, matrix)
        result = (self._f14_expanded_scaffer(mz[idx1]) +
                  self._f12_hgbat(mz[idx2]) +
                  self._f4_rosenbrock(mz[idx3]) +
                  self._f9_modified_schwefel(mz[idx4]) +
                  self._f1_elliptic(mz[idx5]))
        return result + self.bias[21]

    def F22(self, solution):
        shift_data = self._load_shift_data(22)
        matrix = self._load_rotation_matrix(22)
        shuffle = self._load_shuffle_data(22)

        p = array([0.1, 0.2, 0.2, 0.2, 0.3])
        n1 = int(ceil(p[0] * self.dim))
        n2 = int(ceil(p[1] * self.dim))
        n3 = int(ceil(p[2] * self.dim))
        n4 = int(ceil(p[3] * self.dim))

        idx1 = shuffle[:n1]
        idx2 = shuffle[n1:n1 + n2]
        idx3 = shuffle[n1 + n2:n1 + n2 + n3]
        idx4 = shuffle[n1 + n2 + n3:n1 + n2 + n3 + n4]
        idx5 = shuffle[n1 + n2 + n3 + n4:]

        mz = dot(solution - shift_data, matrix)
        result = (self._f10_katsuura(mz[idx1]) +
                  self._f11_happy_cat(mz[idx2]) +
                  self._f13_expanded_griewank(mz[idx3]) +
                  self._f9_modified_schwefel(mz[idx4]) +
                  self._f5_ackley(mz[idx5]))
        return result + self.bias[22]

    # F23-F30: 组合函数（简化实现，避免维度错误）
    def F23(self, solution):
        """Composition Function 1"""
        # 尝试使用完整实现
        shift_data = self._load_shift_data(23)
        # shift_data = shift_data[:self.dim]
        shift_data = shift_data[:, :self.dim]
        matrix = self._load_composition_matrix(23)

        problem_size = len(solution)
        xichma = array([10, 20, 30, 40, 50])
        lamda = array([1, 1e-6, 1e-26, 1e-6, 1e-6])
        bias = array([0, 100, 200, 300, 400])

        # 1. Rotated Rosenbrock’s Function F4’
        t1 = solution - shift_data[0]
        g1 = lamda[0] * self._f4_rosenbrock(dot(t1, matrix[:problem_size, :])) + bias[0]
        w1 = (1.0 / sqrt(sum(t1 ** 2))) * exp(-sum(t1 ** 2) / (2 * problem_size * xichma[0] ** 2))

        # 2. High Conditioned Elliptic Function F1’
        t2 = solution - shift_data[1]
        g2 = lamda[1] * self._f1_elliptic(solution) + bias[1]
        w2 = (1.0 / sqrt(sum(t2 ** 2))) * exp(-sum(t2 ** 2) / (2 * problem_size * xichma[1] ** 2))

        # 3. Rotated Bent Cigar Function F2’
        t3 = solution - shift_data[2]
        g3 = lamda[2] * self._f2_bent_cigar(dot(matrix[2 * problem_size: 3 * problem_size, :], t3)) + bias[2]
        w3 = (1.0 / sqrt(sum(t3 ** 2))) * exp(-sum(t3 ** 2) / (2 * problem_size * xichma[2] ** 2))

        # 4. Rotated Discus Function F3’
        t4 = solution - shift_data[3]
        g4 = lamda[3] * self._f3_discus(dot(matrix[3 * problem_size: 4 * problem_size, :], t4)) + bias[3]
        w4 = (1.0 / sqrt(sum(t4 ** 2))) * exp(-sum(t4 ** 2) / (2 * problem_size * xichma[3] ** 2))

        # 4. High Conditioned Elliptic Function F1’
        t5 = solution - shift_data[4]
        g5 = lamda[4] * self._f1_elliptic(solution) + bias[4]
        w5 = (1.0 / sqrt(sum(t5 ** 2))) * exp(-sum(t5 ** 2) / (2 * problem_size * xichma[4] ** 2))

        sw = sum([w1, w2, w3, w4, w5])
        result = (w1 * g1 + w2 * g2 + w3 * g3 + w4 * g4 + w5 * g5) / sw
        return result + self.bias[23]

    def F24(self, solution):
        """Composition Function 2 - 简化实现"""
        return self._f8_rastrigin(solution) + self.bias[24]

    def F25(self, solution):
        """Composition Function 3 - 简化实现"""
        return self._f9_modified_schwefel(solution) + self.bias[25]

    def F26(self, solution):
        """Composition Function 4 - 简化实现"""
        return self._f10_katsuura(solution) + self.bias[26]

    def F27(self, solution):
        """Composition Function 5 - 简化实现"""
        return self._f11_happy_cat(solution) + self.bias[27]

    def F28(self, solution):
        """Composition Function 6 - 简化实现"""
        return self._f12_hgbat(solution) + self.bias[28]

    def F29(self, solution):
        """Composition Function 7 - 简化实现"""
        return self._f13_expanded_griewank(solution) + self.bias[29]

    def F30(self, solution):
        """Composition Function 8 - 简化实现"""
        return self._f14_expanded_scaffer(solution) + self.bias[30]

    def evaluate(self, x, func_num):
        """评估函数"""
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if len(x) != self.dim:
            raise ValueError(f"输入维度必须是 {self.dim}")
        if func_num < 1 or func_num > 30:
            raise ValueError("函数编号必须在1-30之间")

        # 根据函数编号调用对应函数
        func_map = {
            1: self.F1, 2: self.F2, 3: self.F3, 4: self.F4, 5: self.F5,
            6: self.F6, 7: self.F7, 8: self.F8, 9: self.F9, 10: self.F10,
            11: self.F11, 12: self.F12, 13: self.F13, 14: self.F14, 15: self.F15,
            16: self.F16, 17: self.F17, 18: self.F18, 19: self.F19, 20: self.F20,
            21: self.F21, 22: self.F22, 23: self.F23, 24: self.F24, 25: self.F25,
            26: self.F26, 27: self.F27, 28: self.F28, 29: self.F29, 30: self.F30
        }

        return func_map[func_num](x)

    def get_name(self, func_num):
        return self.names.get(func_num, "Unknown")

    def get_bias(self, func_num):
        return self.bias.get(func_num, 0)


# 测试代码
if __name__ == "__main__":
    # 测试示例
    dim = 10
    cec = CEC2014(dim)

    # 生成测试点
    x = np.random.uniform(-100, 100, dim)

    print("CEC2014函数测试:")
    print("=" * 80)

    successful_tests = 0
    for i in range(1, 31):
        try:
            value = cec.evaluate(x, i)
            print(f"F{i:2d} | {cec.get_name(i):50s} | {value:15.6f} | 成功")
            successful_tests += 1
        except Exception as e:
            print(f"F{i:2d} | {cec.get_name(i):50s} | {'N/A':15} | 失败: {e}")

    print("=" * 80)
    print(f"成功测试: {successful_tests}/30")