# MNIST NumPy PNG：从零实现手写数字识别神经网络

本项目使用 **PNG 版 MNIST + Pillow + NumPy**，从零实现一个两层 MLP 神经网络，用于完成 0～9 手写数字识别。项目不依赖 PyTorch、TensorFlow、Keras、scikit-learn，也不依赖 OpenCV。代码核心集中在一个文件中：`mnist_numpy_png.py`。

本项目的主要目的不是追求最高精度，而是帮助 **零基础学生快速理解并掌握基础神经网络的完整流程**：

```text
图片读取 -> 数据归一化 -> 向量化 -> 前向传播 -> Softmax -> 交叉熵损失
       -> 反向传播 -> SGD 参数更新 -> 保存模型 -> 测试评估 -> 单图推理
```

模型结构如下：

```text
输入图片 28×28
    ↓ flatten
784 维向量
    ↓ Linear: x @ W1 + b1
隐藏层 hidden，默认 128
    ↓ ReLU
隐藏层激活 a1
    ↓ Linear: a1 @ W2 + b2
10 维 logits
    ↓ Softmax
10 类概率，分别对应数字 0～9
```

---

## 1. 项目适合谁

<details open>
<summary><strong>展开 / 收起：1. 项目适合谁</strong></summary>


### 1.1 速览表

如果你是第一次阅读本项目，可以先通过下面的速览表选择自己最关心的部分。表格中的链接可以在 GitHub README 页面中点击，快速跳转到对应位置。

| 你想了解什么 | 快速跳转 | 适合阅读的人 |
|---|---|---|
| 这个项目适合谁、为什么要写这份代码 | [项目适合谁](#1-项目适合谁) | 第一次打开项目的读者 |
| 如何从 GitHub 克隆项目、配置环境、准备数据、训练、评估和推理 | [源码复现](#3-源码复现) | 想完整复现实验流程的读者 |
| 代码如何对应神经网络结构 | [代码和神经网络各部分的对应关系](#4-代码和神经网络各部分的对应关系) | 想建立“代码 ↔ 神经网络”对应关系的读者 |
| 前向传播、损失、反向传播、SGD 怎么算 | [神经网络计算流程详解](#5-神经网络计算流程详解) | 想理解核心数学逻辑的读者 |
| 每个自定义函数分别做什么 | [所有自定义函数说明](#6-所有自定义函数说明) | 想逐函数阅读源码的读者 |
| NumPy / Pillow / argparse 等库函数怎么用 | [Python 库函数和方法调用说明](#7-python-库函数和方法调用说明) | 想学习 Python API 的读者 |
| 遇到常见疑问先看哪里 | [常见问题](#8-常见问题) | 运行中遇到疑问的读者 |
| 想直接复制一组命令跑完整流程 | [一条命令完整跑通](#10-一条命令完整跑通) | 已经配置好环境和数据的读者 |

### 1.2 项目定位

本项目适合：

- 刚开始学习深度学习、神经网络、反向传播的学生；
- 想知道 PyTorch / TensorFlow 背后到底做了什么的人；
- 想用最少依赖跑通一个完整训练和推理流程的人；
- 想在 MacBook Air、普通笔记本、CPU 环境中学习神经网络的人。

本项目不适合：

- 追求 MNIST 极限准确率；
- 想直接训练 CNN、ResNet、Transformer 等复杂模型；
- 想使用 GPU 框架加速训练。

---


</details>

## 2. 环境要求

<details open>
<summary><strong>展开 / 收起：2. 环境要求</strong></summary>


推荐环境：

```text
Python >= 3.10
NumPy
Pillow
```

不需要：

```text
PyTorch
TensorFlow
Keras
scikit-learn
OpenCV
```

---


</details>

## 3. 源码复现

<details open>
<summary><strong>展开 / 收起：3. 源码复现</strong></summary>


### 3.1 克隆项目
```bash
git clone https://github.com/qiuxiuhao/mnist_numpy_png.git
cd mnist_numpy_png
```

---

### 3.2 创建 conda 环境

```bash
conda create -n mnist_numpy_png python=3.11 -y
conda activate mnist_numpy_png
```

安装依赖：

```bash
pip install numpy pillow
```

检查是否安装成功：

```bash
python -c "import numpy, PIL; print('numpy ok'); print('pillow ok')"
```

---

### 3.3 准备 MNIST PNG 数据

本项目使用 PNG 版 MNIST，推荐目录结构如下：

```text
data/mnist_png/
├── training/
│   ├── 0/
│   ├── 1/
│   ├── 2/
│   ├── ...
│   └── 9/
└── testing/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── ...
    └── 9/
```

代码同时支持下面这种命名：

```text
data/mnist_png/
├── train/
│   ├── 0/
│   └── ...
└── test/
    ├── 0/
    └── ...
```

也就是说，下面两种目录名都可以：

| 训练集目录 | 测试集目录 |
|---|---|
| `training` | `testing` |
| `train` | `test` |

检查数据数量：

```bash
find data/mnist_png/training -type f | wc -l
find data/mnist_png/testing -type f | wc -l
```

正常情况下，训练集大约 60000 张，测试集大约 10000 张。

---

### 3.4 小规模测试

第一次运行建议先用少量数据，确认代码、环境、路径没有问题：

```bash
python mnist_numpy_png.py train \
  --data-dir data/mnist_png \
  --epochs 2 \
  --batch-size 128 \
  --hidden 128 \
  --lr 0.1 \
  --max-train 2000 \
  --max-test 500
```

这里的参数含义：

| 参数 | 含义 |
|---|---|
| `--data-dir` | MNIST PNG 数据目录 |
| `--epochs` | 训练轮数 |
| `--batch-size` | 每个 mini-batch 的样本数 |
| `--hidden` | 隐藏层神经元数量 |
| `--lr` | 学习率 |
| `--max-train` | 只使用多少训练样本，调试用 |
| `--max-test` | 只使用多少测试样本，调试用 |

---

### 3.5 正式训练

```bash
python mnist_numpy_png.py train \
  --data-dir data/mnist_png \
  --epochs 10 \
  --batch-size 128 \
  --hidden 256 \
  --lr 0.1 \
  --lr-decay 0.98 \
  --weight-decay 0.0001
```

训练完成后会生成：

```text
models/mnist_mlp_png.npz
logs/train_log.csv
```

其中：

- `models/mnist_mlp_png.npz`：保存训练好的模型参数；
- `logs/train_log.csv`：保存每一轮的训练损失、训练准确率、测试损失、测试准确率和耗时。

---

### 3.6 测试集评估

```bash
python mnist_numpy_png.py eval \
  --data-dir data/mnist_png \
  --model models/mnist_mlp_png.npz
```

输出示例：

```text
[eval]
model        : models/mnist_mlp_png.npz
test samples : 10000
test loss    : 0.xxxx
test accuracy: xx.xx%
```

---

### 3.7 从测试集中推理一张图片

```bash
python mnist_numpy_png.py predict-index \
  --data-dir data/mnist_png \
  --model models/mnist_mlp_png.npz \
  --index 0
```

这个命令会：

1. 加载测试集；
2. 取第 `index` 张测试图片；
3. 加载模型；
4. 输出真实标签、预测标签、10 类概率；
5. 用 ASCII 字符在终端中画出图片。

---

### 3.8 推理自己的手写数字图片

如果你有一张自己的数字图片，例如：

```text
my_digit.png
```

执行：

```bash
python mnist_numpy_png.py predict \
  --model models/mnist_mlp_png.npz \
  --image my_digit.png
```

如果你的图片是白底黑字，可以使用：

```bash
python mnist_numpy_png.py predict \
  --model models/mnist_mlp_png.npz \
  --image my_digit.png \
  --invert yes
```

如果你的图片已经是 MNIST 风格，也就是黑底白字，可以使用：

```bash
python mnist_numpy_png.py predict \
  --model models/mnist_mlp_png.npz \
  --image my_digit.png \
  --invert no
```

默认值是：

```bash
--invert auto
```

它会根据图片平均亮度自动判断是否需要反色。

---


</details>

## 4. 代码和神经网络各部分的对应关系

<details open>
<summary><strong>展开 / 收起：4. 代码和神经网络各部分的对应关系</strong></summary>


| 神经网络概念 | 代码位置 / 函数 | 作用 |
|---|---|---|
| 数据集路径搜索 | `find_split_dir()` | 找到训练集和测试集目录 |
| 类别标签读取 | `list_image_files()` | 从父文件夹名 `0`～`9` 读取标签 |
| 图片读取 | `load_one_image()` | 用 Pillow 读取图片并转成灰度图 |
| 输入归一化 | `load_one_image()` | 把像素从 `0～255` 转成 `0～1` |
| 输入向量化 | `load_one_image()` | 把 `28×28` 图片展平成 `784` 维向量 |
| 数据批处理 | `iterate_minibatches()` | 随机打乱训练集并生成 mini-batch |
| 模型参数初始化 | `MLP.__init__()` | 初始化 `W1, b1, W2, b2` |
| 第一层线性变换 | `MLP.forward()` | 计算 `z1 = x @ W1 + b1` |
| 非线性激活 | `relu()` | 计算 `a1 = max(0, z1)` |
| 第二层线性变换 | `MLP.forward()` | 计算 `logits = a1 @ W2 + b2` |
| 概率输出 | `softmax()` | 把 logits 转成 10 类概率 |
| 损失函数 | `cross_entropy()` | 计算分类交叉熵损失 |
| 反向传播 | `MLP.backward()` | 计算每个参数的梯度 |
| 参数更新 | `MLP.step()` | 用 SGD 更新权重和偏置 |
| 准确率计算 | `accuracy()` | 统计预测正确比例 |
| 模型保存 | `MLP.save()` | 保存参数到 `.npz` 文件 |
| 模型加载 | `MLP.load()` | 从 `.npz` 文件恢复参数 |
| 测试评估 | `evaluate_model()` / `eval_command()` | 在测试集上计算 loss 和 accuracy |
| 单图推理 | `predict_command()` | 对外部图片进行预测 |
| 测试集单样本推理 | `predict_index_command()` | 对测试集中指定 index 的图片预测 |
| 命令行入口 | `build_parser()` / `main()` | 解析 `train/eval/predict` 等命令 |

---


</details>

## 5. 神经网络计算流程详解

<details open>
<summary><strong>展开 / 收起：5. 神经网络计算流程详解</strong></summary>


### 5.1 输入层

MNIST 图片大小是 `28×28`。

代码会把图片变成一维向量：

```text
28 × 28 = 784
```

所以单张图片输入是：

```text
x.shape = [784]
```

一个 batch 的输入是：

```text
x.shape = [batch_size, 784]
```

---

### 5.2 第一层 Linear

代码：

```python
z1 = x @ self.W1 + self.b1
```

数学形式：

```text
z1 = XW1 + b1
```

其中：

```text
X.shape  = [batch_size, 784]
W1.shape = [784, hidden]
b1.shape = [hidden]
z1.shape = [batch_size, hidden]
```

这一层的作用是：把原始像素特征映射到隐藏层特征空间。

---

### 5.3 ReLU 激活函数

代码：

```python
a1 = relu(z1)
```

数学形式：

```text
ReLU(x) = max(0, x)
```

作用：

- 如果输入小于 0，输出 0；
- 如果输入大于 0，保持不变；
- 给模型加入非线性能力。

如果没有 ReLU，多层线性层叠加起来仍然只是一个线性模型，表达能力会弱很多。

---

### 5.4 第二层 Linear

代码：

```python
logits = a1 @ self.W2 + self.b2
```

数学形式：

```text
logits = A1W2 + b2
```

其中：

```text
a1.shape     = [batch_size, hidden]
W2.shape     = [hidden, 10]
b2.shape     = [10]
logits.shape = [batch_size, 10]
```

`logits` 是模型对 10 个类别的原始打分，还不是概率。

---

### 5.5 Softmax 概率

代码：

```python
probs = softmax(logits)
```

数学形式：

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

作用：把 10 个类别分数转成概率。

例如：

```text
[0.01, 0.02, 0.90, 0.01, ...]
```

说明模型认为这张图片最可能是数字 `2`。

代码中还做了数值稳定处理：

```python
logits = logits - np.max(logits, axis=1, keepdims=True)
```

这样可以避免 `np.exp(logits)` 因为数值太大而溢出。

---

### 5.6 交叉熵损失

代码：

```python
correct = probs[np.arange(n), y]
loss = -np.mean(np.log(correct + eps))
```

含义：

- `probs[np.arange(n), y]` 取出每个样本真实类别对应的预测概率；
- 如果真实类别概率越高，loss 越小；
- 如果真实类别概率越低，loss 越大。

数学形式：

```text
Loss = - mean(log(P_true_class))
```

---

### 5.7 反向传播

核心代码在：

```python
MLP.backward()
```

反向传播的目标是计算：

```text
dW1, db1, dW2, db2
```

也就是损失函数对每个参数的梯度。

关键步骤：

```python
dlogits = probs.copy()
dlogits[np.arange(n), y] -= 1.0
dlogits /= n
```

这是 Softmax + Cross Entropy 合并后的经典梯度：

```text
dlogits = (probs - one_hot_labels) / batch_size
```

然后继续通过矩阵乘法把梯度传回第二层、ReLU、第一层。

---

### 5.8 SGD 参数更新

代码：

```python
self.W1 -= lr * grads["W1"]
self.b1 -= lr * grads["b1"]
self.W2 -= lr * grads["W2"]
self.b2 -= lr * grads["b2"]
```

数学形式：

```text
参数 = 参数 - 学习率 × 梯度
```

这就是最基础的随机梯度下降 SGD。

---


</details>

## 6. 所有自定义函数说明

<details open>
<summary><strong>展开 / 收起：6. 所有自定义函数说明</strong></summary>


### 6.1 数据读取相关函数

#### `find_split_dir(data_dir, candidates)`

作用：查找训练集或测试集目录。

输入：

| 参数 | 类型 | 含义 |
|---|---|---|
| `data_dir` | `Path` | 数据集根目录 |
| `candidates` | `List[str]` | 候选目录名，例如 `['train', 'training']` |

输出：

```text
Path
```

计算逻辑：

1. 先在 `data_dir` 下直接找候选目录；
2. 判断这个目录是否存在、是否是文件夹、是否包含数字类别子目录；
3. 如果直接找不到，就递归搜索子目录；
4. 如果还是找不到，抛出 `FileNotFoundError`。

---

#### `has_digit_subdirs(path)`

作用：判断某个目录下是否有数字类别文件夹。

输入：

| 参数 | 类型 | 含义 |
|---|---|---|
| `path` | `Path` | 待检查目录 |

输出：

```text
bool
```

计算逻辑：

1. 检查 `path/0`、`path/1`、...、`path/9`；
2. 每存在一个数字文件夹，计数加 1；
3. 如果至少存在 5 个数字类别文件夹，返回 `True`；
4. 否则返回 `False`。

---

#### `list_image_files(split_dir)`

作用：列出某个数据划分中的所有图片路径和标签。

输入：

| 参数 | 类型 | 含义 |
|---|---|---|
| `split_dir` | `Path` | 训练集或测试集目录 |

输出：

```text
List[Tuple[Path, int]]
```

每个元素格式：

```text
(image_path, label)
```

计算逻辑：

1. 遍历数字 `0～9`；
2. 进入对应类别目录，例如 `training/5/`；
3. 递归查找 `.png`、`.jpg`、`.jpeg`、`.bmp` 文件；
4. 把图片路径和标签加入列表；
5. 如果没有任何图片，抛出 `RuntimeError`。

---

#### `load_one_image(path, image_size=28)`

作用：读取单张图片，并转成神经网络输入向量。

输入：

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `path` | `Path` | 无 | 图片路径 |
| `image_size` | `int` | `28` | 目标图片大小 |

输出：

```text
np.ndarray, shape = [784]
```

计算逻辑：

1. 用 `Image.open(path)` 打开图片；
2. 用 `.convert('L')` 转成灰度图；
3. 如果不是 `28×28`，则 resize 到 `28×28`；
4. 用 `np.asarray()` 转成 NumPy 数组；
5. 除以 `255.0`，归一化到 `[0, 1]`；
6. 用 `.reshape(-1)` 展平成 784 维向量。

---

#### `load_png_split(split_dir, image_size=28)`

作用：加载一个完整数据划分，例如训练集或测试集。

输入：

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `split_dir` | `Path` | 无 | 训练集或测试集目录 |
| `image_size` | `int` | `28` | 图片大小 |

输出：

```text
x: np.ndarray, shape = [N, 784]
y: np.ndarray, shape = [N]
```

计算逻辑：

1. 调用 `list_image_files()` 获取所有图片路径和标签；
2. 对每张图片调用 `load_one_image()`；
3. 把所有图片向量用 `np.stack()` 合并成矩阵；
4. 把标签列表用 `np.asarray()` 转成数组。

---

#### `maybe_subset(x, y, max_samples, seed)`

作用：随机抽取一部分数据，方便快速调试。

输入：

| 参数 | 类型 | 含义 |
|---|---|---|
| `x` | `np.ndarray` | 图片数据 |
| `y` | `np.ndarray` | 标签数据 |
| `max_samples` | `Optional[int]` | 最大样本数 |
| `seed` | `int` | 随机种子 |

输出：

```text
x_subset, y_subset
```

计算逻辑：

1. 如果 `max_samples` 为空或不合法，直接返回原数据；
2. 用随机数生成器生成随机排列；
3. 取前 `max_samples` 个索引；
4. 返回对应子集。

---

#### `load_dataset(...)`

作用：加载完整训练集和测试集，并支持缓存。

输入主要包括：

| 参数 | 含义 |
|---|---|
| `data_dir` | 数据根目录 |
| `image_size` | 图片大小 |
| `cache` | 是否使用 `.npz` 缓存 |
| `max_train` | 最大训练样本数，调试用 |
| `max_test` | 最大测试样本数，调试用 |
| `seed` | 随机种子 |

输出：

```text
x_train, y_train, x_test, y_test
```

计算逻辑：

1. 如果缓存文件存在，就用 `np.load()` 直接读取；
2. 如果没有缓存，就查找训练目录和测试目录；
3. 分别调用 `load_png_split()` 读取训练集和测试集；
4. 如果开启缓存，用 `np.savez_compressed()` 保存；
5. 根据 `max_train` 和 `max_test` 可选抽样；
6. 返回四个数组。

---

### 6.2 神经网络基础函数

#### `relu(x)`

作用：ReLU 激活函数。

输入：

```text
x: np.ndarray
```

输出：

```text
np.ndarray
```

计算方式：

```text
max(x, 0)
```

代码实现：

```python
np.maximum(x, 0.0)
```

---

#### `relu_backward(grad, z)`

作用：ReLU 的反向传播。

输入：

| 参数 | 含义 |
|---|---|
| `grad` | 后一层传回来的梯度 |
| `z` | ReLU 激活前的输入 |

输出：

```text
np.ndarray
```

计算逻辑：

```text
如果 z > 0，梯度保留
如果 z <= 0，梯度变成 0
```

代码实现：

```python
grad * (z > 0)
```

---

#### `softmax(logits)`

作用：把类别分数转换为概率。

输入：

```text
logits: np.ndarray, shape = [batch_size, 10]
```

输出：

```text
probs: np.ndarray, shape = [batch_size, 10]
```

计算逻辑：

1. 每一行减去最大值，防止指数溢出；
2. 对每个类别分数取指数；
3. 除以所有类别指数之和；
4. 输出每个类别的概率。

---

#### `cross_entropy(probs, y)`

作用：计算分类交叉熵损失。

输入：

| 参数 | 含义 |
|---|---|
| `probs` | 模型输出概率，shape = `[batch_size, 10]` |
| `y` | 真实标签，shape = `[batch_size]` |

输出：

```text
float
```

计算逻辑：

1. 取出每个样本真实类别对应的预测概率；
2. 对概率取对数；
3. 加负号；
4. 对 batch 求平均。

---

#### `accuracy(probs, y)`

作用：计算预测准确率。

输入：

| 参数 | 含义 |
|---|---|
| `probs` | 模型输出概率 |
| `y` | 真实标签 |

输出：

```text
float
```

计算逻辑：

1. 用 `np.argmax(probs, axis=1)` 取概率最大的类别；
2. 和真实标签 `y` 比较；
3. 用 `np.mean()` 统计预测正确比例。

---

### 6.3 `MLP` 类

#### `MLP.__init__(input_dim=784, hidden_dim=128, output_dim=10, seed=42)`

作用：初始化神经网络参数。

参数：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `input_dim` | `784` | 输入维度 |
| `hidden_dim` | `128` | 隐藏层维度 |
| `output_dim` | `10` | 输出类别数 |
| `seed` | `42` | 随机种子 |

内部参数：

| 参数 | shape | 含义 |
|---|---|---|
| `W1` | `[784, hidden]` | 第一层权重 |
| `b1` | `[hidden]` | 第一层偏置 |
| `W2` | `[hidden, 10]` | 第二层权重 |
| `b2` | `[10]` | 第二层偏置 |

初始化方式：

```text
W = randn(...) * sqrt(2 / fan_in)
```

这是 He initialization，适合 ReLU 网络。

---

#### `MLP.forward(x)`

作用：前向传播。

输入：

```text
x: np.ndarray, shape = [batch_size, 784]
```

输出：

```text
probs: np.ndarray, shape = [batch_size, 10]
cache: Dict[str, np.ndarray]
```

计算过程：

```python
z1 = x @ W1 + b1
a1 = relu(z1)
logits = a1 @ W2 + b2
probs = softmax(logits)
```

`cache` 保存中间结果，供反向传播使用。

---

#### `MLP.backward(cache, y, weight_decay=0.0)`

作用：反向传播，计算梯度。

输入：

| 参数 | 含义 |
|---|---|
| `cache` | 前向传播保存的中间变量 |
| `y` | 真实标签 |
| `weight_decay` | L2 正则化系数 |

输出：

```text
Dict[str, np.ndarray]
```

包括：

```text
dW1, db1, dW2, db2
```

计算过程：

```text
dlogits = (probs - one_hot(y)) / batch_size
dW2 = a1.T @ dlogits
db2 = sum(dlogits)
da1 = dlogits @ W2.T
dz1 = relu_backward(da1, z1)
dW1 = x.T @ dz1
db1 = sum(dz1)
```

如果 `weight_decay > 0`，则：

```text
dW1 += weight_decay * W1
dW2 += weight_decay * W2
```

---

#### `MLP.step(grads, lr)`

作用：使用 SGD 更新模型参数。

输入：

| 参数 | 含义 |
|---|---|
| `grads` | 参数梯度 |
| `lr` | 学习率 |

计算方式：

```text
W = W - lr * dW
b = b - lr * db
```

---

#### `MLP.predict_proba(x, batch_size=512)`

作用：分批预测概率。

输入：

| 参数 | 含义 |
|---|---|
| `x` | 输入数据 |
| `batch_size` | 每批预测数量 |

输出：

```text
np.ndarray, shape = [N, 10]
```

计算逻辑：

1. 把大数据分成多个 batch；
2. 每个 batch 调用 `forward()`；
3. 把所有概率用 `np.concatenate()` 拼接起来。

---

#### `MLP.predict(x, batch_size=512)`

作用：输出最终预测类别。

输出：

```text
np.ndarray, shape = [N]
```

计算逻辑：

1. 调用 `predict_proba()` 得到概率；
2. 对每一行取最大概率类别。

---

#### `MLP.save(path)`

作用：保存模型参数。

保存内容：

```text
W1, b1, W2, b2, input_dim, hidden_dim, output_dim
```

保存格式：

```text
.npz
```

---

#### `MLP.load(path)`

作用：加载模型参数。

输出：

```text
MLP 实例
```

计算逻辑：

1. 检查模型文件是否存在；
2. 用 `np.load()` 读取参数；
3. 根据保存的维度创建一个 MLP；
4. 把保存的参数赋值给模型；
5. 返回模型。

---

### 6.4 训练与评估函数

#### `iterate_minibatches(x, y, batch_size, rng)`

作用：生成随机 mini-batch。

计算逻辑：

1. 用 `rng.permutation(n)` 打乱样本索引；
2. 按 `batch_size` 切分；
3. 每次返回一批 `xb, yb`。

---

#### `evaluate_model(model, x, y, batch_size=512)`

作用：评估模型。

输出：

```text
loss, acc
```

计算逻辑：

1. 调用 `model.predict_proba()`；
2. 调用 `cross_entropy()`；
3. 调用 `accuracy()`；
4. 返回损失和准确率。

---

#### `write_log_header(log_path)`

作用：创建训练日志文件，并写入表头。

输出 CSV 表头：

```text
epoch, lr, train_loss, train_acc, test_loss, test_acc, time_sec
```

---

#### `append_log(...)`

作用：每个 epoch 结束后，向 CSV 日志追加一行训练结果。

---

#### `train_command(args)`

作用：执行完整训练流程。

流程：

1. 读取数据；
2. 创建 MLP 模型；
3. 初始化随机数生成器；
4. 写日志表头；
5. 循环训练多个 epoch；
6. 每个 batch 执行前向传播、损失计算、反向传播、SGD 更新；
7. 每个 epoch 后在测试集评估；
8. 如果测试准确率变高，就保存最佳模型；
9. 学习率乘以 `lr_decay`。

---

#### `eval_command(args)`

作用：加载模型并在测试集上评估。

流程：

1. 加载测试集；
2. 加载 `.npz` 模型；
3. 计算测试损失和准确率；
4. 打印结果。

---

### 6.5 推理相关函数

#### `preprocess_predict_image(image_path, image_size=28, invert='auto')`

作用：预处理一张外部图片，用于推理。

输入：

| 参数 | 含义 |
|---|---|
| `image_path` | 图片路径 |
| `image_size` | 目标大小 |
| `invert` | 是否反色，可选 `auto`、`yes`、`no` |

输出：

```text
x: shape = [1, 784]
arr: shape = [28, 28]
```

计算逻辑：

1. 检查图片是否存在；
2. 打开图片并转灰度；
3. resize 到 `28×28`；
4. 归一化到 `[0, 1]`；
5. 根据 `invert` 参数决定是否反色；
6. 展平成 `[1, 784]`。

---

#### `image_to_ascii(arr)`

作用：把图片转换成终端可显示的 ASCII 字符图。

计算逻辑：

1. 将像素值限制在 `[0, 1]`；
2. 把亮度映射到字符集：

```text
" .:-=+*#%@"
```

3. 每一行拼接成字符串；
4. 返回完整 ASCII 图片。

---

#### `print_probabilities(probs)`

作用：打印 0～9 每一类的预测概率，并用 `#` 画出简易概率条。

---

#### `predict_command(args)`

作用：对外部图片执行推理。

流程：

1. 加载模型；
2. 调用 `preprocess_predict_image()` 读取并处理图片；
3. 调用 `model.predict_proba()` 得到概率；
4. 用 `np.argmax()` 得到预测类别；
5. 打印预测结果、概率、ASCII 图片。

---

#### `predict_index_command(args)`

作用：对测试集中的某个样本执行推理。

流程：

1. 加载测试集；
2. 检查 index 是否合法；
3. 加载模型；
4. 取出指定测试样本；
5. 预测类别；
6. 打印真实标签、预测标签、概率和 ASCII 图片。

---

### 6.6 命令行函数

#### `build_parser()`

作用：定义命令行参数。

支持 4 个子命令：

```text
train
    训练模型

eval
    测试集评估

predict
    推理一张外部图片

predict-index
    推理测试集中的一张图片
```

---

#### `main()`

作用：程序入口。

流程：

1. 创建命令行解析器；
2. 解析用户输入参数；
3. 根据子命令调用对应函数；
4. 捕获键盘中断和异常。

---


</details>

## 7. Python 库函数和方法调用说明

<details open>
<summary><strong>展开 / 收起：7. Python 库函数和方法调用说明</strong></summary>


下面按模块列出代码中出现过的重要 Python / NumPy / Pillow 调用。

---

### 7.1 `argparse`：命令行参数解析

#### `argparse.ArgumentParser(description=...)`

作用：创建命令行解析器。

输入：

| 参数 | 含义 |
|---|---|
| `description` | 命令行帮助说明 |

输出：

```text
ArgumentParser 对象
```

---

#### `parser.add_subparsers(dest='command', required=True)`

作用：创建子命令，例如 `train`、`eval`、`predict`。

参数：

| 参数 | 含义 |
|---|---|
| `dest` | 子命令名称保存到哪个属性中 |
| `required` | 是否必须指定子命令 |

输出：

```text
_SubParsersAction 对象
```

---

#### `subparsers.add_parser(name, help=...)`

作用：添加一个子命令。

例如：

```python
subparsers.add_parser("train", help="Train model.")
```

输出：

```text
ArgumentParser 对象
```

---

#### `parser.add_argument(...)`

作用：添加命令行参数。

常见参数：

| 参数 | 含义 |
|---|---|
| 参数名，例如 `--epochs` | 命令行中使用的参数名称 |
| `type` | 参数类型，例如 `int`、`float`、`str` |
| `default` | 默认值 |
| `required` | 是否必填 |
| `choices` | 限制可选值 |
| `action='store_true'` | 出现该参数时值为 `True` |
| `help` | 参数说明 |

输出：

```text
Action 对象
```

---

#### `parser.parse_args()`

作用：解析终端输入的参数。

输出：

```text
Namespace 对象
```

例如：

```text
args.epochs
args.batch_size
args.lr
```

---

#### `parser.set_defaults(func=train_command)`

作用：给某个子命令绑定默认执行函数。

例如输入：

```bash
python mnist_numpy_png.py train
```

最终会执行：

```python
args.func(args)
```

也就是 `train_command(args)`。

---

### 7.2 `pathlib.Path`：路径处理

#### `Path(path_string)`

作用：把字符串路径转换成 Path 对象。

输入：

```text
str
```

输出：

```text
Path
```

---

#### `path.resolve()`

作用：得到绝对路径。

输入：无。

输出：

```text
Path
```

---

#### `path.exists()`

作用：判断路径是否存在。

输出：

```text
bool
```

---

#### `path.is_dir()`

作用：判断路径是否是文件夹。

输出：

```text
bool
```

---

#### `path.is_file()`

作用：判断路径是否是文件。

输出：

```text
bool
```

---

#### `path.rglob(pattern)`

作用：递归搜索文件或文件夹。

输入：

| 参数 | 含义 |
|---|---|
| `pattern` | 匹配规则，例如 `'*'` |

输出：

```text
生成器，可遍历 Path 对象
```

---

#### `path.parent`

作用：返回上一级目录。

输出：

```text
Path
```

---

#### `path.mkdir(parents=True, exist_ok=True)`

作用：创建目录。

参数：

| 参数 | 含义 |
|---|---|
| `parents=True` | 父目录不存在时一起创建 |
| `exist_ok=True` | 目录已存在时不报错 |

输出：

```text
None
```

---

#### `path.suffix`

作用：获取文件后缀。

例如：

```text
Path('a.png').suffix -> '.png'
```

输出：

```text
str
```

---

#### `path.name`

作用：获取路径最后一部分名称。

例如：

```text
Path('/a/b/training').name -> 'training'
```

输出：

```text
str
```

---

### 7.3 `Pillow`：图片读取与处理

#### `Image.open(path)`

作用：打开图片文件。

输入：

```text
图片路径
```

输出：

```text
PIL.Image.Image 对象
```

---

#### `img.convert('L')`

作用：把图片转换成灰度图。

参数：

| 参数 | 含义 |
|---|---|
| `'L'` | 8-bit 灰度图模式 |

输出：

```text
PIL.Image.Image 对象
```

---

#### `img.resize((image_size, image_size), Image.Resampling.LANCZOS)`

作用：调整图片大小。

参数：

| 参数 | 含义 |
|---|---|
| `(image_size, image_size)` | 目标尺寸 |
| `Image.Resampling.LANCZOS` | 高质量重采样方式 |

输出：

```text
PIL.Image.Image 对象
```

---

#### `img.size`

作用：获取图片尺寸。

输出：

```text
(width, height)
```

---

### 7.4 `NumPy`：数组与矩阵计算

#### `np.asarray(obj, dtype=...)`

作用：把对象转换成 NumPy 数组。

常见输入：

```text
PIL 图片对象
Python list
```

参数：

| 参数 | 含义 |
|---|---|
| `obj` | 待转换对象 |
| `dtype` | 输出数组类型，例如 `np.float32` |

输出：

```text
np.ndarray
```

---

#### `np.stack(list, axis=0)`

作用：把多个同形状数组堆叠成一个新数组。

输入：

```text
[x1, x2, x3, ...]
```

输出：

```text
shape = [N, ...]
```

在本项目中用于把很多 `[784]` 图片向量合并成 `[N, 784]`。

---

#### `np.asarray(y_list, dtype=np.int64)`

作用：把标签列表转成整数数组。

输出：

```text
shape = [N]
```

---

#### `np.random.default_rng(seed)`

作用：创建随机数生成器。

输入：

| 参数 | 含义 |
|---|---|
| `seed` | 随机种子 |

输出：

```text
np.random.Generator
```

用于可复现实验。

---

#### `rng.standard_normal(shape)`

作用：生成标准正态分布随机数。

输入：

```text
shape，例如 (784, 128)
```

输出：

```text
np.ndarray
```

在本项目中用于初始化权重。

---

#### `rng.permutation(n)`

作用：生成 `0～n-1` 的随机排列。

输入：

```text
n: int
```

输出：

```text
np.ndarray, shape = [n]
```

用于打乱训练样本顺序。

---

#### `np.sqrt(x)`

作用：计算平方根。

在本项目中用于 He initialization：

```python
np.sqrt(2.0 / input_dim)
```

输出：

```text
float 或 np.ndarray
```

---

#### `np.zeros(shape, dtype=...)`

作用：创建全 0 数组。

输入：

| 参数 | 含义 |
|---|---|
| `shape` | 数组形状 |
| `dtype` | 数据类型 |

输出：

```text
np.ndarray
```

在本项目中用于初始化偏置 `b1`、`b2`。

---

#### `array.astype(dtype)`

作用：转换数组数据类型。

输入：

```text
dtype，例如 np.float32
```

输出：

```text
np.ndarray
```

---

#### `array.reshape(...)`

作用：改变数组形状。

例子：

```python
arr.reshape(-1)
```

把 `28×28` 转成 `784`。

```python
arr.reshape(1, -1)
```

把单张图片转成 `[1, 784]`。

输出：

```text
np.ndarray
```

---

#### `x @ W`

作用：矩阵乘法。

虽然不是函数调用，但它是 NumPy 数组的重要运算符。

例子：

```python
z1 = x @ W1
```

形状：

```text
[batch_size, 784] @ [784, hidden] = [batch_size, hidden]
```

---

#### `np.maximum(x, 0.0)`

作用：逐元素取最大值。

在本项目中实现 ReLU：

```python
relu(x) = max(x, 0)
```

输出：

```text
np.ndarray
```

---

#### `np.max(logits, axis=1, keepdims=True)`

作用：沿指定维度取最大值。

参数：

| 参数 | 含义 |
|---|---|
| `axis=1` | 按每一行取最大值 |
| `keepdims=True` | 保持二维形状，方便广播 |

输出：

```text
shape = [batch_size, 1]
```

---

#### `np.exp(logits)`

作用：逐元素计算指数。

输出：

```text
np.ndarray
```

用于 Softmax。

---

#### `np.sum(exp, axis=1, keepdims=True)`

作用：沿指定维度求和。

在 Softmax 中表示每个样本所有类别指数分数之和。

输出：

```text
shape = [batch_size, 1]
```

---

#### `np.arange(n)`

作用：生成 `[0, 1, 2, ..., n-1]`。

输出：

```text
np.ndarray
```

在交叉熵中用于取出每个样本真实类别概率：

```python
probs[np.arange(n), y]
```

---

#### `np.log(x)`

作用：逐元素取自然对数。

在交叉熵中使用。

输出：

```text
np.ndarray
```

---

#### `np.mean(x)`

作用：求平均值。

在本项目中用于：

- 计算平均 loss；
- 计算平均 accuracy；
- 统计图片平均亮度。

输出：

```text
float 或 np.ndarray
```

---

#### `np.argmax(probs, axis=1)`

作用：返回最大值的索引。

在本项目中用于取预测类别。

输入：

```text
probs.shape = [batch_size, 10]
```

输出：

```text
shape = [batch_size]
```

---

#### `np.concatenate(list, axis=0)`

作用：沿指定维度拼接数组。

在本项目中把多个 batch 的预测概率拼接成完整预测结果。

输出：

```text
np.ndarray
```

---

#### `np.savez(path, ...)`

作用：保存多个数组到一个 `.npz` 文件。

输入：

```python
np.savez(path, W1=..., b1=..., W2=..., b2=...)
```

输出：

```text
在磁盘写入文件
```

---

#### `np.savez_compressed(path, ...)`

作用：压缩保存多个数组到 `.npz` 文件。

本项目用于保存数据缓存。

输出：

```text
压缩后的 .npz 文件
```

---

#### `np.load(path)`

作用：读取 `.npy` 或 `.npz` 文件。

输出：

```text
NpzFile 或 np.ndarray
```

在本项目中用于：

- 读取数据缓存；
- 读取模型参数。

---

#### `np.clip(arr, 0.0, 1.0)`

作用：把数组值限制在指定范围内。

参数：

| 参数 | 含义 |
|---|---|
| `arr` | 输入数组 |
| `0.0` | 最小值 |
| `1.0` | 最大值 |

输出：

```text
np.ndarray
```

在本项目中用于 ASCII 图片显示前的亮度限制。

---

### 7.5 `csv`：训练日志保存

#### `open(log_path, 'w', newline='', encoding='utf-8')`

作用：打开文件用于写入。

参数：

| 参数 | 含义 |
|---|---|
| `'w'` | 写模式，会覆盖旧文件 |
| `'a'` | 追加模式，在旧文件后面继续写 |
| `newline=''` | 避免 CSV 多余空行 |
| `encoding='utf-8'` | 使用 UTF-8 编码 |

输出：

```text
文件对象
```

---

#### `csv.writer(f)`

作用：创建 CSV 写入器。

输入：

```text
文件对象
```

输出：

```text
_csv.writer 对象
```

---

#### `writer.writerow(list)`

作用：向 CSV 写入一行。

输入：

```text
list
```

输出：

```text
写入文件，返回写入字符数或相关结果
```

---

### 7.6 `time`：统计训练耗时

#### `time.time()`

作用：返回当前时间戳。

输出：

```text
float，单位是秒
```

用法：

```python
epoch_start = time.time()
...
elapsed = time.time() - epoch_start
```

表示一个 epoch 的耗时。

---

### 7.7 `sys`：程序退出

#### `sys.exit(code)`

作用：退出程序。

参数：

| code | 含义 |
|---|---|
| `0` | 正常退出 |
| `1` | 一般错误 |
| `130` | 键盘中断，通常对应 Ctrl+C |

在本项目中用于异常退出。

---

### 7.8 Python 内置函数和语法

#### `print(...)`

作用：在终端输出信息。

输出：

```text
None
```

---

#### `len(obj)`

作用：返回对象长度。

例如：

```python
len(y_train)
```

返回训练样本数量。

---

#### `range(start, stop, step)`

作用：生成整数序列。

在本项目中用于：

- 遍历类别 `0～9`；
- 遍历 epoch；
- 按 batch size 切分数据。

---

#### `enumerate(iterable, start=1)`

作用：遍历时同时得到编号和元素。

输出：

```text
(index, item)
```

---

#### `sorted(iterable)`

作用：排序。

本项目中用于让图片路径按固定顺序加载。

---

#### `int(x)` / `float(x)`

作用：类型转换。

例如：

```python
pred = int(np.argmax(probs))
```

把 NumPy 整数转换成 Python 整数，方便打印。

---

#### `isinstance` 没有使用

本项目没有使用 `isinstance()`。类型主要通过函数参数注解说明。

---

#### `try / except`

作用：异常捕获。

本项目在 `main()` 中捕获：

```python
KeyboardInterrupt
Exception
```

如果用户按下 Ctrl+C，会输出：

```text
[interrupted]
```

如果发生其他错误，会输出：

```text
[error] 错误信息
```

---

#### `with open(...) as f`

作用：上下文管理器。

优点：

- 自动关闭文件；
- 即使发生异常，也能安全释放资源。

---

#### 列表 `append()`

作用：向列表末尾添加元素。

本项目中用于：

```python
x_list.append(...)
y_list.append(...)
batch_losses.append(...)
batch_accs.append(...)
```

输出：

```text
None
```

---

#### 字符串 `.lower()`

作用：把字符串转成小写。

本项目中用于比较目录名、文件后缀。

输出：

```text
str
```

---

#### 字符串 `.join(list)`

作用：把字符串列表拼接成一个字符串。

本项目中用于生成 ASCII 图片：

```python
"".join(chars[i] for i in row)
```

输出：

```text
str
```

---

#### f-string

作用：格式化字符串。

例如：

```python
f"epoch {epoch:03d}/{args.epochs}"
```

其中：

```text
{epoch:03d}
```

表示整数不足 3 位时前面补 0。

---


</details>

## 8. 常见问题

<details open>
<summary><strong>展开 / 收起：8. 常见问题</strong></summary>


### 8.1 为什么不用 PyTorch？

因为本项目的目标是教学。直接用 PyTorch 很容易把关键计算隐藏起来，例如：

- 前向传播；
- Softmax；
- Cross Entropy；
- 反向传播；
- 参数更新。

本项目用 NumPy 手写这些过程，学生更容易理解神经网络底层逻辑。

---

### 8.2 为什么图片要除以 255？

原始灰度图像素范围是：

```text
0～255
```

神经网络更适合处理较小范围的输入，所以归一化为：

```text
0～1
```

计算方式：

```python
arr = np.asarray(img, dtype=np.float32) / 255.0
```

---

### 8.3 为什么要把 `28×28` 展平成 `784`？

因为本项目使用的是 MLP，不是 CNN。MLP 的输入是一维向量，所以需要把二维图片展平：

```text
28 × 28 = 784
```

---

### 8.4 为什么要用 mini-batch？

如果一次只训练一张图片，训练会很不稳定；如果一次使用全部 60000 张图片，计算又比较慢。

mini-batch 是折中方案，例如：

```text
batch_size = 128
```

每次用 128 张图片计算平均梯度并更新参数。

---

### 8.5 为什么要保存 best model？

训练过程中，测试准确率不一定每轮都上升。代码会记录当前最好的测试准确率：

```python
if test_acc > best_test_acc:
    model.save(output_path)
```

这样最终保存的是测试集表现最好的模型。

---

### 8.6 为什么要有缓存文件？

第一次从 PNG 文件读取 60000 + 10000 张图片会比较慢。代码会把处理好的数组保存为：

```text
data/mnist_png/mnist_png_cache_28.npz
```

之后再次训练时可以直接读取缓存，提高启动速度。

---


</details>

## 9. 建议的 `.gitignore`

<details open>
<summary><strong>展开 / 收起：9. 建议的 `.gitignore`</strong></summary>


由于数据集、模型文件、日志文件通常不建议直接提交到 GitHub，可以添加 `.gitignore`：

```gitignore
# Python
__pycache__/
*.pyc

# Conda / venv
.venv/

# Dataset
data/

# Model and logs
models/
logs/
*.npz
*.csv

# macOS
.DS_Store
```

如果你希望 GitHub 仓库只保存代码和 README，不保存数据集，就使用上面的 `.gitignore`。

---


</details>

## 10. 一条命令完整跑通

<details open>
<summary><strong>展开 / 收起：10. 一条命令完整跑通</strong></summary>


如果环境和数据都准备好了，可以直接执行：

```bash
conda activate mnist_numpy_png

python mnist_numpy_png.py train \
  --data-dir data/mnist_png \
  --epochs 10 \
  --batch-size 128 \
  --hidden 256 \
  --lr 0.1 \
  --lr-decay 0.98 \
  --weight-decay 0.0001

python mnist_numpy_png.py eval \
  --data-dir data/mnist_png \
  --model models/mnist_mlp_png.npz

python mnist_numpy_png.py predict-index \
  --data-dir data/mnist_png \
  --model models/mnist_mlp_png.npz \
  --index 0
```

---


</details>

## 11. 学习路线建议

<details open>
<summary><strong>展开 / 收起：11. 学习路线建议</strong></summary>


读代码时建议按下面顺序：

```text
1. load_one_image()
2. load_png_split()
3. load_dataset()
4. MLP.__init__()
5. MLP.forward()
6. softmax()
7. cross_entropy()
8. MLP.backward()
9. MLP.step()
10. train_command()
11. eval_command()
12. predict_command()
```

这样可以从数据进入模型开始，一步步理解训练和推理全过程。

---


</details>

## 12. 项目总结

<details open>
<summary><strong>展开 / 收起：12. 项目总结</strong></summary>


这个项目用最基础的 Python、NumPy 和 Pillow 实现了一个完整的手写数字识别系统。它包含：

- PNG 图片读取；
- 数据归一化；
- MLP 神经网络；
- ReLU 激活；
- Softmax 输出；
- Cross Entropy 损失；
- 手写反向传播；
- SGD 参数更新；
- 训练日志保存；
- 模型保存和加载；
- 测试集评估；
- 外部图片推理；
- 终端 ASCII 可视化。

通过这个项目，零基础学生可以完整理解：一个神经网络从读取数据，到训练参数，再到最终预测数字的全过程。

</details>
