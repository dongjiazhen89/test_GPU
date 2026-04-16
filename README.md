# test_GPU
test GPU

## PyTorch GPU 压力测试

新增 `gpu_stress_test.py`，可选择单卡或多卡测试，尽量占满显存并持续进行矩阵乘计算以压满算力。

### 依赖

- Python 3
- PyTorch（CUDA 版本）

### 用法

单卡测试（默认 GPU 0，60 秒）：

```bash
python gpu_stress_test.py --mode single --gpu-id 0 --duration 60
```

多卡测试（全部 GPU，60 秒）：

```bash
python gpu_stress_test.py --mode multi --gpus all --duration 60
```

指定多卡：

```bash
python gpu_stress_test.py --mode multi --gpus 0,1 --duration 120 --mem-ratio 0.9 --matrix-size 8192
```
