运行设备: cuda | 可用 GPU 数量: 1
100%|██████████| 170M/170M [00:05<00:00, 32.9MB/s]
============================================================
STEP 1: 运行基准测试 (Standard Single GPU)
============================================================

>>> 启动训练 [模式: standard] (Epochs: 25)
    Epoch [1/25] | LR: 1.0e-01 | Loss: 1.7564 | Acc: 55.87% | Time: 20.4s
    Epoch [2/25] | LR: 1.0e-01 | Loss: 1.1832 | Acc: 63.85% | Time: 17.5s
    Epoch [3/25] | LR: 1.0e-01 | Loss: 0.9802 | Acc: 69.71% | Time: 18.3s
    Epoch [4/25] | LR: 1.0e-01 | Loss: 0.8906 | Acc: 69.43% | Time: 17.4s
    Epoch [5/25] | LR: 1.0e-01 | Loss: 0.8285 | Acc: 67.44% | Time: 18.0s
    Epoch [6/25] | LR: 1.0e-01 | Loss: 0.7973 | Acc: 66.15% | Time: 17.9s
    Epoch [7/25] | LR: 1.0e-01 | Loss: 0.7601 | Acc: 72.04% | Time: 17.4s
    Epoch [8/25] | LR: 1.0e-01 | Loss: 0.7458 | Acc: 74.43% | Time: 18.2s
    Epoch [9/25] | LR: 1.0e-01 | Loss: 0.7275 | Acc: 77.91% | Time: 17.7s
    Epoch [10/25] | LR: 1.0e-01 | Loss: 0.7129 | Acc: 75.80% | Time: 17.4s
    Epoch [11/25] | LR: 1.0e-02 | Loss: 0.5780 | Acc: 84.14% | Time: 18.1s
    Epoch [12/25] | LR: 1.0e-02 | Loss: 0.5342 | Acc: 85.24% | Time: 17.6s
    Epoch [13/25] | LR: 1.0e-02 | Loss: 0.5155 | Acc: 85.24% | Time: 18.1s
    Epoch [14/25] | LR: 1.0e-02 | Loss: 0.5008 | Acc: 85.29% | Time: 17.4s
    Epoch [15/25] | LR: 1.0e-02 | Loss: 0.4894 | Acc: 85.41% | Time: 17.7s
    Epoch [16/25] | LR: 1.0e-02 | Loss: 0.4786 | Acc: 85.84% | Time: 18.3s
    Epoch [17/25] | LR: 1.0e-02 | Loss: 0.4710 | Acc: 85.89% | Time: 17.7s
    Epoch [18/25] | LR: 1.0e-02 | Loss: 0.4614 | Acc: 86.14% | Time: 17.9s
    Epoch [19/25] | LR: 1.0e-02 | Loss: 0.4511 | Acc: 85.73% | Time: 18.3s
    Epoch [20/25] | LR: 1.0e-02 | Loss: 0.4450 | Acc: 86.12% | Time: 17.5s
    Epoch [21/25] | LR: 1.0e-03 | Loss: 0.4194 | Acc: 87.04% | Time: 18.4s
    Epoch [22/25] | LR: 1.0e-03 | Loss: 0.4093 | Acc: 87.13% | Time: 17.4s
    Epoch [23/25] | LR: 1.0e-03 | Loss: 0.4046 | Acc: 87.22% | Time: 17.6s
    Epoch [24/25] | LR: 1.0e-03 | Loss: 0.4034 | Acc: 87.22% | Time: 18.3s
    Epoch [25/25] | LR: 1.0e-03 | Loss: 0.4036 | Acc: 87.36% | Time: 17.2s
基准测试完成 -> Final Acc: 87.36%, Total Time: 504.28s

============================================================
STEP 2: 运行并行测试 (Parallel / Simulated)
============================================================
仅检测到单卡，启用逻辑模拟 (Simulated 4 GPUs)...

>>> 启动训练 [模式: simulated] (Epochs: 25)
    Epoch [1/25] | LR: 1.0e-01 | Loss: 1.7929 | Acc: 57.25% | Time: 21.0s
    Epoch [2/25] | LR: 1.0e-01 | Loss: 1.2054 | Acc: 69.46% | Time: 22.0s
    Epoch [3/25] | LR: 1.0e-01 | Loss: 1.0238 | Acc: 71.10% | Time: 21.2s
    Epoch [4/25] | LR: 1.0e-01 | Loss: 0.9376 | Acc: 74.39% | Time: 21.1s
    Epoch [5/25] | LR: 1.0e-01 | Loss: 0.8869 | Acc: 75.61% | Time: 20.8s
    Epoch [6/25] | LR: 1.0e-01 | Loss: 0.8451 | Acc: 77.55% | Time: 20.0s
    Epoch [7/25] | LR: 1.0e-01 | Loss: 0.8135 | Acc: 78.24% | Time: 20.0s
    Epoch [8/25] | LR: 1.0e-01 | Loss: 0.7886 | Acc: 74.69% | Time: 20.7s
    Epoch [9/25] | LR: 1.0e-01 | Loss: 0.7661 | Acc: 79.13% | Time: 20.9s
    Epoch [10/25] | LR: 1.0e-01 | Loss: 0.7510 | Acc: 77.86% | Time: 21.5s
    Epoch [11/25] | LR: 1.0e-02 | Loss: 0.6228 | Acc: 84.76% | Time: 21.3s
    Epoch [12/25] | LR: 1.0e-02 | Loss: 0.5742 | Acc: 85.61% | Time: 20.9s
    Epoch [13/25] | LR: 1.0e-02 | Loss: 0.5550 | Acc: 85.50% | Time: 20.1s
    Epoch [14/25] | LR: 1.0e-02 | Loss: 0.5382 | Acc: 85.58% | Time: 20.0s
    Epoch [15/25] | LR: 1.0e-02 | Loss: 0.5242 | Acc: 86.17% | Time: 20.9s
    Epoch [16/25] | LR: 1.0e-02 | Loss: 0.5141 | Acc: 86.43% | Time: 20.9s
    Epoch [17/25] | LR: 1.0e-02 | Loss: 0.5044 | Acc: 86.32% | Time: 20.8s
    Epoch [18/25] | LR: 1.0e-02 | Loss: 0.4921 | Acc: 86.18% | Time: 20.8s
    Epoch [19/25] | LR: 1.0e-02 | Loss: 0.4891 | Acc: 86.43% | Time: 19.7s
    Epoch [20/25] | LR: 1.0e-02 | Loss: 0.4813 | Acc: 86.97% | Time: 19.9s
    Epoch [21/25] | LR: 1.0e-03 | Loss: 0.4606 | Acc: 87.77% | Time: 20.7s
    Epoch [22/25] | LR: 1.0e-03 | Loss: 0.4480 | Acc: 87.83% | Time: 20.5s
    Epoch [23/25] | LR: 1.0e-03 | Loss: 0.4421 | Acc: 87.88% | Time: 20.8s
    Epoch [24/25] | LR: 1.0e-03 | Loss: 0.4405 | Acc: 87.86% | Time: 20.2s
    Epoch [25/25] | LR: 1.0e-03 | Loss: 0.4400 | Acc: 87.74% | Time: 19.7s


############################################################
             CIFAR-10 并行计算实验报告
############################################################
模型架构: VGG-Style ImprovedNet (Epochs=25)
------------------------------------------------------------
指标                        | 基准 (Standard)      | 并行/模拟 (Parallel)  
------------------------------------------------------------
最终准确率                    | 87.36             % | 87.74             %
总训练耗时 (s)                | 504.28             | 573.91            
平均Batch耗时 (ms)            | 19.24              | 34.80             
------------------------------------------------------------

【作业分析结论 - 单卡模拟环境】
1. 准确率对比: 两者准确率非常接近 (87.36% vs 87.74%)，
   证明了 '并行切分逻辑' 与 '串行逻辑' 在数学上是等价的。

2. 耗时分析: 模拟并行比基准慢了 15.56 ms/batch。
   原因: 额外的切分(Chunk)和合并(Cat)操作，以及 Python 循环开销。

3. 等效并行时间推算 (假设拥有 4 张显卡):
   理论 Batch 耗时 = 8.70 ms
   预期加速比      = 2.21 倍