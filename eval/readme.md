# 强化学习性能评价指标

### 目的
1. 验证算法的性能


### 常用评价指标

#### 平均reward

在一个episode中，将每一步的reward求和然后取平均。

一个episode定义为：从起始状态到结束状态。

### 多进程时如何评价？

可以将所有子进程的reward求平均， 并计算其中最小及最大的平均reward（针对每个子进程）。





