# UNet训练优化建议

## 常见问题诊断

### 1. Dice Score过低的可能原因

**数据相关问题：**
- 标签值范围不正确（应该是0-1或0-255）
- 数据集标注质量差
- 训练集和验证集分布不一致
- 类别不平衡严重

**模型相关问题：**
- 学习率设置不当
- 损失函数与评估指标不匹配
- 网络架构不适合当前任务
- 过拟合或欠拟合

**训练相关问题：**
- 训练轮数不足
- 批次大小不合适
- 数据增强策略不当

## 自动化诊断工具

### 全面诊断脚本
```bash
# 运行完整诊断套件（推荐）
chmod +x comprehensive_diagnostics.sh
./comprehensive_diagnostics.sh
```

### 快速诊断脚本
```bash
# 运行核心测试（快速）
chmod +x quick_diagnostics.sh  
./quick_diagnostics.sh
```

## 推荐的调试流程

### 第一步：运行调试脚本
```bash
python debug_training.py --img-dir ./data/imgs/ --mask-dir ./data/masks/
```

### 第二步：检查数据质量
```bash
# 查看样本图像和标签
python -c "
import torch
from utils.data_loading import BasicDataset
dataset = BasicDataset('./data/imgs/', './data/masks/', scale=0.5)
sample = dataset[0]
print('Image shape:', sample['image'].shape)
print('Mask shape:', sample['mask'].shape)
print('Image range:', sample['image'].min(), 'to', sample['image'].max())
print('Mask range:', sample['mask'].min(), 'to', sample['mask'].max())
"
```

### 第三步：从小规模开始训练
```bash
# 使用小数据集快速验证
python train.py --epochs 5 --batch-size 2 --learning-rate 1e-4 --subset 0.1
```

## 参数调优建议

### 学习率调整
```bash
# 二分法寻找合适学习率
python train.py --learning-rate 1e-3  # 如果不稳定，降低
python train.py --learning-rate 1e-4  # 通常比较安全的起点
python train.py --learning-rate 1e-5  # 如果收敛太慢，提高
```

### 批次大小优化
```bash
# 根据GPU内存调整
python train.py --batch-size 1   # 最小批次
python train.py --batch-size 2   # 推荐起点
python train.py --batch-size 4   # 如果内存允许
python train.py --batch-size 8   # 大批次训练
```

### 数据增强策略
考虑添加以下增强：
- 随机旋转
- 水平/垂直翻转  
- 颜色抖动
- 弹性变形

## 监控训练过程

### 使用TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir=runs

# 在浏览器中访问 http://localhost:6006
```

### 关键监控指标
- 训练损失曲线（应该逐渐下降）
- 验证Dice分数（应该逐渐上升）
- 学习率变化
- 梯度范数（避免梯度爆炸）

## 故障排除清单

### ✅ 数据检查
- [ ] 图像和标签文件配对正确
- [ ] 标签值在正确范围内
- [ ] 数据集划分合理
- [ ] 没有损坏的文件

### ✅ 模型检查  
- [ ] 网络输入输出维度匹配
- [ ] 损失函数选择正确
- [ ] 参数初始化合理
- [ ] 没有梯度消失/爆炸

### ✅ 训练检查
- [ ] 学习率适中
- [ ] 批次大小合适
- [ ] 训练轮数充足
- [ ] 早停机制工作正常

## 成功训练的标志

当看到以下情况时，说明训练正常：
1. 训练损失稳步下降
2. 验证Dice分数持续提升
3. 训练和验证曲线趋势一致
4. 最终验证Dice > 0.7（一般标准）

如果仍然遇到问题，请提供：
- 完整的错误日志
- 数据样本示例
- 训练参数配置
- TensorBoard截图