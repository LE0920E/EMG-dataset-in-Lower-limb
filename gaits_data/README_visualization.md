# EMG Data Visualization Tool

## 概述

这个Python工具提供了对下肢EMG数据的全面可视化和分析功能。工具支持数据概览、多通道信号可视化、统计分析以及异常/正常组对比分析。

## 功能特性

### 1. 数据概览与统计
- 文件数量统计（异常组 vs 正常组）
- 动作类型分布分析
- 数据持续时间统计
- 各通道信号统计信息

### 2. 多通道信号可视化
- 5个通道同时显示（4个EMG通道 + 1个膝关节角度通道）
- 时间序列信号展示
- 实时统计信息显示

### 3. 组间对比分析
- 异常组与正常组信号对比
- 不同动作类型的信号特征分析
- 统计显著性检验

### 4. 统计分析
- 文件分布统计图
- 数据持续时间箱线图
- 通道相关性热力图
- 信号方差分析

### 5. 交互式探索
- 命令行交互界面
- 动态文件选择
- 自定义参数设置

## 使用方法

### 基本使用

1. **直接运行脚本**：
```bash
cd gaits_data
python data_visualization.py
```

### 使用方法

#### 1. 直接运行脚本
```bash
cd gaits_data
python data_visualization.py
```

#### 2. 在Python中导入使用
```python
from data_visualization import EMGDataVisualizer

# 初始化可视化器
visualizer = EMGDataVisualizer("path/to/gaits_data")

# 加载数据
visualizer.load_all_files()

# 显示数据概览
visualizer.show_data_overview()

# 绘制通道对比图
visualizer.plot_channel_comparison("1Agait - Sheet1.csv")

# 显示统计摘要（6个独立图表）
visualizer.plot_statistical_summary()
```

#### 3. 交互式探索模式

运行脚本后，可以使用交互式菜单：

```
Available Options:
1. Show data overview
2. Plot specific file channels
3. Compare groups for specific action
4. Show statistical summary
5. List available files
6. Exit
```

## 数据通道说明

| 通道 | 英文名称 | 中文名称 | 信号类型 |
|------|----------|----------|----------|
| Ch1 | Recto Femoral | 股直肌 | EMG信号 |
| Ch2 | Biceps Femoral | 股二头肌 | EMG信号 |
| Ch3 | Vasto Medial | 股内侧肌 | EMG信号 |
| Ch4 | EMG Semitendinoso | 半腱肌 | EMG信号 |
| Ch5 | Flexo-Extension | 膝关节角度 | 角度信号 |

## 动作类型说明

| 动作代码 | 英文名称 | 中文名称 | 描述 |
|----------|----------|----------|------|
| gait | Gait | 步行 | 正常行走动作 |
| sitting | Sitting Extension | 坐姿伸腿 | 坐姿膝关节伸展 |
| standing | Standing Flexion | 站立屈膝 | 站立膝关节屈曲 |

## 数据文件结构

```
gaits_data/
├── Abnormal/           # 异常组数据
│   ├── 1Agait - Sheet1.csv
│   ├── 1Asitting - Sheet1.csv
│   └── ...
└── normal/             # 正常组数据
    ├── 1Ngait - Sheet1.csv
    ├── 1Nsitting - Sheet1.csv
    └── ...
```

## 技术细节

### 数据预处理
- 采样频率：1000 Hz
- 滤波处理：巴特沃斯带通滤波（20-450 Hz）
- 标准化：Z-score标准化
- 窗口切片：1000个时间点（1秒）

### 统计分析
- 均值、标准差计算
- 相关性分析
- 方差比较
- 分布特征分析

### 可视化参数
- 时间轴单位：秒
- 信号幅度：标准化后的相对值
- 颜色编码：异常组（红色），正常组（蓝色）

## 依赖库

- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## 示例输出

### 数据概览示例
```
EMG DATA OVERVIEW
================================================================================
Total files: 66
Abnormal group: 33 files
Normal group: 33 files

Files by Action Type:
  步行 (Gait): 22 files
  坐姿伸腿 (Sitting Extension): 22 files
  站立屈膝 (Standing Flexion): 22 files

Data Duration Statistics:
  Average duration: 15.30 seconds
  Minimum duration: 12.97 seconds
  Maximum duration: 16.50 seconds

Channel Signal Statistics (Mean ± Std):
  Ch1 - 股直肌 (RF): 0.0012 ± 0.0456
  Ch2 - 股二头肌 (BF): -0.0023 ± 0.0389
  Ch3 - 股内侧肌 (VM): 0.0034 ± 0.0521
  Ch4 - 半腱肌 (ST): 0.0018 ± 0.0412
  Ch5 - 膝关节角度: -10.2345 ± 3.5678
```

### 可视化图表类型
1. **多通道时间序列图**：显示5个通道的同步信号
2. **组间对比图**：异常组与正常组的信号对比
3. **统计分布图**：文件分布、持续时间、信号特征等
4. **相关性热力图**：EMG通道间的相关性分析

## 故障排除

### 常见问题

1. **文件加载失败**：检查文件路径和权限
2. **内存不足**：数据文件较大，确保有足够内存
3. **依赖库缺失**：使用`pip install pandas numpy matplotlib seaborn`安装

### 性能优化
- 首次运行会缓存数据，后续运行更快
- 可以调整采样点数减少内存使用
- 使用较小的时间窗口进行初步分析

## 图片自动保存功能

### 功能说明

工具现在支持自动保存所有生成的图表到`plots`目录，无需手动截图或保存。

### 保存的图表类型

1. **多通道信号对比图**
   - 文件名格式：`channel_comparison_<文件名>_<时间戳>.png`
   - 示例：`channel_comparison_10Agait_-_Sheet1_20251020_144417.png`

2. **组间对比图**
   - 文件名格式：`group_comparison_<动作类型>_<时间戳>.png`
   - 示例：`group_comparison_gait_20251020_144445.png`

3. **统计摘要图系列**（6个独立图表）
   - **文件分布柱状图**：`file_distribution_<时间戳>.png`
   - **持续时间箱线图**：`duration_distribution_<时间戳>.png`
   - **通道均值对比图**：`channel_means_<时间戳>.png`
   - **相关性热力图**：`correlation_heatmap_<时间戳>.png`
   - **通道方差对比图**：`channel_variance_<时间戳>.png`
   - **动作类型分布饼图**：`action_distribution_<时间戳>.png`

### 使用方法

图表会在调用相应绘图函数时自动保存，无需额外操作：

```python
from data_visualization import EMGDataVisualizer

visualizer = EMGDataVisualizer(".")
visualizer.load_all_files()

# 以下调用会自动保存图片到plots目录
visualizer.plot_channel_comparison()  # 自动保存
visualizer.plot_group_comparison("gait")  # 自动保存
visualizer.plot_statistical_summary()  # 自动保存
```

### 图表说明文档

详细的图表含义和解读指南请参考：[chart_explanations.md](chart_explanations.md)

## 扩展功能

工具支持以下扩展：
- 自定义分析参数
- 导出分析结果
- 批量处理功能
- 高级统计分析
- 图片自动保存功能

## 技术支持

如有问题或建议，请参考项目文档或联系开发团队。