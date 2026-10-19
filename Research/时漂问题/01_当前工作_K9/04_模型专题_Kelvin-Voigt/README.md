# 04_模型专题_Kelvin-Voigt

> 建立时间：2026-09-27 ｜ 性质：模型专题（零上下文论文 + 数据演算复现）

## 这是什么

一篇**只讲 Kelvin-Voigt 模型自身**的文章（论文体，不含项目上下文、代码段与路径），
按 `Document/具体算法/时漂、蠕变/` 下两篇文章（在线双态蠕变观测器、在线滤波器消除慢变基线游走）
的写作结构与绘图范式组织；实测演算部分只研究恒载**慢漂段**。

| 文件 | 说明 |
|---|---|
| `Kelvin-Voigt模型.md` | ★ 文章本体（零上下文） |
| `figure/fig01~fig07*.png` | 全部图件（7 张：总览、解析形态、**蠕变阶段划分**、动态行为、慢漂拟合、群体统计、模型边界） |
| `scripts/make_figures.py` | 数据演算 + 图件生成（自包含，重跑即复现全部数字） |
| `scripts/numbers.json` | 文中引用的全部数字汇总（拟合明细、群体统计、边界对照） |

## 数据与工具（溯源，文章内不出现）

- **数据**：`Research\时漂-蠕变问题研究\data\四指指腹\20260926_194738_single_device_c272fc`
  （52 通道 / 100.7 Hz / 96.3 s / 9698 帧；session.json：算法关闭、ADC 直读）。
  分段：加载沿 1.0~1.7 s，快相 2~5 s，**慢漂段 5~54 s**（研究窗口），87.9 s 整片卸载，89.9 s 重载。
- **画图**：`D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具`
  ——数据加载走 `dptool.api`；图件由 `dptool.snapshot`（FigureSnapshot 冻结契约）构造、
  `dptool.figure_export.export_png` 渲染。

## 复现

```powershell
cd "01_当前工作_K9\04_模型专题_Kelvin-Voigt\scripts"
C:\Python314\python.exe make_figures.py   # 重新生成 figure/*.png 与 numbers.json
```

依赖：numpy、scipy、matplotlib、dptool（数据解析工具包）。

## 文章主要结论速览

- 慢漂段单 Kelvin-Voigt 蠕变律拟合：38 活跃通道 = 12 无显著漂移（A<20 ADC）
  + 16 可辨识饱和（τ̂ 中位 8.7 s，IQR 5.1~14.2 s，A/v∞ 中位 5.2%）
  + 10 窗口内不饱和（τ̂ ≥21.5 s，只能给下界）；残差中位 3.0 ADC ≈ 噪声地板。
- 三条实测边界：① 加载沿瞬跳 53%~88% vs 模型 0.1%~7.4%（缺口 11~380×，需串联弹簧）；
  ② 卸载即回零（−6.4~+5.4 ADC）vs 延迟保留预测 68~1038 ADC（13~205×）——
  读数是应力侧响应，不带应变记忆，模型恢复分支不适用于读数；
  ③ 跨通道 τ̂ 差一个数量级以上，单一时间常数只对逐通道成立。
