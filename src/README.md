# src/ — Phase 0 实现占位

Research Phase（本轮）只完成设计，不写模型代码。Phase 0 实现按以下模块填充：

- `evidence_state/`：证据单元清单加载、2^n 状态枚举、证据屏蔽/替换、页面渲染（HTML→截图 或 证据块序列）
- `blackbox_eval/`：vLLM 引擎封装（Qwen2.5-VL / InternVL）、teacher-forced U_ans 打分、U_cit 引用解析（[EV-i] 标记）、缓存与重试、噪声地板协议
- `fourier/`：Walsh–Hadamard 变换（枚举精确谱）、阶能量 E_k、模态分解（VV/VT/TT）、bootstrap CI
- `anova/`：Sobol 指数、submodularity ratio γ、curvature
- `permutation/`：排列采样（随机+循环移位）、M0–M3 回归阶梯、Mallows/GM 对照
- `set_function/`：边际增益表、单调性/次模违反率
- `statistics/`：噪声地板估计、BH 校正、null 合成函数、跨引擎相似度（cosine/Spearman/Jaccard + Procrustes）

设计细节见 `experiments/phase0_design.md`。
