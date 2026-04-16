# 任务规划

## 任务目标
为 cosmos-predict 的 RL 微调流程新增一项“光流一致性奖励”：对生成视频与 GT 视频分别计算光流，再依据两者差异得到奖励分数。当前奖励定义尚未最终确定，先规划最小可行实现，并评估更稳健的替代方案。

## 当前上下文
- 主要代码目录：`cosmos_predict2/_src/predict2/rl`
- 启动配置：`cosmos_predict2/experiments/base/action.py`
- 已知现状：仓库中已经存在若干奖励设计，需要复用现有奖励注册、执行和聚合方式。

## 阶段

### 阶段 1：梳理现有奖励架构
- 找到奖励类/函数定义位置
- 找到奖励注册与配置注入方式
- 确认奖励计算时可拿到的数据：生成视频、GT 视频、batch 元信息、时序维度/分辨率
- 确认奖励是逐样本、逐帧还是逐 token 聚合

### 阶段 2：设计光流奖励
- 明确可插入点和依赖边界
- 设计最小可行版本：`reward = - MSE(flow_pred, flow_gt)` 或等价归一化版本
- 评估改进方案，例如：
  - Charbonnier / Huber 替代纯 MSE，降低异常流值敏感度
  - 仅比较光流幅值与方向，避免绝对尺度过于主导
  - 加入遮罩或低运动区域降权，避免静止背景放大奖励
  - 多尺度光流一致性
- 评估计算成本与训练吞吐影响

### 阶段 3：输出实施计划
- 列出需要修改的文件
- 明确配置项新增方式
- 明确验证方案（单测/离线检查/小规模训练 smoke test）

## 已确认结论
- reward 统一接口已存在：`RewardInput` + `BaseRewardModel`
- GRPO rollout 结束后会 decode 最终 latent，并把 `gt_video` 放进 `metadata`
- 新 reward 可以直接复用当前 reward 调用链，不需要改 GRPO 主框架
- 当前 reward 选择方式是 `reward.type` 的 `if/elif` 分发
- 仓库已具备 `torchvision`、`opencv-python` 依赖，可支持光流实现

## 待决策项
- 光流提取方案：先调研仓内是否已有实现或依赖；若没有，再决定使用已有库还是新增实现
- 奖励公式：先以 MSE 为基线，同时准备更鲁棒的替代公式
- 奖励粒度：视频级标量 / 帧级聚合

## 当前建议
- 第一版先做单 reward：`optical_flow`
- 第一版 reward 类直接放入 `cosmos_predict2/_src/predict2/rl/reward.py`
- 公式优先使用鲁棒版流场差异（如 Charbonnier/Huber），MSE 作为可切换 baseline
- 如果训练开销过高，优先采用下采样视频后再估计 flow

## 进度
- [x] 创建规划文件
- [x] 调研现有奖励架构
- [x] 形成光流奖励设计建议（第一轮）
- [ ] 输出具体修改计划
