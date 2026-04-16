# 研究记录

## 初始问题
用户希望为 cosmos-predict 的 RL 微调新增“光流一致性奖励”。该奖励需要比较生成视频与 GT 视频的运动模式，而不是只比较像素外观。

## 待验证问题
1. 当前 RL reward 的抽象层是什么？
2. reward 计算阶段是否已经同时持有生成视频与 GT 视频？
3. 现有 reward 是否支持外部重模型/预处理器？
4. 仓库中是否已有 optical flow 相关实现、依赖或可复用模块？
5. `action.py` 中如何启用/组合 reward？

## 当前代码发现
### 1. reward 抽象与入口
- `cosmos_predict2/_src/predict2/rl/reward.py` 定义了统一输入 `RewardInput(video, text, action, metadata)` 和基类 `BaseRewardModel`。
- 当前 reward 实现已经包含：`DummyRewardModel`、`SSIM_Reward`、`VJEPA2Reward`。
- reward 不是单独注册表，而是在 `action_conditioned_video2world_rectified_flow_grpo_model.py` 中通过 `reward.type` 做 `if/elif` 分发。

### 2. reward 计算时可拿到的数据
- 在 `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_grpo_model.py` 中，rollout 结束后会：
  1. 得到最终 `final_latents`
  2. 调用 `self.decode(final_latents)` 解码成像素域视频
  3. 从 `data_batch[input_key]` 取出 GT 视频并放入 `metadata["gt_video"]`
  4. 构造 `RewardInput(video=pred_video_pixels, action=..., metadata={"gt_video": gt_video, ...})`
- 这说明新增 reward 时，不需要改 rollout 主流程就能同时访问生成视频和 GT 视频。

### 3. 配置接线方式
- `cosmos_predict2/experiments/base/action.py` 中 GRPO 实验配置使用：
  - `model.config.grpo` 管 rollout / update 超参
  - `model.config.reward` 管 reward 类型与其参数
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/model.py` 中给出了 GRPO 模型默认 reward 配置，目前默认 `type="vjepa2"`。

### 4. 依赖与可复用能力
- 仓库依赖里已有 `torchvision` 与 `opencv-python`，因此实现光流奖励时无需先引入新基础依赖。
- 当前仓内没有明确现成的光流 reward/RAFT 封装；若要上学习式光流，较可能需要在 reward.py 中自行包一层 torchvision optical-flow 模型或自定义实现。
- 现有 `VJEPA2Reward` 已经展示了 reward 内部“按需加载大模型 + forward 后可 unload”的模式，可直接借鉴给光流模型 reward。

## 先验设计想法
### 基线方案
- 对生成视频和 GT 视频分别计算相邻帧光流
- 使用逐像素 MSE 比较两组光流
- 将损失映射成奖励，例如 `reward = -mean((flow_gen - flow_gt)^2)`

### 可改进方案
- 使用 Charbonnier / Huber 损失，提高鲁棒性
- 分别比较流场幅值和方向，减少异常大位移主导
- 按运动强度加权，只在 GT 有明显运动处提高约束
- 多尺度下计算一致性，提高对局部和全局运动的兼容性

## 当前推荐方向
### 最小可行实现（推荐先做）
新增 `OpticalFlowReward`，放在 `cosmos_predict2/_src/predict2/rl/reward.py`：
- 输入：`inp.video` 和 `inp.metadata["gt_video"]`
- 统一 shape 到 `[B, C, T, H, W]`
- 将每对相邻帧 `(t, t+1)` 送入光流估计器，得到 `flow_pred[t]`、`flow_gt[t]`
- reward 基线：`-mean((flow_pred - flow_gt)^2)`，按时间和空间维聚合成 `[B]`

### 第一版更稳健公式（比纯 MSE 更建议）
相比直接 MSE，我更建议首版用“归一化 EPE / Charbonnier”一类鲁棒形式：
- 先算 `diff = flow_pred - flow_gt`
- 再算 `sqrt(diff^2 + eps^2)` 或 Huber
- 最后取负号作为 reward

原因：
- 纯 MSE 对少数大位移区域过于敏感
- RL 场景下 reward 方差过大时，训练通常更不稳定
- 光流估计本身会有噪声，鲁棒损失更适合做奖励

### 可选增强
- 只在 GT 运动幅值较高区域计算 / 提高权重
- 分开比较 magnitude 与 angle，再线性组合
- 下采样后再算 flow，控制 rollout 开销
- 低频 reward：不是每次 rollout 都算重光流模型

## 注意点
- 光流计算可能显著增加 RL 训练开销
- 若 reward 运行在每 step，都需要考虑缓存、分辨率下采样或低频评估
- 如果 GT 与生成视频长度/帧率不完全一致，需要先确认对齐方式
- 当前 reward dispatch 是 `if/elif`，因此新增 reward 不仅要加类，还要同步扩展 `reward.type` 分发和默认配置结构
