# GRPO 代码改动记录（简版）

## 新增目录
- `cosmos_predict2/_src/predict2/rl/`
  - **用途**：放置 GRPO/RL 相关的 sampler、reward 接口等通用组件（与现有训练/推理解耦）。

## 新增文件
- `cosmos_predict2/_src/predict2/rl/__init__.py`
  - **内容**：RL 组件包的说明注释。
- `cosmos_predict2/_src/predict2/rl/reward_base.py`
  - **内容**：Reward 接口定义（`RewardInput` 数据结构 + `BaseRewardModel` 抽象类）。
- `cosmos_predict2/_src/predict2/rl/reward_dummy.py`
  - **内容**：Dummy reward（`DummyRewardModel`），用于先打通 GRPO 训练闭环（默认返回 0 reward）。
- `cosmos_predict2/_src/predict2/rl/grpo_sde_sampler.py`
  - **内容**：GRPO 的 SDE step（`grpo_sde_step`），支持基于 UniPC `sigmas` 计算 `next_latents` 与 step-level `log_prob`。
- `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_grpo_model.py`
  - **内容**：GRPO 版模型（`ActionVideo2WorldModelRectifiedFlowGRPO` + Config），通过覆盖 `training_step` 实现：online rollout → reward → advantage → clipped loss（保持 `ImaginaireTrainer` 不变）。

## 修改文件
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/model.py`
  - **改动**：新增 model group `action_conditioned_video2world_fsdp_rectified_flow_grpo` 注册，并指向 GRPO 版模型类与其 config 默认值。
- `cosmos_predict2/experiments/base/action.py`
  - **改动**：新增 experiment `ac_reason_embeddings_rectified_flow_2b_256_320_grpo`，用于通过 `-- experiment=...` 启动 GRPO 配置（override 到 GRPO model group，并注入 `model.config.grpo/reward` 超参占位）。


