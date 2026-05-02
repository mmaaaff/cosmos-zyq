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
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/reward.py`，使用 hydra 注册了各类 reward
- `cosmos_predict2/experiments/scripts`: 一些训练和推理的 bash 脚本
- `assets/action_conditioned/basic/concate_videos.py`: 拼接视频进行可视化对比
- `cosmos_predict2/_src/predict2/action/callbacks/rollout_reward_validation.py`: 添加了一个验证 callback 函数
- 

## 修改文件
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/model.py`
  - **改动**：新增 model group `action_conditioned_video2world_fsdp_rectified_flow_grpo` 注册，并指向 GRPO 版模型类与其 config 默认值。
- `cosmos_predict2/experiments/base/action.py`
  - **改动**：新增 experiment `ac_reason_embeddings_rectified_flow_2b_256_320_grpo`，用于通过 `-- experiment=...` 启动 GRPO 配置（override 到 GRPO model group，并注入 `model.config.grpo/reward` 超参占位）。

- `cosmos_predict2/_src/predict2/callbacks/wandb_log.py`，增加了对 grpo 参数记录的支持

- 针对 action conditioned 示例推理代码无法调整 num_steps 问题：
  - `cosmos_predict2/action_conditioned_config.py` L66 新增 num_steps 参数
  - `cosmos_predict2/action_conditioned.py` L347 将 num_steps 参数传入
  - `cosmos-zyq/assets/action_conditioned/basic/inference_params.json` 加入了 num_steps 参数的设置选项

- 原代码 bug
  - `cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py` L599。原版的实现有 bug，会导致 generate_samples_from_batch 无法正确对一批次样本进行去噪，而是输出几乎重复的批次内第一个样本的去噪结果。这会导致训练过程中记录的图片样例有问题（由 every_n_sample_reg 参数控制，具体逻辑在 cosmos_predict2/_src/predict2/callbacks/every_n_draw_sample.py 的 L307
  - `cosmos_predict2/_src/imaginaire/utils/wandb_util.py` L43，加入判断逻辑避免 wandb 重复初始化（默认每个实验会起一个 wandb 和一个 wandb_10x）
  - `cosmos_predict2/_src/imaginaire/utils/callback.py` L76, 添加对于 None 的 callback 的跳过逻辑

- 其他
  - `co-tracker/cotracker/models/core/cotracker/cotracker3_offline.py`, L141 .view() -> .reshape()