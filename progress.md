# 会话进展

## 2026-04-13
- 收到需求：为 `cosmos_predict2/_src/predict2/rl` 增加光流一致性奖励
- 确认关键入口：`cosmos_predict2/experiments/base/action.py`
- 初始化规划文件：`task_plan.md`、`findings.md`、`progress.md`
- 已完成第一轮代码调研：
  - reward 抽象位于 `cosmos_predict2/_src/predict2/rl/reward.py`
  - GRPO reward 接线位于 `cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_grpo_model.py`
  - 实验配置入口位于 `cosmos_predict2/experiments/base/action.py`
- 已确认 rollout 结束后 reward 侧可以同时拿到：
  - 解码后的生成视频 `inp.video`
  - GT 视频 `inp.metadata["gt_video"]`
- 已确认仓库依赖中已有 `torchvision`、`opencv-python`，具备实现光流奖励的基础条件
- 当前判断：新增 reward 的最小改动面主要在 `reward.py`、GRPO model 的 reward type 分发、以及 `action.py` / model config 的 reward 参数配置
- 下一步：把光流 reward 的落地方案细化成明确改动点与推荐公式
