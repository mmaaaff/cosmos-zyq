# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
import os
import signal

import torch
import torch.distributed as dist
import torch.utils.data

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils.context_managers import distributed_init
from cosmos_predict2._src.imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")


from cosmos_predict2._src.imaginaire.lazy_config import LazyConfig, instantiate
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import callback, distributed, ema, log, misc
from cosmos_predict2._src.imaginaire.utils.checkpointer import Checkpointer
from cosmos_predict2._src.imaginaire.utils.misc import StragglerDetectorV2


class ImaginaireTrainer:
    """The base trainer class of Imaginaire.

    All trainers in Imaginaire should inherit ImaginaireTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        """Constructor of the trainer.

        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        super().__init__()
        self.config = config
        # Set up the distributed computing environment.
        with distributed_init():
            distributed.init()
            # Set up parallel states.
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            if USE_MEGATRON:
                if (
                    "create_gloo_process_groups"
                    in inspect.signature(parallel_state.initialize_model_parallel).parameters
                ):
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                        create_gloo_process_groups=False,
                    )
                else:
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                    )
                # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
                # It is not part of the original `parallel_state` API, so we need to set it manually.
                parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
                if parallel_state.sequence_parallel:
                    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local log.
        if distributed.is_rank0():
            os.makedirs(config.job.path_local, exist_ok=True)
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        dist.barrier()
        if INTERNAL:
            log.init_loguru_file(f"{config.job.path_local}/stdout.log")
            if distributed.is_rank0():
                # Print important environment variables and the effective config.
                log.info("Config:\n" + config.pretty_print(use_color=True))
            misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT", "ENABLE_ONELOGGER"])
        else:
            misc.print_environ_variables(["HF_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # Initialize the timer for speed benchmarking.
        self.training_timer = misc.TrainingTimer()
        # Initialize Straggler Detection
        self.straggler_detector = StragglerDetectorV2(
            enabled=self.config.trainer.straggler_detection.enabled,
            report_freq=self.config.trainer.straggler_detection.report_freq,
            profile_freq=self.config.trainer.straggler_detection.profile_freq,
            max_diff=self.config.trainer.straggler_detection.max_diff,
            raise_error=self.config.trainer.straggler_detection.raise_error,
        )
        self.straggler_detector.initialize()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting training...")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
        _end_training = False
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            while True:
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with (
                            self.training_timer("dataloader_train"),
                            self.straggler_detector.profile_section(
                                "dataloading",
                                self.config.trainer.straggler_detection.analyze_dataloading,
                                profile_cuda=False,
                            ),
                        ):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                    )
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    # Save checkpoint.
                    if iteration % self.config.checkpoint.save_iter == 0:
                        self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    self.straggler_detector.generate_report(iteration)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                if _end_training:
                    break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """The training step.

        Args:
            model_ddp (torch.nn.Module | distributed.DistributedDataParallel): The model with a DDP wrapper or, the bare
              module, depending on whether distributed training is enabled or not.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            data (dict[str, torch.Tensor]): Data batch (dictionary of tensors).
            iteration (int): Current iteration number.
            grad_accum_iter (int): Number of gradient accumulation iterations.

        Returns:
            output (dict[str, torch.Tensor]): The model output from the training data batch (dictionary of tensors).
            loss (torch.Tensor): The total loss of the training data batch.
        """
        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            with self.training_timer("forward"):
                with self.straggler_detector.profile_section(
                    "fwd", self.config.trainer.straggler_detection.analyze_forward
                ):
                    output_batch, loss = model_ddp.training_step(data, iteration)
            self.callbacks.on_after_forward(iteration=iteration)
            self.callbacks.on_before_backward(model_ddp, loss, iteration=iteration)
            with self.training_timer("backward"):
                with self.straggler_detector.profile_section(
                    "bwd", self.config.trainer.straggler_detection.analyze_backward
                ):
                    loss_scaled = grad_scaler.scale(loss / self.config.trainer.grad_accum_iter)
                    loss_scaled.backward()
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_after_backward()
                    else:
                        model_ddp.on_after_backward()
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)
        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                with self.straggler_detector.profile_section(
                    "opt", self.config.trainer.straggler_detection.analyze_optimizer
                ):
                    self.callbacks.on_before_optimizer_step(
                        model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step()
                    self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    else:
                        model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                    optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0
        return output_batch, loss, grad_accum_iter

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        # Evaluate on the full validation set.
        with ema.ema_scope(model, enabled=model.config.ema.enabled):
            for val_iter, data_batch in enumerate(dataloader_val):
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)


class trainer_grpo(ImaginaireTrainer):
    """
    GRPO trainer.

    Key difference from `ImaginaireTrainer`:
    - Outer loop collects a rollout batch once.
    - Inner loop performs multiple optimizer updates on the *same* rollout batch, so `old_log_probs` stay fixed and
      `new_log_probs` change across updates (matching DanceGRPO semantics).

    Annotation:
    - This trainer expects the model to implement:
        - `collect_rollout_and_rewards(data_batch) -> GrpoRolloutSamples`
        - `compute_grpo_loss(samples, update_seed) -> (output_batch, loss)`
    """

    def _concat_batches(self, batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Concatenate a list of data_batch dicts along dim=0.

        Annotation:
        - For tensors with a batch dimension, we concatenate on dim=0.
        - For lists/tuples, we concatenate by addition.
        - For other types, we keep the last value.
        """

        assert len(batches) > 0
        out: dict[str, torch.Tensor] = {}
        keys = set().union(*[b.keys() for b in batches])
        for k in keys:
            vals = [b[k] for b in batches if k in b]
            v0 = vals[0]
            if torch.is_tensor(v0):
                out[k] = torch.cat([v for v in vals], dim=0)
            elif isinstance(v0, list):
                merged = []
                for v in vals:
                    merged.extend(v)
                out[k] = merged  # type: ignore[assignment]
            elif isinstance(v0, tuple):
                merged_t = []
                for v in vals:
                    merged_t.extend(list(v))
                out[k] = tuple(merged_t)  # type: ignore[assignment]
            else:
                out[k] = v0  # type: ignore[assignment]
        return out

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
    ) -> None:
        # Same initialization as base trainer
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()

        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        if self.config.trainer.distributed_parallelism == "ddp":
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")

        log.info("Starting GRPO training...")
        self.callbacks.on_train_start(model, iteration=iteration)

        _end_training = False
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            dataloader_train_iter = iter(dataloader_train)
            while True:
                if iteration >= self.config.trainer.max_iter:
                    break
                
                print("New Rollout")

                # -------------------- Outer loop: collect rollout batch --------------------
                # NOTE: `rollout_num_batches` is stored in model.config.grpo (dict) for simplicity.
                rollout_num_batches = int(getattr(getattr(model, "config", None), "grpo", {}).get("rollout_num_batches", 1))  # type: ignore[union-attr]
                rollout_batches = []
                for _ in range(max(1, rollout_num_batches)):
                    try:
                        data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        dataloader_train_iter = iter(dataloader_train)
                        data_batch = next(dataloader_train_iter)
                    rollout_batches.append(data_batch)

                # Move to cuda and concat into a larger rollout batch
                rollout_batches = [misc.to(b, device="cuda") for b in rollout_batches]
                rollout_batch = self._concat_batches(rollout_batches)

                # Rollout should be deterministic w.r.t. policy noise; prefer eval mode for rollout
                model_ddp.eval()
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.eval()

                self.callbacks.on_training_step_start(model, rollout_batch, iteration=iteration)

                with torch.no_grad():
                    # 逐 batch 收集 rollout 样本并保存为一个列表
                    samples_list = []
                    for batch_idx, b in enumerate(rollout_batches):
                        print("new rollout batch")
                        samples_list.append(
                            model_ddp.collect_rollout_and_rewards(  # 每次设置不同的 seed 避免使用同样的初始 noise
                                b, rollout_seed_offset=iteration * 1_000_000 + batch_idx * 1_000
                            )
                        )

                # -------------------- Inner loop: multiple updates on same rollout --------------------
                num_updates = int(getattr(getattr(model, "config", None), "grpo", {}).get("num_updates", 1))  # type: ignore[union-attr]


                # 目前是按照 rollout 阶段未打乱的 batch 进行更新，后续可以考虑按照打乱后的 batch 进行更新
                for update_idx in range(num_updates):
                    print(f"update_idx = {update_idx}")
                    # Switch to train mode for policy update
                    model_ddp.train()
                    if self.config.trainer.distributed_parallelism == "ddp":
                        model_ddp.module.train()

                    total_b = sum(int(s.rewards.shape[0]) for s in samples_list)  # 总样本数
                    total_b = max(1, total_b)

                    output_batch_accum: dict[str, torch.Tensor] = {}
                    last_loss: torch.Tensor | None = None

                    for batch_idx, s in enumerate(samples_list):
                        print(f"batch_idx = {batch_idx}")
                        w = float(int(s.rewards.shape[0])) / float(total_b)  # 本批次权重

                        # DDP 只在 accume 到最后要更新的那一步的时候才同步梯度
                        sync_grad = grad_accum_iter == self.config.trainer.grad_accum_iter - 1
                        with distributed.ddp_sync_grad(model_ddp, sync_grad):
                            self.callbacks.on_before_forward(iteration=iteration)
                            out_i, loss_i = model_ddp.compute_grpo_loss(  # type: ignore[attr-defined]
                                s, update_seed=iteration * 100000 + update_idx * 1000 + batch_idx
                            )
                            self.callbacks.on_after_forward(iteration=iteration)

                            # Weight the micro loss and normalize by grad_accum_iter
                            loss_micro = loss_i * w
                            last_loss = loss_micro
                            # print(f"loss_i: {loss_i}")
                            # print(f"last_loss: {last_loss}")

                            self.callbacks.on_before_backward(model_ddp, loss_micro, iteration=iteration)
                            loss_scaled = grad_scaler.scale(loss_micro / self.config.trainer.grad_accum_iter)
                            loss_scaled.backward()
                            if self.config.trainer.distributed_parallelism == "ddp":
                                model_ddp.module.on_after_backward()
                            else:
                                model_ddp.on_after_backward()
                            self.callbacks.on_after_backward(model_ddp, iteration=iteration)

                        # 标量加权平均，tensor 拼接
                        b_i = int(s.rewards.shape[0])
                        for k, v in out_i.items():
                            if torch.is_tensor(v):
                                if v.ndim == 0:
                                    output_batch_accum[k] = output_batch_accum.get(k, torch.zeros_like(v)) + v.detach() * w
                                elif v.ndim >= 1 and v.shape[0] == b_i:
                                    if k not in output_batch_accum:
                                        output_batch_accum[k] = v.detach()
                                    else:
                                        output_batch_accum[k] = torch.cat([output_batch_accum[k], v.detach()], dim=0)
                                else:
                                    # Shape is not batch-aligned
                                    # output_batch_accum[k] = v.detach()
                                    raise ValueError(f"Shape is not batch-aligned: {v.shape} for key {k}")
                            else:
                                # Non-tensor (e.g., a condition object)
                                raise ValueError(f"Non-tensor: {type(v)} for key {k}")

                        grad_accum_iter += 1

                        if grad_accum_iter == self.config.trainer.grad_accum_iter:
                            self.callbacks.on_before_optimizer_step(
                                model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                            )
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            scheduler.step()

                            self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                            if self.config.trainer.distributed_parallelism == "ddp":
                                model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                            else:
                                model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                            optimizer.zero_grad(set_to_none=True)
                            grad_accum_iter = 0

                            # Treat each optimizer step as an iteration
                            iteration += 1
                            # For callback signatures, pass the (weighted) output_batch stats; loss is the last micro loss.
                            # print(f"last_loss: {last_loss}")
                            if last_loss is None:
                                last_loss = torch.zeros((), device="cuda")
                            self.callbacks.on_training_step_batch_end(
                                model, rollout_batch, output_batch_accum, last_loss, iteration=iteration
                            )
                            self.callbacks.on_training_step_end(
                                model, rollout_batch, output_batch_accum, last_loss, iteration=iteration
                            )

                            if iteration % self.config.checkpoint.save_iter == 0:
                                self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)

                            if torch_profiler:
                                torch_profiler.step()
                            if memory_profiler:
                                memory_profiler.step()

                            if iteration >= self.config.trainer.max_iter:
                                _end_training = True
                                break
                            
                            output_batch_accum = {}
                            last_loss = None

                    if _end_training:
                        break

                if _end_training:
                    break

        log.success("Done with GRPO training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()