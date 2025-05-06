import os
import time

import numpy as np
import torch
import transformers
from accelerate import Accelerator

from .utils import (
    get_world_size,
    gsi,
    plot_xy,
    print_rank_0,
    rank_0,
    save_fig,
)


class EmptycacheCallback(transformers.TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print_rank_0("Cache cleared [after step].")

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print_rank_0("Cache cleared [before train].")

    def on_init_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print_rank_0("Cache cleared [after init].")


class PT_ProfCallback(transformers.TrainerCallback):
    def __init__(self, warmup_step, key, output_dir: str = ""):
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, warmup=warmup_step, active=10, repeat=1
            ),
            profile_memory=True,
            with_stack=True,
            record_shapes=True,
        )
        self.warmup_step = warmup_step
        self.key = key
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        torch.cuda.memory._record_memory_history(max_entries=1048576)

    def on_step_begin(self, args, state, control, **kwargs):
        # To fix the bug with auto_find_batch_size=True
        if state.global_step == 1:
            self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= 1:
            self.prof.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.prof.stop()
        self.dump_trace()
        torch.cuda.memory._record_memory_history(enabled=None)

    @rank_0
    def dump_trace(self):
        if self.warmup_step > self.prof.step_num:
            print_rank_0(
                f"Detected the warmup steps ({self.warmup_step}) have exceeded the profiler steps ({self.prof.step_num}), you may not get any profiler infomation."
            )

        self.prof.export_chrome_trace(
            os.path.join(
                self.output_dir,
                f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.json",
            )
        )
        self.prof.export_memory_timeline(
            os.path.join(
                self.output_dir,
                f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.html",
            )
        )
        torch.cuda.memory._dump_snapshot(
            os.path.join(
                self.output_dir,
                f"Trace_{self.key}_step_{self.warmup_step}_to_{self.prof.step_num}.pickle",
            )
        )


# todo: detailed step info


class StepInfoCallback(transformers.TrainerCallback):
    def __init__(
        self,
        trainer,
        warmup_step,
        key,
        trainable_param,
        step_log: bool = False,
        output_dir: str = "",
    ):
        self.trainer = trainer
        self.warmup_step = warmup_step
        self.key = key
        self.output_dir = output_dir
        self.step_times = []
        self.step_log = step_log
        self.trainable_param = trainable_param

    def get_token_per_step(self) -> list:
        seq = self.trainer.get_trained_seq()
        return seq

    def on_step_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.end_time = time.time()
        self.step_times.append(self.end_time - self.start_time)

    def on_train_end(self, args, state, control, **kwargs):
        accelerator = Accelerator()
        global_step = state.global_step

        if self.warmup_step > global_step:
            print_rank_0(
                f"Detected the warmup steps ({self.warmup_step}) have exceeded the global steps ({global_step}), you may not get any profiler infomation."
            )

        # Get the step time
        mean_step_time = round(np.mean(self.step_times[self.warmup_step :]), 3)
        std_step_time = round(np.std(self.step_times[self.warmup_step :]), 3)

        # Get the FLOPs and FLOPS
        total_FLOPs = state.total_flos
        FLOPs_per_step_per_device = total_FLOPs / global_step / get_world_size()
        FLOPS_per_device = FLOPs_per_step_per_device / mean_step_time

        # Get the average sequence length
        seq = self.get_token_per_step()
        mean_seq = round(np.mean(seq[self.warmup_step :]), 3)

        # Get the throughput
        local_token_per_second = torch.tensor(
            round(mean_seq / mean_step_time, 2), device=accelerator.device
        )
        all_token_per_second = accelerator.gather(local_token_per_second).sum().item()

        # Get the peak memory
        local_mem = torch.tensor(
            (
                torch.cuda.mem_get_info(device=None)[1]
                - torch.cuda.mem_get_info(device=None)[0]
            )
            / 1024
            / 1024
            / 1024,
            device=accelerator.device,
        )
        peak_mem = accelerator.gather(local_mem).max().item()

        # Get the train log and eval log from state
        def filter_log_entry(entry):
            return {k: v for k, v in entry.items() if k != "step"}

        train_log = {}
        eval_log = {}
        log_history = state.log_history

        for his in log_history:
            try:
                if "loss" in his:
                    train_log[his["step"]] = his["loss"]
                elif "eval_loss" in his:
                    eval_log[his["step"]] = his["eval_loss"]
            except KeyError as e:
                print_rank_0(f"Key error: {e} in log entry {his}")
            except Exception as e:
                print_rank_0(f"Unexpected error: {e} in log entry {his}")

        # Dump the profile result to profiler.txt
        profile_dict = {}
        profile_dict["key"] = self.key
        profile_dict["per_device_batch_size"] = state.train_batch_size
        profile_dict["global_batch_size"] = state.train_batch_size * get_world_size()
        profile_dict["trainable_parameter"] = self.trainable_param
        profile_dict["step_time (s)"] = mean_step_time
        profile_dict["step_time_std (s)"] = std_step_time
        profile_dict["token/s"] = round(all_token_per_second, 2)
        profile_dict["FLOPs_per_step_per_device (TFLOPs)"] = round(
            FLOPs_per_step_per_device / 1e12, 3
        )
        profile_dict["FLOPS_per_device (TFLOPS)"] = round(FLOPS_per_device / 1e12, 3)
        profile_dict["mem (GB)"] = round(peak_mem, 2)

        if self.step_log:
            profile_dict["train_log"] = train_log
            profile_dict["eval_log"] = eval_log

        train_fig = plot_xy(
            list(train_log.keys()), list(train_log.values()), "train loss"
        )
        save_fig(train_fig, os.path.join(self.output_dir, "train.png"))

        gsi.info.update(profile_dict)
        gsi.dump(self.output_dir)


r"""
Sparse Aware Training

A sparse schedule is defined by sparsity_ratio, sparse_warmup, sparse_warmup_steps
Suppose the training will run n steps.
1. Sparse will only happen before the first sparse_warmup * n steps.
2. During the sparse, sparsity_ratio is iterately achieved, by number of sparse_warmup_steps.

|---25%---|--50%--|----------------|
"""


def build_sparsity_schedule(
    max_steps: int,
    sparse_warmup: float,
    sparse_ratio: float,
    sparse_end: float,
    sparse_steps: int = 1,
    mode: str = "linear",
) -> dict[int, float]:
    """
    Generate a sparsity-training schedule.

    Parameters
    ----------
    max_steps      : total training steps
    sparse_warmup  : relative point (0-1) where sparsity starts
    sparse_ratio   : target sparsity (0-1)
    sparse_end     : relative point (0-1) where target sparsity is reached
    sparse_steps   : number of pruning milestones (≥1)
    mode           : 'linear' | 'quadratic'

    Returns
    -------
    dict[int, float] : {global_step : sparsity_ratio}
    """
    # --------------------------- sanity checks ---------------------------
    if not (0 <= sparse_warmup <= 1 and 0 <= sparse_end <= 1):
        raise ValueError("sparse_warmup and sparse_end must be within [0, 1]")
    if not (0 <= sparse_ratio <= 1):
        raise ValueError("sparse_ratio must be within [0, 1]")
    if sparse_steps <= 0:
        raise ValueError("sparse_steps must be a positive integer")
    if mode not in {"linear", "quadratic"}:
        raise ValueError("mode must be 'linear' or 'quadratic'")
    if sparse_warmup > sparse_end:
        raise ValueError("sparse_warmup must be <= sparse_end")
    # when we need 1-step sparse - case 1
    if sparse_warmup == sparse_end:
        step = int(round(sparse_end * max_steps))
        return {step: sparse_ratio}

    start_step = int(round(sparse_warmup * max_steps))
    end_step   = int(round(sparse_end   * max_steps))

    schedule: dict[int, float] = {}

    # when we need 1-step sparse - case 2
    if sparse_steps == 1:
        schedule[start_step] = sparse_ratio
        return schedule

    interval = (end_step - start_step) / (sparse_steps - 1)

    for i in range(sparse_steps):               # i = 0 … sparse_steps-1
        t_global = int(round(start_step + i * interval))

        progress = (i + 1) / sparse_steps       # 1/s, 2/s, …, 1
        ratio = (
            sparse_ratio * progress
            if mode == "linear"
            else sparse_ratio * (progress ** 2)
        )

        if i == sparse_steps - 1:
            ratio = sparse_ratio

        schedule[t_global] = ratio

    return schedule

# TODO: check if the model is SQALoraModel
class SparseCallbackBase(transformers.TrainerCallback):
    def __init__(
        self,
        model,
        sparse_ratio: float = 0.5,
        sparse_warmup: float = 0.1,
        sparse_end: float = 0.3,
        sparse_steps: int = 2,
        sparse_prune_largest: bool = False,
        SQAT = False,
    ):
        self.model = model
        self.sparse_ratio = sparse_ratio
        self.sparse_warmup = sparse_warmup
        self.sparse_end = sparse_end
        self.sparse_steps = sparse_steps
        self.sparse_prune_largest = sparse_prune_largest
        self.SQAT = SQAT
        self.sparse_schedule = {}

    def on_train_begin(self, args, state, control, **kwargs):
        max_steps = state.max_steps
        self.sparse_schedule = build_sparsity_schedule(
            max_steps=max_steps,
            sparse_warmup=self.sparse_warmup,
            sparse_end=self.sparse_end,
            sparse_ratio=self.sparse_ratio,
            sparse_steps=self.sparse_steps
        )
        print_rank_0(f"sparse schedule created as : {self.sparse_schedule}")
        if self.SQAT:
            print_rank_0("Sparse-Quantization-Aware Training is triggered, on step 0 the unquantized model will be sparsed, then it will be quantized the whole training session.")
            sparsity4step0 = self.sparse_schedule.pop(min(self.sparse_schedule.keys()))
            self.model.prune(sparsity_ratio = sparsity4step0, sparse_prune_largest = self.sparse_prune_largest)
            self.model.quantize()

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self.sparse_schedule:
            self.model.prune(sparsity_ratio = self.sparse_schedule[step], sparse_prune_largest = self.sparse_prune_largest)
            print_rank_0("sparsity", self.model.calculate_sparsity())
