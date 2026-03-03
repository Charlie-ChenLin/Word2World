# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import gc
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.agent_trainer.ppo import core_algos
from verl.workers.agent_actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)
        self.project_env_feedback_grad_to_pg = bool(self.config.get('project_env_feedback_grad_to_pg', False))
        self.env_feedback_grad_proj_eps = float(self.config.get('env_feedback_grad_proj_eps', 1e-12))
        self.env_feedback_grad_proj_min_free_gb = float(self.config.get('env_feedback_grad_proj_min_free_gb', 0.0))
        self.env_feedback_grad_proj_oom_fallback = bool(self.config.get('env_feedback_grad_proj_oom_fallback', True))
        self.env_feedback_grad_proj_legacy_compat_metrics = bool(
            self.config.get('env_feedback_grad_proj_legacy_compat_metrics', False)
        )
        self._env_feedback_grad_proj_legacy_compat_runtime_disabled = False
        self.env_feedback_grad_proj_impl = str(
            self.config.get('env_feedback_grad_proj_impl', 'reordered_exact')
        ).strip().lower()
        valid_proj_impls = {'legacy_cpu_offload', 'reordered_exact'}
        if self.env_feedback_grad_proj_impl not in valid_proj_impls:
            print(
                f"[actor] invalid env_feedback_grad_proj_impl={self.env_feedback_grad_proj_impl}, "
                "fallback to reordered_exact"
            )
            self.env_feedback_grad_proj_impl = 'reordered_exact'
        self.env_feedback_grad_proj_target = str(
            self.config.get('env_feedback_grad_proj_target', 'pg')
        ).strip().lower()
        valid_proj_targets = {'pg', 'pg_plus_other'}
        if self.env_feedback_grad_proj_target not in valid_proj_targets:
            print(
                f"[actor] invalid env_feedback_grad_proj_target={self.env_feedback_grad_proj_target}, "
                "fallback to pg"
            )
            self.env_feedback_grad_proj_target = 'pg'
        self.env_feedback_grad_proj_algo = str(
            self.config.get('env_feedback_grad_proj_algo', 'alpha_scale')
        ).strip().lower()
        valid_proj_algos = {'alpha_scale', 'pcgrad'}
        if self.env_feedback_grad_proj_algo not in valid_proj_algos:
            print(
                f"[actor] invalid env_feedback_grad_proj_algo={self.env_feedback_grad_proj_algo}, "
                "fallback to alpha_scale"
            )
            self.env_feedback_grad_proj_algo = 'alpha_scale'
        self.env_feedback_pcgrad_lambda_max = float(
            self.config.get('env_feedback_pcgrad_lambda_max', 0.2)
        )
        self.env_feedback_pcgrad_lambda_norm_ratio = float(
            self.config.get('env_feedback_pcgrad_lambda_norm_ratio', 0.3)
        )
        self.env_feedback_pcgrad_eps = float(
            self.config.get('env_feedback_pcgrad_eps', 1e-12)
        )
        self.env_feedback_pcgrad_log_extra_metrics = bool(
            self.config.get('env_feedback_pcgrad_log_extra_metrics', True)
        )
        if (
            self.env_feedback_grad_proj_algo == 'pcgrad'
            and self.env_feedback_grad_proj_impl != 'reordered_exact'
        ):
            print(
                f"[actor] pcgrad currently supports env_feedback_grad_proj_impl='reordered_exact' only, "
                f"got {self.env_feedback_grad_proj_impl}; fallback to reordered_exact"
            )
            self.env_feedback_grad_proj_impl = 'reordered_exact'
        if (
            self.env_feedback_grad_proj_algo == 'pcgrad'
            and self.env_feedback_grad_proj_target != 'pg'
        ):
            print(
                f"[actor] pcgrad currently supports env_feedback_grad_proj_target='pg' only, "
                f"got {self.env_feedback_grad_proj_target}; fallback to pg"
            )
            self.env_feedback_grad_proj_target = 'pg'
        self._actor_params = [param for param in self.actor_module.parameters() if param.requires_grad]
        print(
            f"Actor project_env_feedback_grad_to_pg={self.project_env_feedback_grad_to_pg} "
            f"env_feedback_grad_proj_eps={self.env_feedback_grad_proj_eps} "
            f"env_feedback_grad_proj_min_free_gb={self.env_feedback_grad_proj_min_free_gb} "
            f"env_feedback_grad_proj_oom_fallback={self.env_feedback_grad_proj_oom_fallback} "
            f"env_feedback_grad_proj_legacy_compat_metrics={self.env_feedback_grad_proj_legacy_compat_metrics} "
            f"env_feedback_grad_proj_impl={self.env_feedback_grad_proj_impl} "
            f"env_feedback_grad_proj_target={self.env_feedback_grad_proj_target} "
            f"env_feedback_grad_proj_algo={self.env_feedback_grad_proj_algo} "
            f"env_feedback_pcgrad_lambda_max={self.env_feedback_pcgrad_lambda_max} "
            f"env_feedback_pcgrad_lambda_norm_ratio={self.env_feedback_pcgrad_lambda_norm_ratio} "
                f"env_feedback_pcgrad_eps={self.env_feedback_pcgrad_eps} "
                f"env_feedback_pcgrad_log_extra_metrics={self.env_feedback_pcgrad_log_extra_metrics}"
        )
        self._pcgrad_lambda_max_tensor_cache = {}

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        compute_entropy: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                if compute_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                else:
                    entropy_rmpad = torch.zeros(logits_rmpad.size(0), device=logits_rmpad.device, dtype=logits_rmpad.dtype)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                if compute_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                else:
                    entropy = torch.zeros_like(log_probs)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def _get_pcgrad_lambda_max_tensor(self, device: torch.device) -> torch.Tensor:
        cache_key = (device.type, device.index)
        cached = self._pcgrad_lambda_max_tensor_cache.get(cache_key)
        if cached is None:
            cached = torch.tensor(
                self.env_feedback_pcgrad_lambda_max,
                device=device,
                dtype=torch.float32,
            )
            self._pcgrad_lambda_max_tensor_cache[cache_key] = cached
        return cached

    def _compute_grad_norm_sq(self, device: torch.device) -> torch.Tensor:
        grad_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
        for param in self._actor_params:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            grad_norm_sq.add_(torch.sum(grad * grad))
        return grad_norm_sq

    def _copy_current_grads_to_cpu(self, clear_grads: bool = False):
        offloaded_grads = {}
        offloaded_bytes = 0
        for param_idx, param in enumerate(self._actor_params):
            grad = param.grad
            if grad is None:
                continue
            grad_cpu = grad.detach().to(device='cpu', copy=True)
            offloaded_grads[param_idx] = grad_cpu
            offloaded_bytes += grad_cpu.numel() * grad_cpu.element_size()
            if clear_grads:
                param.grad = None
        return offloaded_grads, offloaded_bytes

    def _offload_current_grads_to_cpu(self):
        return self._copy_current_grads_to_cpu(clear_grads=True)

    def _clear_current_grads(self):
        for param in self._actor_params:
            param.grad = None

    @staticmethod
    def _is_oom_error(err: BaseException) -> bool:
        return 'out of memory' in str(err).lower()

    def _handle_microbatch_oom(self):
        self._clear_current_grads()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _accumulate_offloaded_grads_from_cpu(self, offloaded_grads, scale: float = 1.0):
        scale_value = float(scale)
        for param_idx, grad_cpu in offloaded_grads.items():
            param = self._actor_params[param_idx]
            grad_device = param.grad.device if param.grad is not None else param.device
            # Grad dtype must match the destination grad tensor (or param dtype when grad is None),
            # otherwise PyTorch raises on assignment under mixed precision / FSDP.
            grad_dtype = param.grad.dtype if param.grad is not None else param.dtype
            grad_restore = grad_cpu.to(device=grad_device, dtype=grad_dtype)
            target_numel = param.grad.numel() if param.grad is not None else param.numel()
            if grad_restore.numel() != target_numel:
                # FSDP hook-captured grads can be full flattened tensors while param.grad holds
                # the local shard. Map the full tensor to this rank's contiguous shard.
                if dist.is_available() and dist.is_initialized() and grad_restore.numel() % target_numel == 0:
                    shard_count = grad_restore.numel() // target_numel
                    shard_rank = dist.get_rank() % shard_count
                    start = shard_rank * target_numel
                    grad_restore = grad_restore.reshape(-1).narrow(0, start, target_numel)
                    if param.grad is not None:
                        grad_restore = grad_restore.reshape_as(param.grad)
                    else:
                        grad_restore = grad_restore.reshape_as(param)
                else:
                    raise RuntimeError(
                        f"Grad shape mismatch for param idx {param_idx}: "
                        f"{tuple(grad_restore.shape)} vs target numel {target_numel}"
                    )
            if scale_value != 1.0:
                grad_restore.mul_(scale_value)
            if param.grad is None:
                param.grad = grad_restore
            else:
                param.grad.add_(grad_restore)

    def _align_env_pg_for_dot(
        self,
        pg_grad: torch.Tensor,
        env_grad: torch.Tensor,
        param_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pg_flat = pg_grad.reshape(-1)
        env_flat = env_grad.reshape(-1)
        if pg_flat.numel() == env_flat.numel():
            return env_grad, pg_grad.reshape_as(env_grad)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if pg_flat.numel() > env_flat.numel() and pg_flat.numel() % env_flat.numel() == 0:
            shard_count = pg_flat.numel() // env_flat.numel()
            shard_rank = rank % shard_count
            start = shard_rank * env_flat.numel()
            pg_shard = pg_flat.narrow(0, start, env_flat.numel()).reshape_as(env_grad)
            return env_grad, pg_shard

        if env_flat.numel() > pg_flat.numel() and env_flat.numel() % pg_flat.numel() == 0:
            shard_count = env_flat.numel() // pg_flat.numel()
            shard_rank = rank % shard_count
            start = shard_rank * pg_flat.numel()
            env_shard = env_flat.narrow(0, start, pg_flat.numel()).reshape_as(pg_grad)
            return env_shard, pg_grad

        raise RuntimeError(
            f"PG/env grad shape mismatch for param idx {param_idx}: "
            f"{tuple(pg_grad.shape)} vs {tuple(env_grad.shape)}"
        )

    def _project_env_grad_to_pg(
        self,
        env_feedback_loss_scaled: torch.Tensor,
        pg_grad_norm_sq: torch.Tensor,
        retain_graph: bool,
        pg_grads_cpu: dict = None,
    ):
        dot = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        env_grad_norm_sq = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        pg_grad_present_param_count = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        pg_grad_missing_param_count = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        pg_grad_present_elem_count = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        pg_grad_missing_elem_count = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        handles = []

        def _make_hook(param_idx: int, param):
            def _hook(env_grad):
                if env_grad is None:
                    return env_grad
                env_grad_fp32 = env_grad.detach().float()
                pg_grad_ref = None
                pg_grad = param.grad
                if pg_grad is not None:
                    pg_grad_ref = pg_grad.detach().float()
                if pg_grad_ref is None and pg_grads_cpu is not None:
                    pg_grad_cpu = pg_grads_cpu.get(param_idx)
                    if pg_grad_cpu is not None:
                        pg_grad_ref = pg_grad_cpu.to(device=env_grad.device, dtype=torch.float32)
                if pg_grad_ref is not None:
                    env_for_dot, pg_for_dot = self._align_env_pg_for_dot(
                        pg_grad=pg_grad_ref,
                        env_grad=env_grad_fp32,
                        param_idx=param_idx,
                    )
                    env_grad_norm_sq.add_(torch.sum(env_for_dot * env_for_dot))
                    pg_grad_present_param_count.add_(1.0)
                    pg_grad_present_elem_count.add_(float(env_for_dot.numel()))
                    dot.add_(torch.sum(env_for_dot * pg_for_dot))
                else:
                    env_grad_norm_sq.add_(torch.sum(env_grad_fp32 * env_grad_fp32))
                    pg_grad_missing_param_count.add_(1.0)
                    pg_grad_missing_elem_count.add_(float(env_grad_fp32.numel()))
                env_grad.zero_()
                return env_grad

            return _hook

        for param_idx, param in enumerate(self._actor_params):
            handles.append(param.register_hook(_make_hook(param_idx, param)))

        try:
            env_feedback_loss_scaled.backward(retain_graph=retain_graph)
        finally:
            for handle in handles:
                handle.remove()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(dot, op=dist.ReduceOp.SUM)
            dist.all_reduce(pg_grad_norm_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(env_grad_norm_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(pg_grad_present_param_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(pg_grad_missing_param_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(pg_grad_present_elem_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(pg_grad_missing_elem_count, op=dist.ReduceOp.SUM)

        pg_grad_param_total = (pg_grad_present_param_count + pg_grad_missing_param_count).clamp_min(1.0)
        pg_grad_elem_total = (pg_grad_present_elem_count + pg_grad_missing_elem_count).clamp_min(1.0)
        pg_grad_present_param_ratio = pg_grad_present_param_count / pg_grad_param_total
        pg_grad_present_elem_ratio = pg_grad_present_elem_count / pg_grad_elem_total

        cosine_before = dot / torch.sqrt(
            pg_grad_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
            * env_grad_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
        )

        if pg_grad_norm_sq.item() <= self.env_feedback_grad_proj_eps:
            alpha = torch.zeros_like(dot)
            cosine_after = torch.zeros_like(dot)
            aligned_after = 0.0
            return (
                alpha,
                dot,
                pg_grad_norm_sq,
                env_grad_norm_sq,
                cosine_before,
                cosine_after,
                aligned_after,
                0.0,
                pg_grad_present_param_ratio,
                pg_grad_present_elem_ratio,
            )

        raw_alpha = dot / pg_grad_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
        # Alpha-clamp variant on this branch: only allow non-negative projection scaling.
        alpha = torch.clamp_min(raw_alpha, 0.0)
        for param in self._actor_params:
            if param.grad is None:
                continue
            param.grad.mul_((1.0 + alpha).to(dtype=param.grad.dtype))

        if alpha.item() > self.env_feedback_grad_proj_eps:
            cosine_after = torch.ones_like(alpha)
            aligned_after = 1.0
        elif alpha.item() < -self.env_feedback_grad_proj_eps:
            cosine_after = -torch.ones_like(alpha)
            aligned_after = 0.0
        else:
            cosine_after = torch.zeros_like(alpha)
            aligned_after = 0.0

        return (
            alpha,
            dot,
            pg_grad_norm_sq,
            env_grad_norm_sq,
            cosine_before,
            cosine_after,
            aligned_after,
            1.0,
            pg_grad_present_param_ratio,
            pg_grad_present_elem_ratio,
        )

    def _compute_env_proj_stats_from_pg_cpu(
        self,
        env_feedback_loss_scaled: torch.Tensor,
        pg_grads_cpu: dict,
        pg_grad_norm_sq: torch.Tensor,
        retain_graph: bool,
    ):
        dot = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        env_grad_norm_sq = torch.zeros((), device=env_feedback_loss_scaled.device, dtype=torch.float32)
        handles = []

        def _make_hook(param_idx: int):
            def _hook(env_grad):
                if env_grad is None:
                    return env_grad
                env_grad_fp32 = env_grad.detach().float()
                pg_grad_cpu = pg_grads_cpu.get(param_idx)
                if pg_grad_cpu is not None:
                    env_for_dot, pg_for_dot = self._align_env_pg_for_dot(
                        pg_grad=pg_grad_cpu.to(device=env_grad.device, dtype=torch.float32),
                        env_grad=env_grad_fp32,
                        param_idx=param_idx,
                    )
                    env_grad_norm_sq.add_(torch.sum(env_for_dot * env_for_dot))
                    dot.add_(torch.sum(env_for_dot * pg_for_dot))
                else:
                    env_grad_norm_sq.add_(torch.sum(env_grad_fp32 * env_grad_fp32))
                env_grad.zero_()
                return env_grad

            return _hook

        for param_idx, param in enumerate(self._actor_params):
            handles.append(param.register_hook(_make_hook(param_idx)))

        try:
            env_feedback_loss_scaled.backward(retain_graph=retain_graph)
        finally:
            for handle in handles:
                handle.remove()

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(dot, op=dist.ReduceOp.SUM)
            dist.all_reduce(pg_grad_norm_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(env_grad_norm_sq, op=dist.ReduceOp.SUM)

        cosine_before = dot / torch.sqrt(
            pg_grad_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
            * env_grad_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
        )

        if pg_grad_norm_sq.item() <= self.env_feedback_grad_proj_eps:
            alpha = torch.zeros_like(dot)
            cosine_after = torch.zeros_like(dot)
            aligned_after = 0.0
            return alpha, dot, pg_grad_norm_sq, env_grad_norm_sq, cosine_before, cosine_after, aligned_after, 0.0

        raw_alpha = dot / pg_grad_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
        alpha = torch.clamp_min(raw_alpha, 0.0)
        if alpha.item() > self.env_feedback_grad_proj_eps:
            cosine_after = torch.ones_like(alpha)
            aligned_after = 1.0
        elif alpha.item() < -self.env_feedback_grad_proj_eps:
            cosine_after = -torch.ones_like(alpha)
            aligned_after = 0.0
        else:
            cosine_after = torch.zeros_like(alpha)
            aligned_after = 0.0

        return alpha, dot, pg_grad_norm_sq, env_grad_norm_sq, cosine_before, cosine_after, aligned_after, 1.0

    def _capture_loss_grads_to_cpu_without_accum(
        self,
        loss_scaled: torch.Tensor,
        retain_graph: bool,
    ):
        # Capture fresh grads from this loss only.
        self._clear_current_grads()
        loss_scaled.backward(retain_graph=retain_graph)

        # Snapshot to CPU and clear in-model grads to avoid accidental accumulation.
        grads_cpu, offloaded_bytes = self._copy_current_grads_to_cpu(clear_grads=True)
        grad_norm_sq_cpu = 0.0
        for grad_cpu in grads_cpu.values():
            grad_fp32 = grad_cpu.float()
            grad_norm_sq_cpu += torch.sum(grad_fp32 * grad_fp32).item()
        grad_norm_sq = torch.tensor(grad_norm_sq_cpu, device=loss_scaled.device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(grad_norm_sq, op=dist.ReduceOp.SUM)

        return grads_cpu, offloaded_bytes, grad_norm_sq

    def _capture_pg_grads_to_cpu_without_accum(
        self,
        pg_loss_scaled: torch.Tensor,
        retain_graph: bool,
    ):
        return self._capture_loss_grads_to_cpu_without_accum(
            loss_scaled=pg_loss_scaled,
            retain_graph=retain_graph,
        )

    def _compute_grad_dict_pair_stats(
        self,
        base_grads_cpu: dict,
        ref_grads_cpu: dict,
        device: torch.device,
    ):
        # Keep pairwise stats on CPU to avoid per-parameter CPU->GPU copies.
        dot_cpu = 0.0
        base_norm_sq_cpu = 0.0
        ref_norm_sq_cpu = 0.0
        base_grad_present_param_count_cpu = 0.0
        base_grad_missing_param_count_cpu = 0.0
        base_grad_present_elem_count_cpu = 0.0
        base_grad_missing_elem_count_cpu = 0.0

        all_param_indices = set(base_grads_cpu.keys()) | set(ref_grads_cpu.keys())
        for param_idx in all_param_indices:
            base_grad_cpu = base_grads_cpu.get(param_idx)
            ref_grad_cpu = ref_grads_cpu.get(param_idx)
            base_grad_fp32 = (
                base_grad_cpu.float()
                if base_grad_cpu is not None
                else None
            )
            ref_grad_fp32 = (
                ref_grad_cpu.float()
                if ref_grad_cpu is not None
                else None
            )

            if base_grad_fp32 is not None and ref_grad_fp32 is not None:
                ref_for_dot, base_for_dot = self._align_env_pg_for_dot(
                    pg_grad=base_grad_fp32,
                    env_grad=ref_grad_fp32,
                    param_idx=param_idx,
                )
                dot_cpu += torch.sum(base_for_dot * ref_for_dot).item()
                base_norm_sq_cpu += torch.sum(base_for_dot * base_for_dot).item()
                ref_norm_sq_cpu += torch.sum(ref_for_dot * ref_for_dot).item()
                base_grad_present_param_count_cpu += 1.0
                base_grad_present_elem_count_cpu += float(base_for_dot.numel())
            elif base_grad_fp32 is not None:
                base_norm_sq_cpu += torch.sum(base_grad_fp32 * base_grad_fp32).item()
                base_grad_missing_param_count_cpu += 1.0
                base_grad_missing_elem_count_cpu += float(base_grad_fp32.numel())
            elif ref_grad_fp32 is not None:
                ref_norm_sq_cpu += torch.sum(ref_grad_fp32 * ref_grad_fp32).item()
                base_grad_missing_param_count_cpu += 1.0
                base_grad_missing_elem_count_cpu += float(ref_grad_fp32.numel())

        stats = torch.tensor(
            [
                dot_cpu,
                base_norm_sq_cpu,
                ref_norm_sq_cpu,
                base_grad_present_param_count_cpu,
                base_grad_missing_param_count_cpu,
                base_grad_present_elem_count_cpu,
                base_grad_missing_elem_count_cpu,
            ],
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized():
            # Pack all scalar stats into one collective to reduce synchronization overhead.
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        dot = stats[0]
        base_norm_sq = stats[1]
        ref_norm_sq = stats[2]
        base_grad_present_param_count = stats[3]
        base_grad_missing_param_count = stats[4]
        base_grad_present_elem_count = stats[5]
        base_grad_missing_elem_count = stats[6]

        base_grad_param_total = (base_grad_present_param_count + base_grad_missing_param_count).clamp_min(1.0)
        base_grad_elem_total = (base_grad_present_elem_count + base_grad_missing_elem_count).clamp_min(1.0)
        base_grad_present_param_ratio = base_grad_present_param_count / base_grad_param_total
        base_grad_present_elem_ratio = base_grad_present_elem_count / base_grad_elem_total

        return (
            dot,
            base_norm_sq,
            ref_norm_sq,
            base_grad_present_param_ratio,
            base_grad_present_elem_ratio,
        )

    def _compute_dot_between_cpu_and_live_grads(
        self,
        base_grads_cpu: dict,
        device: torch.device,
    ) -> torch.Tensor:
        dot_cpu = 0.0
        for param_idx, base_grad_cpu in base_grads_cpu.items():
            if base_grad_cpu is None:
                continue
            live_grad = self._actor_params[param_idx].grad
            if live_grad is None:
                continue
            base_grad_fp32 = base_grad_cpu.float()
            live_grad_fp32 = live_grad.detach().float().cpu()
            live_for_dot, base_for_dot = self._align_env_pg_for_dot(
                pg_grad=base_grad_fp32,
                env_grad=live_grad_fp32,
                param_idx=param_idx,
            )
            dot_cpu += torch.sum(base_for_dot * live_for_dot).item()

        dot = torch.tensor(dot_cpu, device=device, dtype=torch.float32)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(dot, op=dist.ReduceOp.SUM)
        return dot

    def compute_log_prob(self, data: DataProto, train_mode: bool = False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        prev_training = self.actor_module.training
        self.actor_module.train(mode=train_mode)
        try:
            micro_batch_size = data.meta_info['micro_batch_size']
            temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
            use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

            select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
            batch = data.select(batch_keys=select_keys).batch

            if use_dynamic_bsz:
                # split using dynamic bsz
                max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            else:
                micro_batches = batch.split(micro_batch_size)

            log_probs_lst = []
            for micro_batch in micro_batches:
                with torch.no_grad():
                    _, log_probs = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        compute_entropy=False,
                    )
                log_probs_lst.append(log_probs)
            log_probs = torch.concat(log_probs_lst, dim=0)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                log_probs = log_probs[revert_indices]

            return log_probs
        finally:
            self.actor_module.train(mode=prev_training)

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        env_feedback_coef = data.meta_info.get(
            'env_feedback_loss_coef',
            self.config.get('env_feedback_loss_coef', 1.0),
        )

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'responses', 'response_mask', 'env_feedback_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()
            proj_runtime_disabled = False
            entropy_runtime_disabled = False

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                response_mask = data['response_mask']
                env_feedback_mask = data['env_feedback_mask']
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']
                # One-time sanity log to confirm env_feedback_mask arrives and aligns with response_mask.
                if not getattr(self, "_logged_env_feedback_mask", False):
                    print(
                        "[actor] env_feedback_mask received:",
                        f"shape={tuple(env_feedback_mask.shape)}",
                        f"sum={env_feedback_mask.sum().item()}",
                        f"response_sum={response_mask.sum().item()}",
                    )
                    self._logged_env_feedback_mask = True

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff
                entropy_coeff_effective = 0.0 if entropy_runtime_disabled else entropy_coeff
                entropy_term_enabled = float(entropy_coeff_effective) != 0.0

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    scale_factor = len(data) / self.config.ppo_mini_batch_size
                else:
                    scale_factor = 1.0 / self.gradient_accumulation

                ref_log_prob = data['ref_log_prob'] if self.config.use_kl_loss else None
                env_token_count = env_feedback_mask.sum()
                device = response_mask.device

                # Initialize values so logging stays consistent across both branches.
                entropy_loss = torch.zeros((), device=device)
                pg_loss = torch.zeros((), device=device)
                pg_clipfrac = torch.zeros((), device=device)
                ppo_kl = torch.zeros((), device=device)
                env_feedback_loss = torch.zeros((), device=device)
                env_feedback_loss_weighted = torch.zeros((), device=device)
                policy_loss = torch.zeros((), device=device)
                loss = torch.zeros((), device=device)

                env_feedback_proj_alpha = torch.zeros((), device=device)
                env_feedback_proj_dot = torch.zeros((), device=device)
                env_feedback_proj_pg_norm_sq = torch.zeros((), device=device)
                env_feedback_proj_env_norm_sq = torch.zeros((), device=device)
                env_feedback_proj_cosine_before = torch.zeros((), device=device)
                env_feedback_proj_cosine_after = torch.zeros((), device=device)
                env_feedback_proj_alpha_abs = torch.zeros((), device=device)
                env_feedback_proj_dot_abs = torch.zeros((), device=device)
                env_feedback_proj_alpha_legacy_compat = torch.zeros((), device=device)
                env_feedback_proj_dot_legacy_compat = torch.zeros((), device=device)
                env_feedback_proj_pg_norm_sq_legacy_compat = torch.zeros((), device=device)
                env_feedback_proj_env_norm_sq_legacy_compat = torch.zeros((), device=device)
                env_feedback_proj_cosine_before_legacy_compat = torch.zeros((), device=device)
                env_feedback_proj_pg_grad_present_param_ratio = torch.zeros((), device=device)
                env_feedback_proj_pg_grad_present_elem_ratio = torch.zeros((), device=device)
                env_feedback_proj_aligned_after = 0.0
                env_feedback_proj_pg_grad_offloaded = 0.0
                env_feedback_proj_pg_grad_offload_mb = 0.0
                env_feedback_proj_applied = 0.0
                env_feedback_proj_skipped_low_mem = 0.0
                env_feedback_proj_oom_fallback = 0.0
                env_feedback_proj_legacy_compat_enabled = 0.0
                env_feedback_proj_legacy_compat_valid = 0.0
                env_feedback_proj_alpha_pos_count = 0.0
                env_feedback_proj_alpha_valid_count = 0.0
                env_feedback_pcgrad_dot_pg_env_before = torch.zeros((), device=device)
                env_feedback_pcgrad_cos_pg_env_before = torch.zeros((), device=device)
                env_feedback_pcgrad_dot_pg_env_after_conflict_fix = torch.zeros((), device=device)
                env_feedback_pcgrad_lambda = torch.zeros((), device=device)
                env_feedback_pcgrad_norm_pg = torch.zeros((), device=device)
                env_feedback_pcgrad_norm_other = torch.zeros((), device=device)
                env_feedback_pcgrad_norm_env = torch.zeros((), device=device)
                env_feedback_pcgrad_norm_env_pos = torch.zeros((), device=device)
                env_feedback_pcgrad_norm_final = torch.zeros((), device=device)
                env_feedback_pcgrad_final_vs_pg_cosine = torch.zeros((), device=device)
                env_feedback_pcgrad_conflict_applied = 0.0
                env_feedback_pcgrad_env_inject_ratio = torch.zeros((), device=device)
                microbatch_oom_skipped = 0.0
                entropy_runtime_disabled_metric = 1.0 if entropy_runtime_disabled else 0.0
                env_feedback_proj_impl_id = 0.0 if self.env_feedback_grad_proj_impl == 'legacy_cpu_offload' else 1.0
                env_feedback_proj_target_id = 0.0 if self.env_feedback_grad_proj_target == 'pg' else 1.0
                env_feedback_proj_algo_id = 0.0 if self.env_feedback_grad_proj_algo == 'alpha_scale' else 1.0
                use_env_feedback_grad_proj = (
                    self.project_env_feedback_grad_to_pg
                    and (not proj_runtime_disabled)
                    and env_token_count.item() > 0
                    and abs(float(env_feedback_coef)) > self.env_feedback_grad_proj_eps
                )
                if use_env_feedback_grad_proj and torch.cuda.is_available() and self.env_feedback_grad_proj_min_free_gb > 0.0:
                    free_bytes, _ = torch.cuda.mem_get_info(device=device)
                    min_free_bytes = self.env_feedback_grad_proj_min_free_gb * 1024.0 * 1024.0 * 1024.0
                    if free_bytes < min_free_bytes:
                        use_env_feedback_grad_proj = False
                        env_feedback_proj_skipped_low_mem = 1.0

                if use_env_feedback_grad_proj:
                    if self.env_feedback_grad_proj_impl == 'legacy_cpu_offload':
                        offloaded_accum_grads, _ = self._offload_current_grads_to_cpu()
                        try:
                            # Legacy exact projection path: offload PG grads to CPU, then use
                            # hook-collected env stats to compute alpha and add correction.
                            entropy_other, log_prob_other = self._forward_micro_batch(
                                micro_batch=data,
                                temperature=temperature,
                                compute_entropy=entropy_term_enabled,
                            )
                            pg_loss_other, _, _ = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                 log_prob=log_prob_other,
                                                                                 advantages=advantages,
                                                                                 eos_mask=response_mask,
                                                                                 cliprange=clip_ratio)
                            entropy_loss = verl_F.masked_mean(entropy_other.detach(), response_mask)
                            policy_no_env_loss_recomputed = pg_loss_other
                            if entropy_term_enabled:
                                policy_no_env_loss_recomputed = policy_no_env_loss_recomputed - entropy_loss * entropy_coeff_effective
                            if self.config.use_kl_loss:
                                kld_other = core_algos.kl_penalty(logprob=log_prob_other,
                                                                  ref_logprob=ref_log_prob,
                                                                  kl_penalty=self.config.kl_loss_type)
                                kl_loss_other = masked_mean(kld_other, response_mask)
                                policy_no_env_loss_recomputed = policy_no_env_loss_recomputed + kl_loss_other * self.config.kl_loss_coef
                            policy_no_env_loss_scaled = policy_no_env_loss_recomputed * scale_factor
                            policy_no_env_loss_scaled.backward()

                            entropy, log_prob = self._forward_micro_batch(
                                micro_batch=data,
                                temperature=temperature,
                                compute_entropy=False,
                            )
                            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                         log_prob=log_prob,
                                                                                         advantages=advantages,
                                                                                         eos_mask=response_mask,
                                                                                         cliprange=clip_ratio)
                            policy_loss = pg_loss.detach() - entropy_loss * entropy_coeff_effective

                            env_feedback_loss = -verl_F.masked_mean(log_prob, env_feedback_mask).detach()
                            env_feedback_loss_weighted = env_feedback_loss * env_feedback_coef
                            policy_loss = policy_loss + env_feedback_loss_weighted

                            if self.config.use_kl_loss:
                                kld = core_algos.kl_penalty(logprob=log_prob,
                                                            ref_logprob=ref_log_prob,
                                                            kl_penalty=self.config.kl_loss_type)
                                kl_loss = masked_mean(kld, response_mask).detach()
                                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                                metrics['actor/kl_loss'] = kl_loss.item()
                                metrics['actor/kl_coef'] = self.config.kl_loss_coef

                            loss = policy_loss * scale_factor
                            pg_loss_scaled = pg_loss * scale_factor
                            (
                                offloaded_pg_grads,
                                offloaded_bytes,
                                env_feedback_proj_pg_norm_sq,
                            ) = self._capture_pg_grads_to_cpu_without_accum(
                                pg_loss_scaled=pg_loss_scaled,
                                retain_graph=False,
                            )
                            env_feedback_proj_pg_grad_offloaded = 1.0 if len(offloaded_pg_grads) > 0 else 0.0
                            env_feedback_proj_pg_grad_offload_mb = offloaded_bytes / (1024.0 * 1024.0)

                            _, log_prob_env = self._forward_micro_batch(
                                micro_batch=data,
                                temperature=temperature,
                                compute_entropy=False,
                            )
                            env_feedback_loss_proj = -verl_F.masked_mean(log_prob_env, env_feedback_mask)
                            env_feedback_loss_weighted_scaled = env_feedback_loss_proj * env_feedback_coef * scale_factor
                            (
                                env_feedback_proj_alpha,
                                env_feedback_proj_dot,
                                env_feedback_proj_pg_norm_sq,
                                env_feedback_proj_env_norm_sq,
                                env_feedback_proj_cosine_before,
                                env_feedback_proj_cosine_after,
                                env_feedback_proj_aligned_after,
                                env_feedback_proj_applied,
                            ) = self._compute_env_proj_stats_from_pg_cpu(
                                env_feedback_loss_scaled=env_feedback_loss_weighted_scaled,
                                pg_grads_cpu=offloaded_pg_grads,
                                pg_grad_norm_sq=env_feedback_proj_pg_norm_sq,
                                retain_graph=False,
                            )

                            self._accumulate_offloaded_grads_from_cpu(
                                offloaded_grads=offloaded_pg_grads,
                                scale=env_feedback_proj_alpha.detach().item(),
                            )
                        except RuntimeError as runtime_err:
                            is_oom = self._is_oom_error(runtime_err)
                            if not (self.env_feedback_grad_proj_oom_fallback and is_oom):
                                raise

                            env_feedback_proj_oom_fallback = 1.0
                            proj_runtime_disabled = True
                            entropy_runtime_disabled = True
                            entropy_runtime_disabled_metric = 1.0
                            microbatch_oom_skipped = 1.0
                            self._handle_microbatch_oom()
                        finally:
                            if len(offloaded_accum_grads) > 0:
                                self._accumulate_offloaded_grads_from_cpu(offloaded_grads=offloaded_accum_grads, scale=1.0)
                    else:
                        offloaded_accum_grads, _ = self._offload_current_grads_to_cpu()
                        try:
                            if self.env_feedback_grad_proj_target == 'pg_plus_other':
                                # Optional projection target:
                                #   base_grad = g_pg + g_other
                                #   final_grad = (1 + alpha) * base_grad
                                # This path avoids splitting PG/other backward passes.
                                entropy_base, log_prob_base = self._forward_micro_batch(
                                    micro_batch=data,
                                    temperature=temperature,
                                    compute_entropy=entropy_term_enabled,
                                )
                                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob_base,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=clip_ratio,
                                )
                                if entropy_term_enabled:
                                    entropy_loss = verl_F.masked_mean(entropy_base, response_mask)
                                else:
                                    entropy_loss = torch.zeros((), device=device)

                                if env_token_count.item() > 0:
                                    env_feedback_loss = -verl_F.masked_mean(log_prob_base, env_feedback_mask).detach()
                                else:
                                    env_feedback_loss = torch.zeros((), device=device)
                                env_feedback_loss_weighted = env_feedback_loss * env_feedback_coef

                                policy_no_env_loss = pg_loss - entropy_loss * entropy_coeff_effective
                                kl_loss = torch.zeros((), device=device)
                                if self.config.use_kl_loss:
                                    kld_base = core_algos.kl_penalty(
                                        logprob=log_prob_base,
                                        ref_logprob=ref_log_prob,
                                        kl_penalty=self.config.kl_loss_type,
                                    )
                                    kl_loss = masked_mean(kld_base, response_mask)
                                    policy_no_env_loss = policy_no_env_loss + kl_loss * self.config.kl_loss_coef
                                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                                policy_no_env_loss_scaled = policy_no_env_loss * scale_factor
                                policy_no_env_loss_scaled.backward()
                                env_feedback_proj_pg_norm_sq = self._compute_grad_norm_sq(device=device)
                                pg_grads_cpu_for_proj, pg_offloaded_bytes_for_proj = self._copy_current_grads_to_cpu(
                                    clear_grads=False
                                )
                                if len(pg_grads_cpu_for_proj) > 0:
                                    env_feedback_proj_pg_grad_offloaded = 1.0
                                    env_feedback_proj_pg_grad_offload_mb = (
                                        pg_offloaded_bytes_for_proj / (1024.0 * 1024.0)
                                    )

                                _, log_prob_env = self._forward_micro_batch(
                                    micro_batch=data,
                                    temperature=temperature,
                                    compute_entropy=False,
                                )
                                env_feedback_loss_proj = -verl_F.masked_mean(log_prob_env, env_feedback_mask)
                                env_feedback_loss_weighted_scaled = env_feedback_loss_proj * env_feedback_coef * scale_factor
                                (
                                    env_feedback_proj_alpha,
                                    env_feedback_proj_dot,
                                    env_feedback_proj_pg_norm_sq,
                                    env_feedback_proj_env_norm_sq,
                                    env_feedback_proj_cosine_before,
                                    env_feedback_proj_cosine_after,
                                    env_feedback_proj_aligned_after,
                                    env_feedback_proj_applied,
                                    env_feedback_proj_pg_grad_present_param_ratio,
                                    env_feedback_proj_pg_grad_present_elem_ratio,
                                ) = self._project_env_grad_to_pg(
                                    env_feedback_loss_scaled=env_feedback_loss_weighted_scaled,
                                    pg_grad_norm_sq=env_feedback_proj_pg_norm_sq,
                                    retain_graph=False,
                                    pg_grads_cpu=pg_grads_cpu_for_proj,
                                )
                                env_feedback_proj_alpha_abs = torch.abs(env_feedback_proj_alpha.detach())
                                env_feedback_proj_dot_abs = torch.abs(env_feedback_proj_dot.detach())

                                policy_loss = policy_no_env_loss.detach() + env_feedback_loss_weighted
                                loss = policy_loss * scale_factor
                            else:
                                # Reordered strict-exact projection path:
                                #   alpha_scale: historical scalar projection scaling
                                #   pcgrad: PG-anchored conflict resolution + bounded aligned fusion
                                _, log_prob_pg = self._forward_micro_batch(
                                    micro_batch=data,
                                    temperature=temperature,
                                    compute_entropy=False,
                                )
                                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                             log_prob=log_prob_pg,
                                                                                             advantages=advantages,
                                                                                             eos_mask=response_mask,
                                                                                             cliprange=clip_ratio)
                                env_feedback_loss = -verl_F.masked_mean(log_prob_pg, env_feedback_mask).detach()
                                env_feedback_loss_weighted = env_feedback_loss * env_feedback_coef

                                legacy_compat_metrics_enabled = (
                                    self.env_feedback_grad_proj_legacy_compat_metrics
                                    and (not self._env_feedback_grad_proj_legacy_compat_runtime_disabled)
                                )
                                env_feedback_proj_legacy_compat_enabled = 1.0 if legacy_compat_metrics_enabled else 0.0
                                if legacy_compat_metrics_enabled:
                                    try:
                                        _, log_prob_pg_legacy = self._forward_micro_batch(
                                            micro_batch=data,
                                            temperature=temperature,
                                            compute_entropy=False,
                                        )
                                        pg_loss_legacy, _, _ = core_algos.compute_policy_loss(
                                            old_log_prob=old_log_prob,
                                            log_prob=log_prob_pg_legacy,
                                            advantages=advantages,
                                            eos_mask=response_mask,
                                            cliprange=clip_ratio,
                                        )
                                        pg_loss_legacy_scaled = pg_loss_legacy * scale_factor
                                        (
                                            offloaded_pg_grads_legacy,
                                            _,
                                            env_feedback_proj_pg_norm_sq_legacy_compat,
                                        ) = self._capture_pg_grads_to_cpu_without_accum(
                                            pg_loss_scaled=pg_loss_legacy_scaled,
                                            retain_graph=False,
                                        )
                                        _, log_prob_env_legacy = self._forward_micro_batch(
                                            micro_batch=data,
                                            temperature=temperature,
                                            compute_entropy=False,
                                        )
                                        env_feedback_loss_proj_legacy = -verl_F.masked_mean(log_prob_env_legacy, env_feedback_mask)
                                        env_feedback_loss_weighted_scaled_legacy = (
                                            env_feedback_loss_proj_legacy * env_feedback_coef * scale_factor
                                        )
                                        (
                                            env_feedback_proj_alpha_legacy_compat,
                                            env_feedback_proj_dot_legacy_compat,
                                            env_feedback_proj_pg_norm_sq_legacy_compat,
                                            env_feedback_proj_env_norm_sq_legacy_compat,
                                            env_feedback_proj_cosine_before_legacy_compat,
                                            _,
                                            _,
                                            env_feedback_proj_legacy_compat_valid,
                                        ) = self._compute_env_proj_stats_from_pg_cpu(
                                            env_feedback_loss_scaled=env_feedback_loss_weighted_scaled_legacy,
                                            pg_grads_cpu=offloaded_pg_grads_legacy,
                                            pg_grad_norm_sq=env_feedback_proj_pg_norm_sq_legacy_compat,
                                            retain_graph=False,
                                        )
                                        self._clear_current_grads()
                                    except RuntimeError as runtime_err:
                                        if not self._is_oom_error(runtime_err):
                                            raise
                                        self._env_feedback_grad_proj_legacy_compat_runtime_disabled = True
                                        print("[actor] disable env_feedback_grad_proj_legacy_compat_metrics due to OOM")
                                        self._handle_microbatch_oom()
                                if self.env_feedback_grad_proj_algo == 'pcgrad':
                                    pg_loss_scaled = pg_loss * scale_factor
                                    (
                                        pg_grads_cpu_for_proj,
                                        pg_offloaded_bytes_for_proj,
                                        env_feedback_proj_pg_norm_sq,
                                    ) = self._capture_loss_grads_to_cpu_without_accum(
                                        loss_scaled=pg_loss_scaled,
                                        retain_graph=False,
                                    )
                                    if len(pg_grads_cpu_for_proj) > 0:
                                        env_feedback_proj_pg_grad_offloaded = 1.0
                                        env_feedback_proj_pg_grad_offload_mb = (
                                            pg_offloaded_bytes_for_proj / (1024.0 * 1024.0)
                                        )

                                    _, log_prob_env = self._forward_micro_batch(
                                        micro_batch=data,
                                        temperature=temperature,
                                        compute_entropy=False,
                                    )
                                    env_feedback_loss_proj = -verl_F.masked_mean(log_prob_env, env_feedback_mask)
                                    env_feedback_loss_weighted_scaled = env_feedback_loss_proj * env_feedback_coef * scale_factor
                                    (
                                        env_grads_cpu_for_proj,
                                        _,
                                        env_feedback_proj_env_norm_sq,
                                    ) = self._capture_loss_grads_to_cpu_without_accum(
                                        loss_scaled=env_feedback_loss_weighted_scaled,
                                        retain_graph=False,
                                    )
                                    (
                                        env_feedback_proj_dot,
                                        env_feedback_proj_pg_norm_sq,
                                        env_feedback_proj_env_norm_sq,
                                        env_feedback_proj_pg_grad_present_param_ratio,
                                        env_feedback_proj_pg_grad_present_elem_ratio,
                                    ) = self._compute_grad_dict_pair_stats(
                                        base_grads_cpu=pg_grads_cpu_for_proj,
                                        ref_grads_cpu=env_grads_cpu_for_proj,
                                        device=device,
                                    )
                                    env_feedback_proj_cosine_before = env_feedback_proj_dot / torch.sqrt(
                                        env_feedback_proj_pg_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
                                        * env_feedback_proj_env_norm_sq.clamp_min(self.env_feedback_grad_proj_eps)
                                    )
                                    env_feedback_proj_alpha = torch.clamp_min(
                                        env_feedback_proj_dot / env_feedback_proj_pg_norm_sq.clamp_min(self.env_feedback_grad_proj_eps),
                                        0.0,
                                    )
                                    env_feedback_proj_alpha_abs = torch.abs(env_feedback_proj_alpha.detach())
                                    env_feedback_proj_dot_abs = torch.abs(env_feedback_proj_dot.detach())
                                    env_feedback_proj_applied = 1.0
                                    if env_feedback_proj_alpha.item() > self.env_feedback_grad_proj_eps:
                                        env_feedback_proj_cosine_after = torch.ones_like(env_feedback_proj_alpha)
                                        env_feedback_proj_aligned_after = 1.0
                                    elif env_feedback_proj_alpha.item() < -self.env_feedback_grad_proj_eps:
                                        env_feedback_proj_cosine_after = -torch.ones_like(env_feedback_proj_alpha)
                                        env_feedback_proj_aligned_after = 0.0
                                    else:
                                        env_feedback_proj_cosine_after = torch.zeros_like(env_feedback_proj_alpha)
                                        env_feedback_proj_aligned_after = 0.0

                                    kl_loss = torch.zeros((), device=device)
                                    other_policy_loss = torch.zeros((), device=device)
                                    other_norm_sq = torch.zeros((), device=device, dtype=torch.float32)
                                    need_other_policy = entropy_term_enabled or (
                                        self.config.use_kl_loss and float(self.config.kl_loss_coef) != 0.0
                                    )
                                    if need_other_policy:
                                        entropy_other, log_prob_other = self._forward_micro_batch(
                                            micro_batch=data,
                                            temperature=temperature,
                                            compute_entropy=entropy_term_enabled,
                                        )
                                        if entropy_term_enabled:
                                            entropy_loss = verl_F.masked_mean(entropy_other, response_mask)
                                            other_policy_loss = other_policy_loss - entropy_loss * entropy_coeff_effective
                                        if self.config.use_kl_loss:
                                            kld_other = core_algos.kl_penalty(logprob=log_prob_other,
                                                                              ref_logprob=ref_log_prob,
                                                                              kl_penalty=self.config.kl_loss_type)
                                            kl_loss = masked_mean(kld_other, response_mask)
                                            other_policy_loss = other_policy_loss + kl_loss * self.config.kl_loss_coef
                                        (other_policy_loss * scale_factor).backward()
                                        other_norm_sq = self._compute_grad_norm_sq(device=device)
                                        if dist.is_available() and dist.is_initialized():
                                            dist.all_reduce(other_norm_sq, op=dist.ReduceOp.SUM)

                                    if self.config.use_kl_loss:
                                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                                    dot_pg_other = self._compute_dot_between_cpu_and_live_grads(
                                        base_grads_cpu=pg_grads_cpu_for_proj,
                                        device=device,
                                    )
                                    pcgrad_eps = max(self.env_feedback_pcgrad_eps, self.env_feedback_grad_proj_eps)
                                    dot_pg_env = env_feedback_proj_dot
                                    neg_dot_pg_env = torch.clamp_min(-dot_pg_env, 0.0)
                                    pos_dot_pg_env = torch.clamp_min(dot_pg_env, 0.0)
                                    conflict_scale = neg_dot_pg_env / env_feedback_proj_env_norm_sq.clamp_min(pcgrad_eps)
                                    env_pos_scale = pos_dot_pg_env / env_feedback_proj_pg_norm_sq.clamp_min(pcgrad_eps)
                                    env_pos_norm_sq = env_pos_scale * env_pos_scale * env_feedback_proj_pg_norm_sq
                                    pg_norm = torch.sqrt(env_feedback_proj_pg_norm_sq.clamp_min(0.0))
                                    env_norm = torch.sqrt(env_feedback_proj_env_norm_sq.clamp_min(0.0))
                                    env_pos_norm = torch.sqrt(env_pos_norm_sq.clamp_min(0.0))
                                    lambda_from_norm = (
                                        self.env_feedback_pcgrad_lambda_norm_ratio
                                        * pg_norm
                                        / env_pos_norm.clamp_min(pcgrad_eps)
                                    )
                                    env_feedback_pcgrad_lambda = torch.minimum(
                                        self._get_pcgrad_lambda_max_tensor(device=device),
                                        lambda_from_norm,
                                    )
                                    invalid_env_injection = (
                                        (pos_dot_pg_env <= 0.0)
                                        | (env_pos_norm <= pcgrad_eps)
                                        | (pg_norm <= pcgrad_eps)
                                    )
                                    if bool(invalid_env_injection.item()):
                                        env_feedback_pcgrad_lambda = torch.zeros_like(env_feedback_pcgrad_lambda)

                                    inject_pg_scale = env_feedback_pcgrad_lambda * env_pos_scale
                                    pg_scale = torch.ones_like(inject_pg_scale) + inject_pg_scale

                                    if not need_other_policy:
                                        self._clear_current_grads()
                                    if len(pg_grads_cpu_for_proj) > 0:
                                        self._accumulate_offloaded_grads_from_cpu(
                                            offloaded_grads=pg_grads_cpu_for_proj,
                                            scale=float(pg_scale.detach().item()),
                                        )
                                    if conflict_scale.item() > 0.0 and len(env_grads_cpu_for_proj) > 0:
                                        self._accumulate_offloaded_grads_from_cpu(
                                            offloaded_grads=env_grads_cpu_for_proj,
                                            scale=float(conflict_scale.detach().item()),
                                        )

                                    final_norm_sq = self._compute_grad_norm_sq(device=device)
                                    if dist.is_available() and dist.is_initialized():
                                        dist.all_reduce(final_norm_sq, op=dist.ReduceOp.SUM)
                                    dot_pg_env_after = dot_pg_env + conflict_scale * env_feedback_proj_env_norm_sq
                                    dot_final_pg = (
                                        pg_scale * env_feedback_proj_pg_norm_sq
                                        + dot_pg_other
                                        + conflict_scale * dot_pg_env
                                    )
                                    env_feedback_pcgrad_final_vs_pg_cosine = dot_final_pg / torch.sqrt(
                                        final_norm_sq.clamp_min(pcgrad_eps)
                                        * env_feedback_proj_pg_norm_sq.clamp_min(pcgrad_eps)
                                    )
                                    inject_norm_sq = inject_pg_scale * inject_pg_scale * env_feedback_proj_pg_norm_sq
                                    env_feedback_pcgrad_env_inject_ratio = torch.sqrt(
                                        inject_norm_sq.clamp_min(0.0)
                                    ) / pg_norm.clamp_min(pcgrad_eps)

                                    env_feedback_pcgrad_dot_pg_env_before = dot_pg_env
                                    env_feedback_pcgrad_cos_pg_env_before = env_feedback_proj_cosine_before
                                    env_feedback_pcgrad_dot_pg_env_after_conflict_fix = dot_pg_env_after
                                    env_feedback_pcgrad_norm_pg = pg_norm
                                    env_feedback_pcgrad_norm_other = torch.sqrt(other_norm_sq.clamp_min(0.0))
                                    env_feedback_pcgrad_norm_env = env_norm
                                    env_feedback_pcgrad_norm_env_pos = env_pos_norm
                                    env_feedback_pcgrad_norm_final = torch.sqrt(final_norm_sq.clamp_min(0.0))
                                    env_feedback_pcgrad_conflict_applied = (
                                        1.0 if neg_dot_pg_env.item() > pcgrad_eps else 0.0
                                    )
                                    if not self.env_feedback_pcgrad_log_extra_metrics:
                                        env_feedback_pcgrad_norm_other = torch.zeros_like(env_feedback_pcgrad_norm_other)

                                    policy_loss = pg_loss.detach() - entropy_loss.detach() * entropy_coeff_effective
                                    policy_loss = policy_loss + env_feedback_loss_weighted
                                    if self.config.use_kl_loss:
                                        policy_loss = policy_loss + kl_loss.detach() * self.config.kl_loss_coef
                                    loss = policy_loss * scale_factor
                                else:
                                    pg_loss_scaled = pg_loss * scale_factor
                                    pg_loss_scaled.backward()
                                    env_feedback_proj_pg_norm_sq = self._compute_grad_norm_sq(device=device)
                                    # FSDP (use_orig_params=False) may not expose pg grads via param.grad inside
                                    # env backward hooks. Snapshot PG grads to CPU for robust dot computation.
                                    pg_grads_cpu_for_proj, pg_offloaded_bytes_for_proj = self._copy_current_grads_to_cpu(
                                        clear_grads=False
                                    )
                                    if len(pg_grads_cpu_for_proj) > 0:
                                        env_feedback_proj_pg_grad_offloaded = 1.0
                                        env_feedback_proj_pg_grad_offload_mb = (
                                            pg_offloaded_bytes_for_proj / (1024.0 * 1024.0)
                                        )

                                    _, log_prob_env = self._forward_micro_batch(
                                        micro_batch=data,
                                        temperature=temperature,
                                        compute_entropy=False,
                                    )
                                    env_feedback_loss_proj = -verl_F.masked_mean(log_prob_env, env_feedback_mask)
                                    env_feedback_loss_weighted_scaled = env_feedback_loss_proj * env_feedback_coef * scale_factor
                                    (
                                        env_feedback_proj_alpha,
                                        env_feedback_proj_dot,
                                        env_feedback_proj_pg_norm_sq,
                                        env_feedback_proj_env_norm_sq,
                                        env_feedback_proj_cosine_before,
                                        env_feedback_proj_cosine_after,
                                        env_feedback_proj_aligned_after,
                                        env_feedback_proj_applied,
                                        env_feedback_proj_pg_grad_present_param_ratio,
                                        env_feedback_proj_pg_grad_present_elem_ratio,
                                    ) = self._project_env_grad_to_pg(
                                        env_feedback_loss_scaled=env_feedback_loss_weighted_scaled,
                                        pg_grad_norm_sq=env_feedback_proj_pg_norm_sq,
                                        retain_graph=False,
                                        pg_grads_cpu=pg_grads_cpu_for_proj,
                                    )
                                    env_feedback_proj_alpha_abs = torch.abs(env_feedback_proj_alpha.detach())
                                    env_feedback_proj_dot_abs = torch.abs(env_feedback_proj_dot.detach())

                                    kl_loss = torch.zeros((), device=device)
                                    other_policy_loss = torch.zeros((), device=device)
                                    need_other_policy = entropy_term_enabled or (
                                        self.config.use_kl_loss and float(self.config.kl_loss_coef) != 0.0
                                    )
                                    if need_other_policy:
                                        entropy_other, log_prob_other = self._forward_micro_batch(
                                            micro_batch=data,
                                            temperature=temperature,
                                            compute_entropy=entropy_term_enabled,
                                        )
                                        if entropy_term_enabled:
                                            entropy_loss = verl_F.masked_mean(entropy_other, response_mask)
                                            other_policy_loss = other_policy_loss - entropy_loss * entropy_coeff_effective
                                        if self.config.use_kl_loss:
                                            kld_other = core_algos.kl_penalty(logprob=log_prob_other,
                                                                              ref_logprob=ref_log_prob,
                                                                              kl_penalty=self.config.kl_loss_type)
                                            kl_loss = masked_mean(kld_other, response_mask)
                                            other_policy_loss = other_policy_loss + kl_loss * self.config.kl_loss_coef
                                        (other_policy_loss * scale_factor).backward()

                                    if self.config.use_kl_loss:
                                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                                    policy_loss = pg_loss.detach() - entropy_loss.detach() * entropy_coeff_effective
                                    policy_loss = policy_loss + env_feedback_loss_weighted
                                    if self.config.use_kl_loss:
                                        policy_loss = policy_loss + kl_loss.detach() * self.config.kl_loss_coef
                                    loss = policy_loss * scale_factor
                        except RuntimeError as runtime_err:
                            is_oom = self._is_oom_error(runtime_err)
                            if not (self.env_feedback_grad_proj_oom_fallback and is_oom):
                                raise

                            env_feedback_proj_oom_fallback = 1.0
                            proj_runtime_disabled = True
                            entropy_runtime_disabled = True
                            entropy_runtime_disabled_metric = 1.0
                            microbatch_oom_skipped = 1.0
                            self._handle_microbatch_oom()
                        finally:
                            if len(offloaded_accum_grads) > 0:
                                self._accumulate_offloaded_grads_from_cpu(offloaded_grads=offloaded_accum_grads, scale=1.0)
                else:
                    offloaded_accum_grads, _ = self._offload_current_grads_to_cpu()
                    try:
                        # Standard path: single forward/backward.
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            compute_entropy=entropy_term_enabled,
                        )
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                     log_prob=log_prob,
                                                                                     advantages=advantages,
                                                                                     eos_mask=response_mask,
                                                                                     cliprange=clip_ratio)
                        entropy_loss = verl_F.masked_mean(entropy, response_mask)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff_effective

                        if env_token_count.item() > 0:
                            env_feedback_loss = -verl_F.masked_mean(log_prob, env_feedback_mask)
                        else:
                            env_feedback_loss = torch.zeros((), device=device)
                        env_feedback_loss_weighted = env_feedback_loss * env_feedback_coef
                        policy_loss = policy_loss + env_feedback_loss_weighted

                        if self.config.use_kl_loss:
                            kld = core_algos.kl_penalty(logprob=log_prob,
                                                        ref_logprob=ref_log_prob,
                                                        kl_penalty=self.config.kl_loss_type)
                            kl_loss = masked_mean(kld, response_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                            metrics['actor/kl_coef'] = self.config.kl_loss_coef

                        loss = policy_loss * scale_factor
                        loss.backward()
                    except RuntimeError as runtime_err:
                        if not self._is_oom_error(runtime_err):
                            raise
                        proj_runtime_disabled = True
                        entropy_runtime_disabled = True
                        entropy_runtime_disabled_metric = 1.0
                        microbatch_oom_skipped = 1.0
                        self._handle_microbatch_oom()
                    finally:
                        if len(offloaded_accum_grads) > 0:
                            self._accumulate_offloaded_grads_from_cpu(offloaded_grads=offloaded_accum_grads, scale=1.0)

                env_feedback_proj_alpha_abs = torch.abs(env_feedback_proj_alpha.detach())
                env_feedback_proj_dot_abs = torch.abs(env_feedback_proj_dot.detach())
                if env_feedback_proj_applied > 0.0:
                    env_feedback_proj_alpha_valid_count = 1.0
                    env_feedback_proj_alpha_pos_count = (
                        1.0 if env_feedback_proj_alpha.detach().item() > self.env_feedback_grad_proj_eps else 0.0
                    )

                # Micro-batch metrics for logging; scaled_* reflects the actual backward scaling.
                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                    'actor/env_feedback_loss': env_feedback_loss.detach().item(),
                    'actor/env_feedback_loss_weighted': env_feedback_loss_weighted.detach().item(),
                    'actor/env_feedback_tokens': env_token_count.detach().item(),
                    'actor/env_feedback_coef': env_feedback_coef,
                    'actor/policy_loss': policy_loss.detach().item(),
                    'actor/loss': loss.detach().item(),
                    'actor/pg_loss_scaled': (pg_loss * scale_factor).detach().item(),
                    'actor/env_feedback_loss_scaled': (env_feedback_loss * scale_factor).detach().item(),
                    'actor/env_feedback_loss_weighted_scaled': (env_feedback_loss_weighted * scale_factor).detach().item(),
                    'actor/env_feedback_proj_enabled': float(self.project_env_feedback_grad_to_pg),
                    'actor/env_feedback_proj_applied': env_feedback_proj_applied,
                    'actor/env_feedback_proj_alpha': env_feedback_proj_alpha.detach().item(), # averaged across all micro-batches
                    'actor/env_feedback_proj_alpha_abs': env_feedback_proj_alpha_abs.detach().item(), # averaged acrosss all micro-batches
                    'actor/env_feedback_proj_dot': env_feedback_proj_dot.detach().item(),
                    'actor/env_feedback_proj_dot_abs': env_feedback_proj_dot_abs.detach().item(),
                    'actor/env_feedback_proj_pg_norm_sq': env_feedback_proj_pg_norm_sq.detach().item(),
                    'actor/env_feedback_proj_env_norm_sq': env_feedback_proj_env_norm_sq.detach().item(),
                    'actor/env_feedback_proj_cosine_before': env_feedback_proj_cosine_before.detach().item(),
                    'actor/env_feedback_proj_cosine_after': env_feedback_proj_cosine_after.detach().item(),
                    'actor/env_feedback_proj_pg_grad_present_param_ratio': env_feedback_proj_pg_grad_present_param_ratio.detach().item(),
                    'actor/env_feedback_proj_pg_grad_present_elem_ratio': env_feedback_proj_pg_grad_present_elem_ratio.detach().item(),
                    'actor/env_feedback_proj_alpha_legacy_compat': env_feedback_proj_alpha_legacy_compat.detach().item(),
                    'actor/env_feedback_proj_dot_legacy_compat': env_feedback_proj_dot_legacy_compat.detach().item(),
                    'actor/env_feedback_proj_pg_norm_sq_legacy_compat': env_feedback_proj_pg_norm_sq_legacy_compat.detach().item(),
                    'actor/env_feedback_proj_env_norm_sq_legacy_compat': env_feedback_proj_env_norm_sq_legacy_compat.detach().item(),
                    'actor/env_feedback_proj_cosine_before_legacy_compat': env_feedback_proj_cosine_before_legacy_compat.detach().item(),
                    'actor/env_feedback_proj_legacy_compat_enabled': env_feedback_proj_legacy_compat_enabled,
                    'actor/env_feedback_proj_legacy_compat_valid': env_feedback_proj_legacy_compat_valid,
                    'actor/env_feedback_proj_alpha_pos_count': env_feedback_proj_alpha_pos_count,
                    'actor/env_feedback_proj_alpha_valid_count': env_feedback_proj_alpha_valid_count,
                    'actor/env_feedback_proj_aligned_after': env_feedback_proj_aligned_after,
                    'actor/env_feedback_proj_pg_grad_offloaded': env_feedback_proj_pg_grad_offloaded,
                    'actor/env_feedback_proj_pg_grad_offload_mb': env_feedback_proj_pg_grad_offload_mb,
                    'actor/env_feedback_proj_impl': env_feedback_proj_impl_id,
                    'actor/env_feedback_proj_target': env_feedback_proj_target_id,
                    'actor/env_feedback_proj_algo': env_feedback_proj_algo_id,
                    'actor/env_feedback_proj_skipped_low_mem': env_feedback_proj_skipped_low_mem,
                    'actor/env_feedback_proj_oom_fallback': env_feedback_proj_oom_fallback,
                    'actor/env_feedback_pcgrad_dot_pg_env_before': env_feedback_pcgrad_dot_pg_env_before.detach().item(),
                    'actor/env_feedback_pcgrad_cos_pg_env_before': env_feedback_pcgrad_cos_pg_env_before.detach().item(),
                    'actor/env_feedback_pcgrad_dot_pg_env_after_conflict_fix': (
                        env_feedback_pcgrad_dot_pg_env_after_conflict_fix.detach().item()
                    ),
                    'actor/env_feedback_pcgrad_lambda': env_feedback_pcgrad_lambda.detach().item(),
                    'actor/env_feedback_pcgrad_norm_pg': env_feedback_pcgrad_norm_pg.detach().item(),
                    'actor/env_feedback_pcgrad_norm_other': env_feedback_pcgrad_norm_other.detach().item(),
                    'actor/env_feedback_pcgrad_norm_env': env_feedback_pcgrad_norm_env.detach().item(),
                    'actor/env_feedback_pcgrad_norm_env_pos': env_feedback_pcgrad_norm_env_pos.detach().item(),
                    'actor/env_feedback_pcgrad_norm_final': env_feedback_pcgrad_norm_final.detach().item(),
                    'actor/env_feedback_pcgrad_final_vs_pg_cosine': env_feedback_pcgrad_final_vs_pg_cosine.detach().item(),
                    'actor/env_feedback_pcgrad_conflict_applied': env_feedback_pcgrad_conflict_applied,
                    'actor/env_feedback_pcgrad_env_inject_ratio': env_feedback_pcgrad_env_inject_ratio.detach().item(),
                    'actor/microbatch_oom_skipped': microbatch_oom_skipped,
                    'actor/entropy_runtime_disabled': entropy_runtime_disabled_metric,
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        # Step-level aggregates (mean over micro-batches) for logging.
        step_metric_map = {
            'actor/policy_loss': 'actor_step/policy_loss',
            'actor/loss': 'actor_step/loss',
            'actor/pg_loss': 'actor_step/pg_loss',
            'actor/env_feedback_loss': 'actor_step/env_feedback_loss',
            'actor/env_feedback_loss_weighted': 'actor_step/env_feedback_loss_weighted',
            'actor/pg_loss_scaled': 'actor_step/pg_loss_scaled',
            'actor/env_feedback_loss_scaled': 'actor_step/env_feedback_loss_scaled',
            'actor/env_feedback_loss_weighted_scaled': 'actor_step/env_feedback_loss_weighted_scaled',
            'actor/env_feedback_proj_applied': 'actor_step/env_feedback_proj_applied',
            'actor/env_feedback_proj_alpha': 'actor_step/env_feedback_proj_alpha',
            'actor/env_feedback_proj_alpha_abs': 'actor_step/env_feedback_proj_alpha_abs',
            'actor/env_feedback_proj_dot': 'actor_step/env_feedback_proj_dot',
            'actor/env_feedback_proj_dot_abs': 'actor_step/env_feedback_proj_dot_abs',
            'actor/env_feedback_proj_pg_norm_sq': 'actor_step/env_feedback_proj_pg_norm_sq',
            'actor/env_feedback_proj_env_norm_sq': 'actor_step/env_feedback_proj_env_norm_sq',
            'actor/env_feedback_proj_cosine_before': 'actor_step/env_feedback_proj_cosine_before',
            'actor/env_feedback_proj_cosine_after': 'actor_step/env_feedback_proj_cosine_after',
            'actor/env_feedback_proj_pg_grad_present_param_ratio': 'actor_step/env_feedback_proj_pg_grad_present_param_ratio',
            'actor/env_feedback_proj_pg_grad_present_elem_ratio': 'actor_step/env_feedback_proj_pg_grad_present_elem_ratio',
            'actor/env_feedback_proj_alpha_legacy_compat': 'actor_step/env_feedback_proj_alpha_legacy_compat',
            'actor/env_feedback_proj_dot_legacy_compat': 'actor_step/env_feedback_proj_dot_legacy_compat',
            'actor/env_feedback_proj_pg_norm_sq_legacy_compat': 'actor_step/env_feedback_proj_pg_norm_sq_legacy_compat',
            'actor/env_feedback_proj_env_norm_sq_legacy_compat': 'actor_step/env_feedback_proj_env_norm_sq_legacy_compat',
            'actor/env_feedback_proj_cosine_before_legacy_compat': 'actor_step/env_feedback_proj_cosine_before_legacy_compat',
            'actor/env_feedback_proj_legacy_compat_enabled': 'actor_step/env_feedback_proj_legacy_compat_enabled',
            'actor/env_feedback_proj_legacy_compat_valid': 'actor_step/env_feedback_proj_legacy_compat_valid',
            'actor/env_feedback_proj_alpha_pos_count': 'actor_step/env_feedback_proj_alpha_pos_count',
            'actor/env_feedback_proj_alpha_valid_count': 'actor_step/env_feedback_proj_alpha_valid_count',
            'actor/env_feedback_proj_aligned_after': 'actor_step/env_feedback_proj_aligned_after',
            'actor/env_feedback_proj_pg_grad_offloaded': 'actor_step/env_feedback_proj_pg_grad_offloaded',
            'actor/env_feedback_proj_pg_grad_offload_mb': 'actor_step/env_feedback_proj_pg_grad_offload_mb',
            'actor/env_feedback_proj_impl': 'actor_step/env_feedback_proj_impl',
            'actor/env_feedback_proj_target': 'actor_step/env_feedback_proj_target',
            'actor/env_feedback_proj_algo': 'actor_step/env_feedback_proj_algo',
            'actor/env_feedback_proj_skipped_low_mem': 'actor_step/env_feedback_proj_skipped_low_mem',
            'actor/env_feedback_proj_oom_fallback': 'actor_step/env_feedback_proj_oom_fallback',
            'actor/env_feedback_pcgrad_dot_pg_env_before': 'actor_step/env_feedback_pcgrad_dot_pg_env_before',
            'actor/env_feedback_pcgrad_cos_pg_env_before': 'actor_step/env_feedback_pcgrad_cos_pg_env_before',
            'actor/env_feedback_pcgrad_dot_pg_env_after_conflict_fix': (
                'actor_step/env_feedback_pcgrad_dot_pg_env_after_conflict_fix'
            ),
            'actor/env_feedback_pcgrad_lambda': 'actor_step/env_feedback_pcgrad_lambda',
            'actor/env_feedback_pcgrad_norm_pg': 'actor_step/env_feedback_pcgrad_norm_pg',
            'actor/env_feedback_pcgrad_norm_other': 'actor_step/env_feedback_pcgrad_norm_other',
            'actor/env_feedback_pcgrad_norm_env': 'actor_step/env_feedback_pcgrad_norm_env',
            'actor/env_feedback_pcgrad_norm_env_pos': 'actor_step/env_feedback_pcgrad_norm_env_pos',
            'actor/env_feedback_pcgrad_norm_final': 'actor_step/env_feedback_pcgrad_norm_final',
            'actor/env_feedback_pcgrad_final_vs_pg_cosine': 'actor_step/env_feedback_pcgrad_final_vs_pg_cosine',
            'actor/env_feedback_pcgrad_conflict_applied': 'actor_step/env_feedback_pcgrad_conflict_applied',
            'actor/env_feedback_pcgrad_env_inject_ratio': 'actor_step/env_feedback_pcgrad_env_inject_ratio',
            'actor/microbatch_oom_skipped': 'actor_step/microbatch_oom_skipped',
            'actor/entropy_runtime_disabled': 'actor_step/entropy_runtime_disabled',
        }
        for src_key, dst_key in step_metric_map.items():
            vals = metrics.get(src_key)
            if isinstance(vals, list) and len(vals) > 0:
                metrics[dst_key] = float(sum(vals) / len(vals))

        def _add_step_distribution_stats(src_key: str, dst_prefix: str):
            vals = metrics.get(src_key)
            if not (isinstance(vals, list) and len(vals) > 0):
                return
            vals_tensor = torch.tensor(vals, dtype=torch.float32)
            metrics[f'{dst_prefix}_max'] = float(vals_tensor.max().item())
            metrics[f'{dst_prefix}_min'] = float(vals_tensor.min().item())
            metrics[f'{dst_prefix}_std'] = float(vals_tensor.std(unbiased=False).item())

        _add_step_distribution_stats('actor/env_feedback_proj_alpha', 'actor_step/env_feedback_proj_alpha')
        _add_step_distribution_stats('actor/env_feedback_proj_cosine_before', 'actor_step/env_feedback_proj_cosine_before')
        _add_step_distribution_stats('actor/env_feedback_proj_cosine_after', 'actor_step/env_feedback_proj_cosine_after')
        _add_step_distribution_stats('actor/env_feedback_pcgrad_lambda', 'actor_step/env_feedback_pcgrad_lambda')
        _add_step_distribution_stats(
            'actor/env_feedback_pcgrad_final_vs_pg_cosine',
            'actor_step/env_feedback_pcgrad_final_vs_pg_cosine',
        )
        _add_step_distribution_stats(
            'actor/env_feedback_pcgrad_env_inject_ratio',
            'actor_step/env_feedback_pcgrad_env_inject_ratio',
        )

        cosine_before_vals = metrics.get('actor/env_feedback_proj_cosine_before')
        if isinstance(cosine_before_vals, list) and len(cosine_before_vals) > 0:
            cosine_before_tensor = torch.tensor(cosine_before_vals, dtype=torch.float32)
            cosine_before_pos_mask = cosine_before_tensor > 0
            cosine_before_neg_mask = cosine_before_tensor < 0
            metrics['actor_step/env_feedback_proj_cosine_before_pos_ratio'] = float(
                cosine_before_pos_mask.float().mean().item()
            )
            metrics['actor_step/env_feedback_proj_cosine_before_pos_mean'] = (
                float(cosine_before_tensor[cosine_before_pos_mask].mean().item())
                if bool(cosine_before_pos_mask.any())
                else 0.0
            )
            metrics['actor_step/env_feedback_proj_cosine_before_neg_mean'] = (
                float(cosine_before_tensor[cosine_before_neg_mask].mean().item())
                if bool(cosine_before_neg_mask.any())
                else 0.0
            )

        alpha_pos_vals = metrics.get('actor/env_feedback_proj_alpha_pos_count')
        alpha_valid_vals = metrics.get('actor/env_feedback_proj_alpha_valid_count')
        if (
            isinstance(alpha_pos_vals, list)
            and isinstance(alpha_valid_vals, list)
            and len(alpha_pos_vals) > 0
            and len(alpha_valid_vals) > 0
        ):
            alpha_valid_total = float(sum(alpha_valid_vals))
            if alpha_valid_total > 0.0:
                metrics['actor_step/env_feedback_proj_alpha_pos_ratio'] = float(sum(alpha_pos_vals)) / alpha_valid_total
            else:
                metrics['actor_step/env_feedback_proj_alpha_pos_ratio'] = 0.0
        return metrics
