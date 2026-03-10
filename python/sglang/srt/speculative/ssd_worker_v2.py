import contextlib
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.ssd_tree_cache import SSDTreeCache
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context

logger = logging.getLogger(__name__)


def _get_plan_stream(device: str):
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class SSDDraftWorker:
    """
    SSD Draft Worker - 处理Draft模型的运行和树状缓存
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        
        self.ssd_async = server_args.ssd_async
        self.ssd_fan_out = server_args.ssd_fan_out or 4
        self.ssd_jit_speculate = server_args.ssd_jit_speculate
        self.ssd_sampler_x = server_args.ssd_sampler_x or 0.6
        self.ssd_max_cached_sequences = server_args.ssd_max_cached_sequences or 1024
        
        logger.info(f"SSDDraftWorker initialized with:")
        logger.info(f"  - ssd_async: {self.ssd_async}")
        logger.info(f"  - ssd_fan_out: {self.ssd_fan_out}")
        
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        
        server_args.context_length = target_worker.model_runner.model_config.context_len
        
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
        self._eagle_draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )
        
        self._eagle_draft_worker.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        
        vocab_size = self.target_worker.model_runner.model_config.vocab_size
        hidden_size = getattr(
            self.target_worker.model_runner.model_config.hf_config,
            "hidden_size",
            None,
        )
        
        self.ssd_tree_cache = SSDTreeCache(
            device=self.device,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_cached_sequences=self.ssd_max_cached_sequences,
        )
        
        logger.info(f"SSD tree cache initialized with vocab_size={vocab_size}")
        
        self._init_fan_out_buffers()
        
        self.draft_tp_context = self._eagle_draft_worker.draft_tp_context
        self.draft_runner = self._eagle_draft_worker.draft_runner

    def _init_fan_out_buffers(self):
        """初始化分支因子相关的预分配buffer"""
        K = self.speculative_num_steps
        F = self.ssd_fan_out
        
        self._fan_idx_hit = torch.arange(K + 1, device=self.device, dtype=torch.int64).repeat_interleave(F)
        self._fan_idx_miss = torch.arange(K + 1, device=self.device, dtype=torch.int64).repeat_interleave(F)
        self._arange_kp1 = torch.arange(K + 1, device=self.device, dtype=torch.int64)
        self._arange_f = torch.arange(F, device=self.device, dtype=torch.int64)
        
        logger.info(f"SSD fan-out buffers initialized with K={K}, F={F}")

    def draft(self, model_worker_batch):
        """
        SSD的Draft阶段，带树状缓存
        """
        return self._eagle_draft_worker.draft(model_worker_batch)

    def _draft_extend_for_prefill(self, *args, **kwargs):
        return self._eagle_draft_worker._draft_extend_for_prefill(*args, **kwargs)

    def _draft_extend_for_decode(self, *args, **kwargs):
        return self._eagle_draft_worker._draft_extend_for_decode(*args, **kwargs)


class SSDWorkerV2(BaseSpecWorker):
    """
    SSD Worker V2 - Overlap模式，实现Draft和Verify的并行
    这是SSD的第三个核心创新：异步模式
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        
        self.ssd_async = server_args.ssd_async
        self.ssd_fan_out = server_args.ssd_fan_out or 4
        self.ssd_jit_speculate = server_args.ssd_jit_speculate
        self.ssd_sampler_x = server_args.ssd_sampler_x or 0.6
        
        logger.info(f"SSDWorkerV2 initialized with:")
        logger.info(f"  - ssd_async: {self.ssd_async}")
        logger.info(f"  - ssd_fan_out: {self.ssd_fan_out}")
        logger.info(f"  - overlap mode: enabled (SSDWorkerV2)")

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = SSDDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        self._draft_worker.ssd_tree_cache.clear()
        logger.info("SSD tree cache cleared")

    def forward_batch_generation(self, model_worker_batch):
        """
        SSD的前向生成，支持Overlap模式（Draft和Verify并行）
        """
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
        
        eagle_worker_v2 = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        eagle_worker_v2.__dict__.update(self.__dict__)
        eagle_worker_v2._eagle_draft_worker = self._draft_worker._eagle_draft_worker
        
        return eagle_worker_v2.forward_batch_generation(model_worker_batch)

    def verify(self, batch):
        """
        SSD的Verify阶段
        """
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
        
        eagle_worker_v2 = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        eagle_worker_v2.__dict__.update(self.__dict__)
        eagle_worker_v2._eagle_draft_worker = self._draft_worker._eagle_draft_worker
        
        return eagle_worker_v2.verify(batch)

    def update_weights_from_tensor(self, recv_req):
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
        
        eagle_worker_v2 = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        eagle_worker_v2.__dict__.update(self.__dict__)
        eagle_worker_v2._eagle_draft_worker = self._draft_worker._eagle_draft_worker
        
        return eagle_worker_v2.update_weights_from_tensor(recv_req)
