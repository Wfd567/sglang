import logging
from typing import Optional

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.ssd_tree_cache import SSDTreeCache
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import empty_context

logger = logging.getLogger(__name__)


class SSDWorker(EAGLEWorker, BaseSpecWorker):
    """
    SSD (Speculative Speculative Decoding) Worker 实现
    包含SSD的三大核心创新：
    1. 树状缓存机制 - 缓存之前的投机结果
    2. 分支因子多路径预测 - 预先验证多个可能的验证结果
    3. (待实现) 异步模式 - Draft和Target并行运行
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
        self.ssd_async = server_args.ssd_async
        self.ssd_fan_out = server_args.ssd_fan_out or 4
        self.ssd_jit_speculate = server_args.ssd_jit_speculate
        self.ssd_sampler_x = server_args.ssd_sampler_x or 0.6
        self.ssd_max_cached_sequences = server_args.ssd_max_cached_sequences or 1024
        
        logger.info(f"SSDWorker initialized with:")
        logger.info(f"  - ssd_async: {self.ssd_async}")
        logger.info(f"  - ssd_fan_out: {self.ssd_fan_out}")
        
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        
        self.draft_model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        
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

    def _init_fan_out_buffers(self):
        """初始化分支因子相关的预分配buffer"""
        K = self.speculative_num_steps
        F = self.ssd_fan_out
        
        self._fan_idx_hit = torch.arange(K + 1, device=self.device, dtype=torch.int64).repeat_interleave(F)
        self._fan_idx_miss = torch.arange(K + 1, device=self.device, dtype=torch.int64).repeat_interleave(F)
        self._arange_kp1 = torch.arange(K + 1, device=self.device, dtype=torch.int64)
        self._arange_f = torch.arange(F, device=self.device, dtype=torch.int64)
        
        logger.info(f"SSD fan-out buffers initialized with K={K}, F={F}")

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @target_worker.setter
    def target_worker(self, value: TpModelWorker):
        self._target_worker = value

    @property
    def draft_worker(self):
        return self

    def clear_cache_pool(self):
        """清空缓存池"""
        self.ssd_tree_cache.clear()
        logger.info("SSD tree cache cleared")
