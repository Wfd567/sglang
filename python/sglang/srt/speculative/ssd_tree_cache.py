import torch
from typing import Optional, Tuple


class SSDTreeCache:
    """
    SSD的树状缓存核心实现
    这是SSD的核心创新之一：缓存之前的投机结果以避免重复计算
    """

    def __init__(
        self,
        device: torch.device,
        vocab_size: int,
        hidden_size: Optional[int] = None,
        max_cached_sequences: int = 1024,
    ):
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_cached_sequences = max_cached_sequences

        self._reset_tree_cache_tensors()
        self.hit_count = 0
        self.total_count = 0

    def _reset_tree_cache_tensors(self):
        """重置树状缓存为空"""
        self.tree_cache_keys = torch.zeros((0, 3), dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        self.tree_cache_activations = None

    def hit_cache_and_respond(
        self,
        request_keys: torch.Tensor,
        batch_size: int,
        speculate_k: int,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检查缓存并返回结果
        
        Args:
            request_keys: 请求键 [B, 3]，格式为(seq_id, k_idx, recovery_token)
            batch_size: 批大小
            speculate_k: 投机深度
            dtype: 数据类型
            
        Returns:
            (out_tokens, out_logits, cache_hits)
        """
        self.total_count += batch_size
        
        out_logits = torch.empty(
            (batch_size, speculate_k, self.vocab_size),
            dtype=dtype,
            device=self.device
        ).uniform_()
        out_tokens = out_logits.argmax(dim=-1)
        cache_hits = torch.zeros(batch_size, dtype=torch.int64, device=self.device)

        if self.tree_cache_keys.numel() > 0:
            eq = (request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0))
            match = torch.all(eq, dim=2)
            cache_hits = match.any(dim=1)
            self.hit_count += int(cache_hits.sum().item())

            if cache_hits.any():
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits
                out_tokens[sel] = self.tree_cache_tokens[idx[sel]]
                out_logits[sel] = self.tree_cache_logits[idx[sel]]

        return out_tokens, out_logits, cache_hits

    def populate_tree_cache(
        self,
        seq_ids_expanded: torch.Tensor,
        k_flat: torch.Tensor,
        rec_flat: torch.Tensor,
        tokens: torch.Tensor,
        logits: torch.Tensor,
        activations: Optional[torch.Tensor] = None,
    ):
        """
        填充树状缓存
        
        Args:
            seq_ids_expanded: 序列ID
            k_flat: k索引
            rec_flat: recovery token
            tokens: 投机token
            logits: 投机logits
            activations: 可选的激活值
        """
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1).contiguous()
        
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits
        
        if activations is not None:
            self.tree_cache_activations = activations

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count

    def clear(self):
        """清空缓存"""
        self._reset_tree_cache_tensors()
        self.hit_count = 0
        self.total_count = 0
