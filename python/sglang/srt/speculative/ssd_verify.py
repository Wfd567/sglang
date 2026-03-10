import torch
from typing import Optional, Tuple, List


def ssd_verify(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    speculations: torch.Tensor,
    temperatures_target: torch.Tensor,
    temperatures_draft: torch.Tensor,
    cache_hits: Optional[torch.Tensor] = None,
    sampler_x: Optional[float] = None,
    async_fan_out: Optional[int] = None,
    jit_speculate: bool = False,
) -> Tuple[List[List[int]], List[int]]:
    """
    SSD (Speculative Speculative Decoding) verification
    
    This implements the verification logic from the SSD paper.
    """
    device = logits_p.device
    B, Kp1, V = logits_p.shape
    K = Kp1 - 1

    draft_tokens = speculations[:, 1:]
    preds_p = logits_p.argmax(dim=-1)

    matches = draft_tokens == preds_p[:, :-1]
    any_mismatch = (~matches).any(dim=1)
    first_mismatch = (~matches).int().argmax(dim=1)
    
    accept_greedy = torch.where(
        any_mismatch,
        first_mismatch,
        torch.full_like(first_mismatch, K)
    )
    batch_idx = torch.arange(B, device=device)
    rec_greedy = preds_p[batch_idx, accept_greedy]

    accepted_suffixes: List[List[int]] = []
    starts = speculations[:, 0].tolist()
    counts = accept_greedy.tolist()

    for b in range(B):
        n = counts[b]
        suffix = [starts[b]] + draft_tokens[b, :n].tolist()
        accepted_suffixes.append(suffix)

    return accepted_suffixes, rec_greedy.tolist()
