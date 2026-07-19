# PyTorch operators that use atomics on CUDA, and near-misses.
import torch


def moe_combine(out, expert_out, row_ids, weights):
    # Class A candidate: index_add_ on float accumulates with atomics on CUDA.
    out.index_add_(0, row_ids, expert_out * weights[:, None])
    # Class A candidate: scatter_add_ on float.
    out.scatter_add_(0, row_ids[:, None].expand_as(expert_out), expert_out)
    # Class A candidate: index_put_ with accumulate=True.
    out.index_put_((row_ids,), expert_out, accumulate=True)
    return out


def counts(expert_ids, weights):
    # Class B: integer bincount with no weights.
    c = torch.bincount(expert_ids, minlength=8)
    # Class A candidate: float-weighted bincount.
    w = torch.bincount(expert_ids, weights=weights, minlength=8)
    # Store race: scatter_ with a tensor source and possibly duplicate indices.
    dst = torch.zeros(8, device=expert_ids.device)
    dst.scatter_(0, expert_ids, weights)
    return c, w, dst


def cumsums(x_int, x_float):
    # Class B: integer cumsum is deterministic; float cumsum on CUDA is listed
    # by PyTorch as non-deterministic.
    return torch.cumsum(x_int, 0), torch.cumsum(x_float, 0)


def not_sites(x):
    # Comment mention only: index_add_ should not match here.
    s = "scatter_add_ in a string"
    torch.use_deterministic_algorithms(True)
    return x + 1, s
