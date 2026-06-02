"""TAPS-Lite scorer with DFlash hidden state features + paper Algorithm 1 selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import JointDDTConfig
from .lattice import TopKLattice
from .pool import CandidateTrie, build_marginal_candidate_trie
from .segments import grouped_softmax
from .selector import propagate_reach, _closure_mask_from_ranked_nodes
from .tree import SelectedTree, compact_selected_trie


class TAPSLiteScorer(nn.Module):
    SCALAR_DIM = 5

    def __init__(
        self,
        hash_buckets: int = 8192,
        token_embed_dim: int = 16,
        depth_embed_dim: int = 8,
        hidden_dim: int = 64,
        max_depth: int = 32,
        draft_hidden_dim: int = 0,
        hidden_proj_dim: int = 32,
        scalar_dim: int = 5,
        use_target_embeds: bool = False,
        vocab_embed_dim: int = 2560,
        token_proj_dim: int = 32,
    ) -> None:
        super().__init__()
        self.config = dict(
            hash_buckets=hash_buckets,
            token_embed_dim=token_embed_dim,
            depth_embed_dim=depth_embed_dim,
            hidden_dim=hidden_dim,
            max_depth=max_depth,
            draft_hidden_dim=draft_hidden_dim,
            hidden_proj_dim=hidden_proj_dim,
            scalar_dim=scalar_dim,
            use_target_embeds=use_target_embeds,
            vocab_embed_dim=vocab_embed_dim,
            token_proj_dim=token_proj_dim,
        )
        self.hash_buckets = hash_buckets
        self.max_depth = max_depth
        self.draft_hidden_dim = draft_hidden_dim
        self.scalar_dim = scalar_dim
        self.use_target_embeds = use_target_embeds

        hp = hidden_proj_dim if draft_hidden_dim > 0 else 0

        if use_target_embeds:
            self.token_proj = nn.Linear(vocab_embed_dim, token_proj_dim, bias=False)
            tok_dim = token_proj_dim
        else:
            self.token_embed = nn.Embedding(hash_buckets, token_embed_dim)
            tok_dim = token_embed_dim

        self.depth_embed = nn.Embedding(max_depth, depth_embed_dim)

        if draft_hidden_dim > 0:
            self.hidden_proj = nn.Linear(draft_hidden_dim, hp, bias=False)

        edge_in = 2 * tok_dim + depth_embed_dim + scalar_dim + hp
        self.edge_norm = nn.LayerNorm(edge_in)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        other_in = tok_dim + depth_embed_dim + scalar_dim + hp
        self.other_norm = nn.LayerNorm(other_in)
        self.other_mlp = nn.Sequential(
            nn.Linear(other_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _hash(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.long() % self.hash_buckets

    @torch.no_grad()
    def set_vocab_embeds(self, embed_weight: torch.Tensor) -> None:
        projected = self.token_proj(
            embed_weight.to(dtype=self.token_proj.weight.dtype, device=self.token_proj.weight.device)
        )
        self.register_buffer("_projected_vocab", projected, persistent=False)

    def _tok_emb(self, token_ids: torch.Tensor, vocab_embeds: torch.Tensor | None) -> torch.Tensor:
        if self.use_target_embeds:
            if vocab_embeds is not None:
                return self.token_proj(vocab_embeds[token_ids])
            return self._projected_vocab[token_ids]
        return self.token_embed(self._hash(token_ids))

    def forward(
        self,
        child_ids: torch.Tensor,
        parent_ids: torch.Tensor,
        depths: torch.Tensor,
        scalars: torch.Tensor,
        parent_ids_other: torch.Tensor,
        parent_depths_other: torch.Tensor,
        parent_scalars_other: torch.Tensor,
        edge_hidden_proj: torch.Tensor | None = None,
        other_hidden_proj: torch.Tensor | None = None,
        vocab_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        child_emb = self._tok_emb(child_ids, vocab_embeds)
        parent_emb = self._tok_emb(parent_ids, vocab_embeds)
        depth_emb = self.depth_embed(depths.clamp(0, self.max_depth - 1))
        parts = [child_emb, parent_emb, depth_emb, scalars.float()]
        if edge_hidden_proj is not None:
            parts.append(edge_hidden_proj.float())
        edge_input = torch.cat(parts, dim=-1)
        edge_logits = self.edge_mlp(self.edge_norm(edge_input)).squeeze(-1)

        par_emb = self._tok_emb(parent_ids_other, vocab_embeds)
        par_depth = self.depth_embed(parent_depths_other.clamp(0, self.max_depth - 1))
        oparts = [par_emb, par_depth, parent_scalars_other.float()]
        if other_hidden_proj is not None:
            oparts.append(other_hidden_proj.float())
        other_input = torch.cat(oparts, dim=-1)
        other_logits = self.other_mlp(self.other_norm(other_input)).squeeze(-1)

        return edge_logits, other_logits

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        payload = {
            "model_type": "TAPSLiteScorer",
            "model_config": self.config,
            "state_dict": self.state_dict(),
            **extra,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str | torch.device = "cpu"
    ) -> tuple["TAPSLiteScorer", dict[str, Any]]:
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["model_config"]
        config.setdefault("draft_hidden_dim", 0)
        config.setdefault("hidden_proj_dim", 32)
        config.setdefault("scalar_dim", 5)
        config.setdefault("use_target_embeds", False)
        config.setdefault("vocab_embed_dim", 2560)
        config.setdefault("token_proj_dim", 32)
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        return model, payload


def load_taps_lite_scorer(
    path: str | Path, device: torch.device | str
) -> tuple[TAPSLiteScorer, dict[str, Any]]:
    model, payload = TAPSLiteScorer.from_checkpoint(path, device=device)
    model = model.to(device)
    model.eval()
    return model, payload


def build_lite_edge_features(
    trie: CandidateTrie,
    lattice: TopKLattice,
    root_token_id: torch.Tensor | int,
    draft_hidden: torch.Tensor | None = None,
    hidden_proj: nn.Module | None = None,
    scalar_dim: int = 5,
) -> dict[str, torch.Tensor]:
    device = trie.device
    N = trie.num_nodes
    T = trie.num_total_nodes
    H = lattice.horizon

    if isinstance(root_token_id, int):
        root_token_id = torch.tensor(root_token_id, dtype=torch.long, device=device)
    root_token_id = root_token_id.reshape(1).to(device=device, dtype=torch.long)

    if N == 0:
        z_long = torch.empty(0, dtype=torch.long, device=device)
        z_float = torch.empty(0, scalar_dim, dtype=torch.float32, device=device)
        p_float = torch.zeros(1, scalar_dim, dtype=torch.float32, device=device)
        return dict(
            child_ids=z_long, parent_ids=z_long, depths=z_long,
            scalars=z_float,
            parent_ids_other=root_token_id,
            parent_depths_other=torch.zeros(1, dtype=torch.long, device=device),
            parent_scalars_other=p_float,
        )

    all_token_ids = torch.cat([root_token_id, trie.token_ids], dim=0)
    parent_token_ids = all_token_ids[trie.edge_parent_ids]

    depth_index = (trie.depths - 1).clamp(0, H - 1)
    scalar_cols = [
        trie.step_log_probs,
        lattice.position_entropy[depth_index].float(),
        lattice.top1_top2_margin[depth_index].float(),
        lattice.topk_mass[depth_index].float(),
        trie.depths.float() / max(H, 1),
    ]
    if scalar_dim >= 7:
        scalar_cols.extend([
            trie.cum_log_probs,
            trie.cum_log_probs / trie.depths.float().clamp_min(1.0),
        ])
    scalars = torch.stack(scalar_cols, dim=-1)

    parent_depths_all = torch.zeros(T, dtype=torch.long, device=device)
    parent_depths_all[1:] = trie.depths

    ent = torch.zeros(T, dtype=torch.float32, device=device)
    mar = torch.zeros(T, dtype=torch.float32, device=device)
    mas = torch.zeros(T, dtype=torch.float32, device=device)
    if H > 0:
        ent[1:] = lattice.position_entropy[depth_index].float()
        mar[1:] = lattice.top1_top2_margin[depth_index].float()
        mas[1:] = lattice.topk_mass[depth_index].float()

    parent_scalar_cols = [
        torch.zeros(T, dtype=torch.float32, device=device),
        ent, mar, mas,
        parent_depths_all.float() / max(H, 1),
    ]
    if scalar_dim >= 7:
        parent_cum = torch.zeros(T, dtype=torch.float32, device=device)
        parent_cum[1:] = trie.cum_log_probs
        parent_scalar_cols.extend([
            parent_cum,
            parent_cum / parent_depths_all.float().clamp_min(1.0),
        ])
    parent_scalars = torch.stack(parent_scalar_cols, dim=-1)

    result = dict(
        child_ids=trie.token_ids,
        parent_ids=parent_token_ids,
        depths=trie.depths,
        scalars=scalars,
        parent_ids_other=all_token_ids,
        parent_depths_other=parent_depths_all,
        parent_scalars_other=parent_scalars,
    )

    if draft_hidden is not None and hidden_proj is not None:
        proj = hidden_proj(draft_hidden.to(device=device, dtype=hidden_proj.weight.dtype))
        HP = proj.shape[0]

        edge_idx = (trie.depths - 1).clamp(0, HP - 1)
        result["edge_hidden_proj"] = proj[edge_idx]

        other_idx = parent_depths_all.clamp(0, HP - 1)
        result["other_hidden_proj"] = proj[other_idx]

    return result


@torch.inference_mode()
def taps_lite_select_v2(
    scorer: TAPSLiteScorer,
    lattice: TopKLattice,
    root_token_id: torch.Tensor | int,
    max_tree_nodes: int = 64,
    max_tree_seqs: int = 64,
    max_pool_nodes: int = 768,
    max_pool_seqs: int = 48,
    lambda_node: float = 1.0,
    lambda_depth: float = 0.002,
    draft_hidden: torch.Tensor | None = None,
) -> SelectedTree:
    """Paper Algorithm 1: score edges → q_cond → q_reach → utility → prefix-closed selection."""
    device = lattice.top_log_probs.device
    H = lattice.horizon
    if H == 0:
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    cfg = JointDDTConfig(
        candidate_pool_nodes=max_pool_nodes,
        candidate_pool_sequences=max_pool_seqs,
        joint_topk=lattice.topk,
    )
    trie = build_marginal_candidate_trie(lattice, cfg, max_nodes=max_pool_nodes, width=max_pool_seqs)
    if trie.num_nodes == 0:
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    hp = getattr(scorer, "hidden_proj", None)
    dh = draft_hidden if scorer.draft_hidden_dim > 0 else None
    sd = getattr(scorer, "scalar_dim", TAPSLiteScorer.SCALAR_DIM)
    features = build_lite_edge_features(trie, lattice, root_token_id,
                                        draft_hidden=dh, hidden_proj=hp,
                                        scalar_dim=sd)
    edge_logits, other_logits = scorer(**features)

    q_cond, _ = grouped_softmax(edge_logits, trie.edge_parent_ids, other_logits)
    q_reach = propagate_reach(trie, q_cond, max_depth=H)

    _ln = getattr(scorer, "_lambda_node", lambda_node)
    _ld = getattr(scorer, "_lambda_depth", lambda_depth)
    depth_cost = _ln + _ld * trie.depths.float().clamp_min(1.0)
    utility = q_reach[1:] / depth_cost.clamp_min(0.01)

    take = min(trie.num_nodes, max_tree_seqs * 4)
    order = torch.argsort(utility, descending=True)[:take] + 1
    selected = _closure_mask_from_ranked_nodes(
        order, trie.parents,
        max_nonroot_nodes=max_tree_nodes,
        min_nonroot_nodes=min(4, trie.num_nodes, max_tree_nodes),
        max_depth=H,
        max_endpoints=max_tree_seqs,
    )

    if not selected[1:].any():
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    return compact_selected_trie(trie, selected[1:], max_depth=H)


# ---------------------------------------------------------------------------
# Pre-allocated numpy buffers for CPU beam search
# ---------------------------------------------------------------------------

class _FastBufs:
    """Pre-allocated numpy buffers for fast beam search."""
    __slots__ = ("cap", "nd_parent", "nd_token", "nd_depth", "nd_score_cum", "nd_draft_cum")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.nd_parent = np.empty(cap, dtype=np.int32)
        self.nd_token = np.empty(cap, dtype=np.int64)
        self.nd_depth = np.empty(cap, dtype=np.int32)
        self.nd_score_cum = np.empty(cap, dtype=np.float64)
        self.nd_draft_cum = np.empty(cap, dtype=np.float64)


_fast_bufs: _FastBufs | None = None


# ---------------------------------------------------------------------------
# hybrid: CPU beam search (fast pool) + GPU scoring (correct parents)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def taps_hybrid_select(
    scorer: TAPSLiteScorer,
    lattice: TopKLattice,
    root_token_id: torch.Tensor | int,
    max_tree_nodes: int = 64,
    max_tree_seqs: int = 64,
    max_pool_nodes: int = 512,
    max_pool_seqs: int = 34,
    draft_hidden: torch.Tensor | None = None,
) -> SelectedTree:
    """Hybrid: CPU beam search for pool + GPU scoring for quality.

    Saves ~5ms by replacing GPU beam search with CPU numpy beam search,
    then applies proper GPU scoring with correct parent context.
    """
    global _fast_bufs

    device = lattice.top_log_probs.device
    H, K = lattice.top_log_probs.shape
    if H == 0:
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    # --- Phase 1: Single GPU→CPU transfer of lattice data ---
    lattice_cpu = torch.stack([
        lattice.top_log_probs.float(),
        lattice.top_token_ids.float(),
    ]).cpu().numpy()
    cpu_lp = lattice_cpu[0]
    cpu_tok = lattice_cpu[1].astype(np.int64)

    # --- Phase 2: CPU beam search to build pool (~0.3ms) ---
    cap = min(max_pool_nodes, H * K) + 1
    if _fast_bufs is None or _fast_bufs.cap < cap:
        _fast_bufs = _FastBufs(cap)
    buf = _fast_bufs
    buf.nd_parent[0] = -1
    buf.nd_token[0] = -1
    buf.nd_depth[0] = 0
    buf.nd_draft_cum[0] = 0.0
    num_nodes = 1
    beam = np.array([0], dtype=np.int32)

    for d in range(H):
        if beam.size == 0 or num_nodes >= cap:
            break
        n_beam = beam.size
        n_children = n_beam * K
        child_draft_cum = buf.nd_draft_cum[beam, None] + cpu_lp[d]
        child_parents = np.repeat(beam, K)
        child_tokens = np.tile(cpu_tok[d], n_beam)
        flat_scores = child_draft_cum.ravel()
        budget = min(max_pool_seqs, cap - num_nodes, n_children)
        if budget <= 0:
            break
        if budget < n_children:
            top_idx = np.argpartition(flat_scores, -budget)[-budget:]
        else:
            top_idx = np.arange(n_children)
        start_idx = num_nodes
        end_idx = start_idx + top_idx.size
        sl = slice(start_idx, end_idx)
        buf.nd_parent[sl] = child_parents[top_idx]
        buf.nd_token[sl] = child_tokens[top_idx]
        buf.nd_depth[sl] = d + 1
        buf.nd_draft_cum[sl] = flat_scores[top_idx]
        num_nodes = end_idx
        beam = np.arange(start_idx, end_idx, dtype=np.int32)

    if num_nodes <= 1:
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    N = num_nodes - 1

    # --- Phase 3: Build CandidateTrie on GPU ---
    pool_tokens = torch.from_numpy(buf.nd_token[1:num_nodes].copy()).to(device, dtype=torch.long, non_blocking=True)
    pool_depths = torch.from_numpy(buf.nd_depth[1:num_nodes].copy()).to(device, dtype=torch.long, non_blocking=True)
    pool_parents = torch.from_numpy(buf.nd_parent[:num_nodes].copy()).to(device, dtype=torch.long, non_blocking=True)
    pool_cum_lp = torch.from_numpy(buf.nd_draft_cum[1:num_nodes].copy()).to(device, dtype=torch.float32, non_blocking=True)

    parent_cum = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    parent_cum[1:] = pool_cum_lp
    pool_step_lp = pool_cum_lp - parent_cum[pool_parents[1:]]

    trie = CandidateTrie(
        token_ids=pool_tokens,
        depths=pool_depths,
        parents=pool_parents,
        ranks=torch.zeros(N, dtype=torch.long, device=device),
        step_log_probs=pool_step_lp,
        cum_log_probs=pool_cum_lp,
        source_ids=torch.zeros(N, dtype=torch.long, device=device),
        path_hashes=torch.arange(1, N + 1, dtype=torch.long, device=device),
    )

    # --- Phase 4: GPU scoring ---
    hp = getattr(scorer, "hidden_proj", None)
    dh = draft_hidden if scorer.draft_hidden_dim > 0 else None
    sd = getattr(scorer, "scalar_dim", TAPSLiteScorer.SCALAR_DIM)
    features = build_lite_edge_features(trie, lattice, root_token_id,
                                        draft_hidden=dh, hidden_proj=hp,
                                        scalar_dim=sd)
    edge_logits, other_logits = scorer(**features)

    _lean = getattr(scorer, "_lean_hybrid", False)
    if _lean:
        # Skip softmax + reach: use edge_logits directly (saves ~0.7ms)
        score_per_node = torch.zeros(trie.num_total_nodes, dtype=torch.float32, device=device)
        score_per_node[1:] = edge_logits
    else:
        q_cond, _ = grouped_softmax(edge_logits, trie.edge_parent_ids, other_logits)
        score_per_node = propagate_reach(trie, q_cond, max_depth=H)

    # --- Phase 5: CPU tree selection ---
    cpu_data = torch.stack([
        score_per_node,
        trie.parents.float(),
        torch.cat([torch.tensor([0], device=device), trie.token_ids.float()]),
        torch.cat([torch.tensor([0], device=device), trie.depths.float()]),
    ]).cpu().numpy()
    qr = cpu_data[0]
    nd_parent = cpu_data[1].astype(np.int32)
    nd_token = cpu_data[2].astype(np.int64)
    nd_depth = cpu_data[3].astype(np.int32)
    T = N + 1

    non_root = np.arange(1, T)
    ranked = non_root[np.argsort(-qr[1:])]

    selected = {0}
    n_leaves = 0
    for nid in ranked:
        if n_leaves >= max_tree_seqs or len(selected) >= max_tree_nodes + 1:
            break
        ancestors = []
        cur = int(nid)
        while cur > 0 and cur not in selected:
            ancestors.append(cur)
            cur = int(nd_parent[cur])
        if cur not in selected or len(selected) + len(ancestors) > max_tree_nodes + 1:
            continue
        selected.update(ancestors)
        n_leaves += 1

    selected.discard(0)
    if not selected:
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    sel_sorted = sorted(selected)
    old_to_new_map = {0: 0}
    edge_to_new: dict[tuple[int, int], int] = {}
    kept_tokens: list[int] = []
    kept_depths: list[int] = []
    kept_parents_new: list[int] = []
    child_maps: list[dict[int, int]] = [{}]

    for old_id in sel_sorted:
        par_old = int(nd_parent[old_id])
        if par_old not in old_to_new_map:
            continue
        par_new = old_to_new_map[par_old]
        tok = int(nd_token[old_id])
        key = (par_new, tok)
        if key in edge_to_new:
            old_to_new_map[old_id] = edge_to_new[key]
            continue
        new_id = len(kept_tokens) + 1
        old_to_new_map[old_id] = new_id
        edge_to_new[key] = new_id
        kept_tokens.append(tok)
        kept_depths.append(int(nd_depth[old_id]))
        kept_parents_new.append(par_new)
        child_maps.append({})
        child_maps[par_new][tok] = new_id

    if not kept_tokens:
        from .tiny_scorer import _make_empty_tree
        return _make_empty_tree(device)

    total = len(kept_tokens) + 1
    vis = np.zeros((total, total), dtype=np.bool_)
    vis[0, 0] = True
    for i in range(1, total):
        vis[i, i] = True
        cur = kept_parents_new[i - 1]
        while cur >= 0:
            vis[i, cur] = True
            if cur == 0:
                break
            cur = kept_parents_new[cur - 1] if cur > 0 else -1

    parents_arr = np.array([-1] + kept_parents_new, dtype=np.int64)
    return SelectedTree(
        token_ids=torch.from_numpy(np.array(kept_tokens, dtype=np.int64)).to(device, non_blocking=True),
        depths=torch.from_numpy(np.array(kept_depths, dtype=np.int64)).to(device, non_blocking=True),
        parents=torch.from_numpy(parents_arr).to(device, non_blocking=True),
        visibility=torch.from_numpy(vis).to(device, non_blocking=True),
        old_to_new=torch.zeros(T, dtype=torch.long, device=device),
        selected_old_node_ids=torch.empty(0, dtype=torch.long, device=device),
        child_maps=child_maps,
    )
