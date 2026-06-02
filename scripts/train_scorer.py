"""Train TAPSLiteScorer: paper equation (4) — KL + λ·BCE_reach."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint.lattice import TopKLattice
from joint.pool import candidate_trie_from_dict
from joint.taps_lite_scorer import TAPSLiteScorer, build_lite_edge_features
from joint.segments import (
    grouped_kl_divergence,
    grouped_softmax,
    propagate_reach_from_edges,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--target-model", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-records", type=int, default=64)
    p.add_argument("--depth-embed-dim", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=32)
    p.add_argument("--draft-hidden-dim", type=int, default=2560)
    p.add_argument("--hidden-proj-dim", type=int, default=32)
    p.add_argument("--scalar-dim", type=int, default=7)
    p.add_argument("--token-proj-dim", type=int, default=32)
    p.add_argument("--lambda-reach", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vocab_embeds(model_path: str) -> torch.Tensor:
    model_dir = Path(model_path)
    index_file = model_dir / "model.safetensors.index.json"
    try:
        from safetensors.torch import load_file
        if index_file.exists():
            index = json.loads(index_file.read_text())
            shard = index["weight_map"]["model.embed_tokens.weight"]
            weights = load_file(str(model_dir / shard))
        else:
            weights = load_file(str(model_dir / "model.safetensors"))
        return weights["model.embed_tokens.weight"].float()
    except Exception:
        from transformers import AutoModelForCausalLM
        _target = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        embed = _target.model.embed_tokens.weight.detach().float()
        del _target
        return embed


def load_trace_records(path, max_records=None):
    trace_path = Path(path)
    files = sorted(trace_path.rglob("*.pt")) if trace_path.is_dir() else [trace_path]
    print(f"Loading {len(files)} trace files...")
    records = []
    drop_keys = [
        "target_hidden_proj", "target_child_logits", "target_greedy_tokens",
        "target_next_token_per_parent", "hidden_provenance",
        "trace_candidate_source", "trace_tree_budget_baseline", "trace_candidate_nodes",
    ]
    for fp in files:
        loaded = torch.load(fp, map_location="cpu", weights_only=False)
        batch = loaded if isinstance(loaded, list) else [loaded]
        for rec in batch:
            for k in drop_keys:
                rec.pop(k, None)
            records.append(rec)
        if max_records and len(records) >= max_records:
            records = records[:max_records]
            break
    has_hidden = sum(1 for r in records if "draft_hidden" in r)
    print(f"  {len(records)} total records, {has_hidden} with draft_hidden")
    return records


def extract_structured_features(records, hidden_proj, scalar_dim=7):
    edge_child_ids_l, edge_parent_tok_l, edge_depths_l, edge_scalars_l, edge_targets_l = [], [], [], [], []
    other_parent_tok_l, other_depths_l, other_scalars_l, other_targets_l = [], [], [], []
    edge_hidden_l, other_hidden_l = [], []
    edge_parent_nid_l, target_reach_l = [], []
    edge_counts, other_counts = [], []
    skipped = 0
    use_hidden = hidden_proj is not None

    for rec in records:
        trie_dict = rec.get("candidate_trie")
        if trie_dict is None:
            skipped += 1
            continue
        trie = candidate_trie_from_dict(trie_dict, device="cpu")
        N = trie.num_nodes
        T = trie.num_total_nodes
        if N == 0:
            skipped += 1
            continue

        top_lp = rec["top_log_probs"].float()
        H = top_lp.shape[0]
        if H == 0:
            skipped += 1
            continue

        tcp = rec["target_child_probs"].float()
        top = rec["target_other_probs"].float()
        if tcp.numel() != N or top.numel() != T:
            skipped += 1
            continue

        draft_hidden = rec.get("draft_hidden")
        if use_hidden and draft_hidden is None:
            skipped += 1
            continue

        lattice = TopKLattice(
            top_token_ids=rec["top_token_ids"].long(),
            top_log_probs=top_lp,
            position_entropy=rec["position_entropy"].float(),
            top1_top2_margin=rec["top1_top2_margin"].float(),
            topk_mass=rec["topk_mass"].float(),
            log_z=torch.zeros(H, dtype=torch.float32),
        )

        dh = draft_hidden.float() if use_hidden else None
        feat = build_lite_edge_features(trie, lattice, int(rec["root_token_id"]),
                                        draft_hidden=dh, hidden_proj=hidden_proj,
                                        scalar_dim=scalar_dim)

        edge_child_ids_l.append(feat["child_ids"])
        edge_parent_tok_l.append(feat["parent_ids"])
        edge_depths_l.append(feat["depths"])
        edge_scalars_l.append(feat["scalars"])
        edge_targets_l.append(tcp.clamp(1e-6, 1 - 1e-6))

        other_parent_tok_l.append(feat["parent_ids_other"])
        other_depths_l.append(feat["parent_depths_other"])
        other_scalars_l.append(feat["parent_scalars_other"])
        other_targets_l.append(top.clamp(1e-6, 1 - 1e-6))

        if use_hidden:
            edge_hidden_l.append(feat["edge_hidden_proj"].detach())
            other_hidden_l.append(feat["other_hidden_proj"].detach())

        edge_parent_nid_l.append(trie.edge_parent_ids)

        child_nids = torch.arange(1, N + 1, dtype=torch.long)
        tr = propagate_reach_from_edges(
            tcp.clamp(0, 1), trie.edge_parent_ids, child_nids,
            torch.tensor([0], dtype=torch.long), T, trie.depths,
        )
        target_reach_l.append(tr)

        edge_counts.append(N)
        other_counts.append(T)

    if skipped:
        print(f"  skipped {skipped} invalid records")

    eo = torch.zeros(len(edge_counts) + 1, dtype=torch.long)
    eo[1:] = torch.tensor(edge_counts).cumsum(0)
    oo = torch.zeros(len(other_counts) + 1, dtype=torch.long)
    oo[1:] = torch.tensor(other_counts).cumsum(0)
    nr = len(edge_counts)
    print(f"  {int(eo[-1])} edges, {int(oo[-1])} parents ({nr} records)")

    result = {
        "edge_child_ids": torch.cat(edge_child_ids_l),
        "edge_parent_tok": torch.cat(edge_parent_tok_l),
        "edge_depths": torch.cat(edge_depths_l),
        "edge_scalars": torch.cat(edge_scalars_l),
        "edge_targets": torch.cat(edge_targets_l),
        "other_parent_tok": torch.cat(other_parent_tok_l),
        "other_depths": torch.cat(other_depths_l),
        "other_scalars": torch.cat(other_scalars_l),
        "other_targets": torch.cat(other_targets_l),
        "edge_parent_nid": torch.cat(edge_parent_nid_l),
        "target_reach": torch.cat(target_reach_l),
        "edge_offsets": eo, "other_offsets": oo, "num_records": nr,
    }
    if use_hidden:
        result["edge_hidden_proj"] = torch.cat(edge_hidden_l)
        result["other_hidden_proj"] = torch.cat(other_hidden_l)
    return result


def collate_batch(data, indices, device, use_hidden=False):
    b = {k: [] for k in [
        "ec", "ept", "ed", "es", "et",
        "opt", "od", "os", "ot",
        "epn", "cni", "tr",
    ]}
    if use_hidden:
        b["ehp"] = []
        b["ohp"] = []
    root_ids = []
    e_off, t_off = 0, 0

    for r in indices:
        es, ee = int(data["edge_offsets"][r]), int(data["edge_offsets"][r + 1])
        os, oe = int(data["other_offsets"][r]), int(data["other_offsets"][r + 1])
        N, T = ee - es, oe - os

        b["ec"].append(data["edge_child_ids"][es:ee])
        b["ept"].append(data["edge_parent_tok"][es:ee])
        b["ed"].append(data["edge_depths"][es:ee])
        b["es"].append(data["edge_scalars"][es:ee])
        b["et"].append(data["edge_targets"][es:ee])

        b["opt"].append(data["other_parent_tok"][os:oe])
        b["od"].append(data["other_depths"][os:oe])
        b["os"].append(data["other_scalars"][os:oe])
        b["ot"].append(data["other_targets"][os:oe])

        b["epn"].append(data["edge_parent_nid"][es:ee] + t_off)
        b["cni"].append(torch.arange(1, N + 1, dtype=torch.long) + t_off)
        b["tr"].append(data["target_reach"][os:oe])

        if use_hidden:
            b["ehp"].append(data["edge_hidden_proj"][es:ee])
            b["ohp"].append(data["other_hidden_proj"][os:oe])

        root_ids.append(t_off)
        e_off += N
        t_off += T

    result = {
        "edge_child_ids": torch.cat(b["ec"]).to(device),
        "edge_parent_tok": torch.cat(b["ept"]).to(device),
        "edge_depths": torch.cat(b["ed"]).to(device),
        "edge_scalars": torch.cat(b["es"]).to(device).float(),
        "edge_targets": torch.cat(b["et"]).to(device),
        "other_parent_tok": torch.cat(b["opt"]).to(device),
        "other_depths": torch.cat(b["od"]).to(device),
        "other_scalars": torch.cat(b["os"]).to(device).float(),
        "other_targets": torch.cat(b["ot"]).to(device),
        "edge_parent_nid": torch.cat(b["epn"]).to(device),
        "child_node_ids": torch.cat(b["cni"]).to(device),
        "target_reach": torch.cat(b["tr"]).to(device),
        "root_ids": torch.tensor(root_ids, dtype=torch.long, device=device),
        "total_nodes": t_off,
    }
    if use_hidden:
        result["edge_hidden_proj"] = torch.cat(b["ehp"]).to(device)
        result["other_hidden_proj"] = torch.cat(b["ohp"]).to(device)
    return result


def forward_and_loss(model, batch, lambda_reach, use_hidden=False, vocab_embeds=None):
    ehp = batch.get("edge_hidden_proj") if use_hidden else None
    ohp = batch.get("other_hidden_proj") if use_hidden else None

    edge_logits, other_logits = model(
        batch["edge_child_ids"], batch["edge_parent_tok"],
        batch["edge_depths"], batch["edge_scalars"],
        batch["other_parent_tok"], batch["other_depths"], batch["other_scalars"],
        edge_hidden_proj=ehp, other_hidden_proj=ohp,
        vocab_embeds=vocab_embeds,
    )

    pids = batch["edge_parent_nid"]

    kl = grouped_kl_divergence(edge_logits, pids, other_logits,
                               batch["edge_targets"], batch["other_targets"])

    q_cond, _ = grouped_softmax(edge_logits, pids, other_logits)
    pred_reach = propagate_reach_from_edges(
        q_cond, pids, batch["child_node_ids"], batch["root_ids"],
        batch["total_nodes"], batch["edge_depths"],
    )
    bce_reach = F.binary_cross_entropy(
        pred_reach.clamp(1e-6, 1 - 1e-6),
        batch["target_reach"].clamp(1e-6, 1 - 1e-6),
    )

    total = kl + lambda_reach * bce_reach
    return total, kl.item(), bce_reach.item()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"device: {device}")

    print(f"Loading vocab embeddings from {args.target_model}...")
    vocab_embeds = load_vocab_embeds(args.target_model)
    vocab_size, embed_dim = vocab_embeds.shape
    print(f"  vocab_embeds: [{vocab_size}, {embed_dim}]")
    vocab_embeds = vocab_embeds.to(device)

    use_hidden = args.draft_hidden_dim > 0

    model = TAPSLiteScorer(
        depth_embed_dim=args.depth_embed_dim, hidden_dim=args.hidden_dim,
        max_depth=args.max_depth,
        draft_hidden_dim=args.draft_hidden_dim,
        hidden_proj_dim=args.hidden_proj_dim,
        scalar_dim=args.scalar_dim,
        use_target_embeds=True,
        vocab_embed_dim=embed_dim,
        token_proj_dim=args.token_proj_dim,
    ).to(device)
    print(f"TAPSLiteScorer params: {sum(p.numel() for p in model.parameters())}")
    print(f"  use_target_embeds=True, token_proj_dim={args.token_proj_dim}")
    print(f"  scalar_dim={args.scalar_dim}, draft_hidden_dim={args.draft_hidden_dim}")
    print(f"  loss: KL + {args.lambda_reach} * BCE_reach")

    records = load_trace_records(args.traces, max_records=args.max_records)

    hp = model.hidden_proj if use_hidden else None
    if hp is not None:
        hp = hp.cpu()
    print("Extracting structured features...")
    t0 = time.time()
    data = extract_structured_features(records, hidden_proj=hp, scalar_dim=args.scalar_dim)
    del records
    print(f"  extraction took {time.time() - t0:.0f}s")

    nr = data["num_records"]
    rng = random.Random(args.seed)
    rec_idx = list(range(nr))
    rng.shuffle(rec_idx)
    val_n = max(1, int(nr * args.val_fraction))
    val_recs = rec_idx[:val_n]
    train_recs = rec_idx[val_n:]
    print(f"Train: {len(train_recs)} records, Val: {len(val_recs)} records")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    lambda_reach = args.lambda_reach

    best_val_loss = float("inf")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    br = args.batch_records

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        rng.shuffle(train_recs)
        run_kl, run_reach, steps = 0., 0., 0

        for bi in range(0, len(train_recs), br):
            batch_idx = train_recs[bi:bi + br]
            batch = collate_batch(data, batch_idx, device, use_hidden=use_hidden)
            if batch["edge_child_ids"].numel() == 0:
                continue

            loss, lkl, lreach = forward_and_loss(model, batch, lambda_reach,
                                                  use_hidden=use_hidden, vocab_embeds=vocab_embeds)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            run_kl += lkl; run_reach += lreach
            steps += 1

        scheduler.step()

        model.eval()
        val_kl, val_reach, val_steps = 0., 0., 0
        with torch.inference_mode():
            for bi in range(0, len(val_recs), br):
                batch_idx = val_recs[bi:bi + br]
                batch = collate_batch(data, batch_idx, device, use_hidden=use_hidden)
                if batch["edge_child_ids"].numel() == 0:
                    continue
                _, lkl, lreach = forward_and_loss(model, batch, lambda_reach,
                                                   use_hidden=use_hidden, vocab_embeds=vocab_embeds)
                val_kl += lkl; val_reach += lreach
                val_steps += 1

        s = max(steps, 1); vs = max(val_steps, 1)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s): "
            f"train(kl={run_kl/s:.4f} bce_reach={run_reach/s:.4f}) "
            f"val(kl={val_kl/vs:.4f} bce_reach={val_reach/vs:.4f}) "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        val_total = val_kl / vs + lambda_reach * val_reach / vs
        if val_total < best_val_loss:
            best_val_loss = val_total
            model.save_checkpoint(
                output_dir / "best.pt", epoch=epoch + 1,
                val_kl=val_kl / vs, val_bce_reach=val_reach / vs,
                lambda_reach=lambda_reach,
            )
            print(f"  -> saved best (val_total={val_total:.4f})")

    model.save_checkpoint(output_dir / "last.pt", epoch=args.epochs)
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
