"""Gate 2: rank agreement between the geometric proxy and the projected GGN of the real model.

Implements Section 8 of ET_skip采样算法_最终方案.md. Results only evaluate the proxy; they are
never fed back into the skip table produced by gate1_doptimal.py.

Quantities (all at the unperturbed checkpoint, PI applied):
  directions   V = {v_i}: orthonormal in a fixed Q/K subspace. v_i = u_i w_i^T inside one
               module, u_i orthonormal columns, ||w_i|| = 1, so <v_i, v_j>_F = delta_ij.
  JVP          d/de log p(y | ...; W + e v_i) by central finite differences in float32.
  GGN          for softmax cross-entropy, J^T H J = E_{y~pi}[grad log p_y grad log p_y^T]
               (H = w(diag(pi) - pi pi^T)). We estimate it with labels y sampled once from the
               unperturbed pi and shared across directions (common random numbers).
  G_0          Delta-independent prediction terms: context targets (weight 1) and the first cue
               token predicted by the last context token (weight w_cue).
  C_Delta      cue targets l in [a0, a0+b-2] (weight w_cue); the only terms that move with Delta.
  G_p0         G_0 + mean of C_Delta over uniformly drawn skips (seeded, recorded).
  gain(Delta)  tr[(G_p0 + lambda I)^{-1} C_Delta], lambda = ratio * tr(G_p0)/k.
  proxy        g_Delta(p0) from gate1_doptimal.Proxy.

Pre-registered decision (set the thresholds before running, do not tune afterwards):
  undetermined  if split-half reliability of gain ranks < --min_reliability
  pass          if Spearman(proxy, gain) >= --pass_spearman and top/bottom-quartile AUC >= --pass_auc
  fail          otherwise

Example (EP_1, 8 texts, 48 directions):
  python gate2_ggn_rank.py --model /path/Meta-Llama-3-8B --data lceg_sample.jsonl \
      --cue_text "This is the end of text, please pay attention here," --b 10 --out gate2_ep1.json
  python gate2_ggn_rank.py --model /path/Meta-Llama-3-8B --data lceg_sample.jsonl \
      --cue_token "<|eot_id|>" --b 10 --out gate2_eot.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate1_doptimal import Proxy  # noqa: E402


def set_linear_pi(config, s):
    """Linear PI with factor s; supports rope_parameters (transformers>=5) and rope_scaling (4.x)."""
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict):
        rp = dict(rp)
        rp.update(rope_type="linear", factor=float(s))
        config.rope_parameters = rp
    else:
        config.rope_scaling = {"rope_type": "linear", "type": "linear", "factor": float(s)}


def rope_theta(config):
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(config.rope_theta)


def head_dim(config):
    return int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)


def load_texts(args, tok, rng):
    """Return a list of LongTensor of length a0 (context only, BOS included if requested)."""
    a0 = args.a0
    if args.synthetic_texts:
        vocab = args.synthetic_vocab
        return [torch.from_numpy(rng.integers(0, vocab, a0)).long() for _ in range(args.synthetic_texts)]
    texts = []
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)[args.text_field] if args.data.endswith(".jsonl") else line
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if args.add_bos:
                ids = [tok.bos_token_id] + ids
            if len(ids) >= a0:
                texts.append(torch.tensor(ids[:a0], dtype=torch.long))
            if len(texts) == args.n_texts:
                break
    if len(texts) < args.n_texts:
        raise SystemExit(f"only {len(texts)} documents with >= {a0} tokens in {args.data}")
    return texts


def cue_ids(args, tok):
    if args.cue_ids:
        ids = [int(x) for x in args.cue_ids.split(",")]
    elif args.cue_token:
        tid = tok.convert_tokens_to_ids(args.cue_token)
        if tid is None or tid == tok.unk_token_id:
            raise SystemExit(f"{args.cue_token!r} is not a single token")
        ids = [tid] * args.b
    else:
        ids = tok(args.cue_text, add_special_tokens=False)["input_ids"]
    if len(ids) != args.b:
        raise SystemExit(f"cue has {len(ids)} tokens after tokenization, but --b={args.b}")
    return torch.tensor(ids, dtype=torch.long)


def rank_offsets(Dmax, a0, n):
    """Pre-selected offsets: half spread over [0, Dmax], a quarter in [0, a0), a quarter in the last 1024."""
    h, q = n // 2, n // 4
    pts = np.concatenate([np.linspace(0, Dmax, h), np.linspace(0, a0 - 1, q),
                          np.linspace(max(Dmax - 1023, 0), Dmax, n - h - q)])
    return np.unique(np.round(pts).astype(int))


def build_directions(model, layers, modules, k_per, gen):
    """Orthonormal rank-one directions v = u w^T, u orthonormal within a module."""
    dirs = []
    for li in layers:
        attn = model.model.layers[li].self_attn
        for name in modules:
            W = getattr(attn, name).weight
            out_f, in_f = W.shape
            U, _ = torch.linalg.qr(torch.randn(out_f, k_per, generator=gen, dtype=torch.float64))
            for j in range(k_per):
                w = torch.randn(in_f, generator=gen, dtype=torch.float64)
                dirs.append(dict(layer=li, module=name, W=W,
                                 u=U[:, j].to(W.dtype).to(W.device), w=(w / w.norm()).to(W.dtype).to(W.device)))
    return dirs


@torch.no_grad()
def ctx_forward(model, ids):
    """Context block at positions 0..a0-1; returns hidden states (a0, H) and the KV cache."""
    a0 = ids.numel()
    dev = model.device
    cache = DynamicCache()
    out = model.model(input_ids=ids[None].to(dev), position_ids=torch.arange(a0, device=dev)[None],
                      past_key_values=cache, use_cache=True)
    return out.last_hidden_state[0], cache


@torch.no_grad()
def cue_forward(model, cue, cache, a0, delta):
    """Cue block at assigned positions a0+delta .. a0+delta+b-1 on top of the context cache (Eq. 3)."""
    dev = model.device
    b = cue.numel()
    pos = torch.arange(a0 + delta, a0 + delta + b, device=dev)[None]
    out = model.model(input_ids=cue[None].to(dev), position_ids=pos, past_key_values=cache, use_cache=True)
    cache.crop(-b)  # drop the cue entries; negative crop works in transformers 4.x and 5.x
    assert cache.get_seq_length() == a0
    return out.last_hidden_state[0]


@torch.no_grad()
def log_softmax_chunks(model, h, chunk):
    for s in range(0, h.shape[0], chunk):
        yield s, torch.log_softmax(model.lm_head(h[s:s + chunk]).double(), dim=-1)


@torch.no_grad()
def sample_labels(model, h, m, gen, chunk):
    """m labels per position drawn from the unperturbed predictive distribution."""
    out = torch.empty(h.shape[0], m, dtype=torch.long, device=h.device)
    for s, lsm in log_softmax_chunks(model, h, chunk):
        out[s:s + lsm.shape[0]] = torch.multinomial(lsm.exp(), m, replacement=True, generator=gen)
    return out


@torch.no_grad()
def gather_logp(model, h, labels, chunk):
    out = torch.empty(labels.shape, dtype=torch.float64, device=h.device)
    for s, lsm in log_softmax_chunks(model, h, chunk):
        out[s:s + lsm.shape[0]] = lsm.gather(1, labels[s:s + lsm.shape[0]]).double()
    return out


@torch.no_grad()
def true_target_prob(model, h, targets):
    lsm = torch.log_softmax(model.lm_head(h).float(), dim=-1)
    return lsm.gather(1, targets[:, None].to(h.device)).exp().squeeze(1).cpu().numpy()


def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for val in np.unique(x):
        idx = x == val
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def quartile_auc(proxy, gain):
    """P(gain_top > gain_bottom) over pairs from the proxy's top and bottom quartiles (ties = 0.5)."""
    n = len(proxy)
    q = max(n // 4, 1)
    order = np.argsort(proxy)
    bot, top = gain[order[:q]], gain[order[-q:]]
    diff = top[:, None] - bot[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def directional_scores(model, dirs, eps_of, W0, ids, cue, offsets, lab_ctx, lab_cue, chunk, scale=1.0):
    """Central-difference d/de log p(label) for every direction; returns S_ctx (k,a0,m), S_cue (nD,k,b-1,mc)."""
    a0, b = ids.numel(), cue.numel()
    k = len(dirs)
    S_ctx = np.zeros((k,) + tuple(lab_ctx.shape))
    S_cue = np.zeros((len(offsets), k) + tuple(lab_cue[0].shape))
    for i, d in enumerate(dirs):
        eps = eps_of[(d["layer"], d["module"])] * scale
        lp_ctx, lp_cue = {}, {}
        for sign in (1.0, -1.0):
            with torch.no_grad():
                d["W"].add_(torch.outer(d["u"], d["w"]), alpha=sign * eps)
            h_ctx, cache = ctx_forward(model, ids)
            lp_ctx[sign] = gather_logp(model, h_ctx, lab_ctx, chunk).cpu().numpy()
            lp_cue[sign] = np.stack([
                gather_logp(model, cue_forward(model, cue, cache, a0, int(D))[: b - 1], lab_cue[j], chunk).cpu().numpy()
                for j, D in enumerate(offsets)])
            del cache, h_ctx
            with torch.no_grad():
                d["W"].copy_(W0[(d["layer"], d["module"])])
        S_ctx[i] = (lp_ctx[1.0] - lp_ctx[-1.0]) / (2 * eps)
        S_cue[:, i] = (lp_cue[1.0] - lp_cue[-1.0]) / (2 * eps)
    return S_ctx, S_cue


def ggn_from_scores(S, weights, m):
    """Monte Carlo GGN projection: sum_l w_l (1/m) sum_y s s^T, with S of shape (k, P, m)."""
    A = S * np.sqrt(weights / m)[None, :, None]
    A = A.reshape(S.shape[0], -1)
    return A @ A.T


def analyze(G0_t, C_t, Ch1_t, Ch2_t, rank_idx, unif_idx, proxy_g, ratios, pre, n_boot, rng):
    """Gains, Spearman, quartile AUC, split-half reliability, text bootstrap, lambda sweep."""
    def gains(G0, C, Cref, ratio):
        Gp0 = G0 + Cref[unif_idx].mean(0)
        k = Gp0.shape[0]
        lam = ratio * np.trace(Gp0) / k
        Ainv = np.linalg.inv(Gp0 + lam * np.eye(k))
        return np.array([np.trace(Ainv @ C[j]) for j in rank_idx]), lam

    G0, C = G0_t.mean(0), C_t.mean(0)
    out = dict(sweep={})
    for ratio in ratios:
        g, lam = gains(G0, C, C, ratio)
        out["sweep"][str(ratio)] = dict(lambda_=float(lam), spearman=spearman(proxy_g, g),
                                       auc=quartile_auc(proxy_g, g))
    g, lam = gains(G0, C, C, pre["lambda_ratio"])
    g1, _ = gains(G0, Ch1_t.mean(0), C, pre["lambda_ratio"])
    g2, _ = gains(G0, Ch2_t.mean(0), C, pre["lambda_ratio"])
    rho, auc, rel = spearman(proxy_g, g), quartile_auc(proxy_g, g), spearman(g1, g2)
    T = G0_t.shape[0]
    boot = []
    for _ in range(n_boot if T > 1 else 0):
        idx = rng.integers(0, T, T)
        gb, _ = gains(G0_t[idx].mean(0), C_t[idx].mean(0), C_t[idx].mean(0), pre["lambda_ratio"])
        boot.append(spearman(proxy_g, gb))
    if not np.isfinite(rel) or rel < pre["min_reliability"]:
        verdict = "undetermined"
    elif rho >= pre["pass_spearman"] and auc >= pre["pass_auc"]:
        verdict = "pass"
    else:
        verdict = "fail"
    out.update(lambda_=float(lam), gains=g.tolist(), spearman=rho, auc=auc, split_half_reliability=rel,
               spearman_boot_ci=[float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))] if boot else None,
               gain_cv=float(g.std() / abs(g.mean())) if g.mean() != 0 else float("nan"),
               cue_to_ctx_trace=float(np.trace(C[unif_idx].mean(0)) / np.trace(G0)), verdict=verdict)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", help=".jsonl (field --text_field) or .txt, one document per line")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--n_texts", type=int, default=8)
    ap.add_argument("--add_bos", type=int, default=1, help="prepend BOS to the context (match training)")
    ap.add_argument("--synthetic_texts", type=int, default=0, help="smoke test only: random token contexts")
    ap.add_argument("--synthetic_vocab", type=int, default=1000)
    ap.add_argument("--cue_text")
    ap.add_argument("--cue_token", help="single token repeated b times, e.g. <|eot_id|>")
    ap.add_argument("--cue_ids", help="comma-separated token ids (overrides cue_text/cue_token)")
    ap.add_argument("--L", type=int, default=65536)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--a0", type=int, default=8182)
    ap.add_argument("--b", type=int, default=10)
    ap.add_argument("--w_cue", type=float, default=0.5)
    ap.add_argument("--set_pi", type=int, default=1, help="1: apply linear PI factor s to the config")
    ap.add_argument("--layers", default="4,16,28")
    ap.add_argument("--modules", default="q_proj,k_proj")
    ap.add_argument("--dirs_per_module", type=int, default=8)
    ap.add_argument("--eps_rel", type=float, default=1e-3, help="perturbation size relative to ||W||_F")
    ap.add_argument("--n_rank_offsets", type=int, default=32)
    ap.add_argument("--n_uniform_offsets", type=int, default=16)
    ap.add_argument("--m_ctx", type=int, default=4)
    ap.add_argument("--m_cue", type=int, default=256)
    ap.add_argument("--lambda_ratio", type=float, default=1e-2)
    ap.add_argument("--lambda_sweep", default="1e-3,1e-2,1e-1")
    ap.add_argument("--min_reliability", type=float, default=0.5)
    ap.add_argument("--pass_spearman", type=float, default=0.5)
    ap.add_argument("--pass_auc", type=float, default=0.75)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--fd_check", type=int, default=1, help="1: repeat text 0 / first 2 directions at eps/2")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64", "bfloat16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gate2_result.json")
    return ap.parse_args()


def main():
    A = parse_args()
    t0 = time.time()
    torch.manual_seed(A.seed)
    rng = np.random.default_rng(A.seed)
    gen_dir = torch.Generator().manual_seed(A.seed)
    gen_lab = torch.Generator(device=A.device).manual_seed(A.seed + 1)

    config = AutoConfig.from_pretrained(A.model)
    if A.set_pi:
        set_linear_pi(config, A.s)
    dtype = dict(float32=torch.float32, float64=torch.float64, bfloat16=torch.bfloat16)[A.dtype]
    model = AutoModelForCausalLM.from_pretrained(A.model, config=config, dtype=dtype).to(A.device).eval()
    need_tok = not (A.synthetic_texts and A.cue_ids)
    tok = AutoTokenizer.from_pretrained(A.model) if need_tok else None
    texts = load_texts(A, tok, rng)
    cue = cue_ids(A, tok)
    a0, b = A.a0, A.b
    assert b >= 2, "b >= 2 required: with b = 1 there is no Delta-dependent prediction (Section 2)"

    proxy = Proxy(A.L, A.s, rope_theta(config), head_dim(config), a0, b, A.w_cue)
    f0, v0 = proxy.evaluate(proxy.uniform())
    g_all = proxy.grad(v0)
    Dmax = proxy.Dmax
    r_off = rank_offsets(Dmax, a0, A.n_rank_offsets)
    u_off = np.sort(rng.integers(0, Dmax + 1, A.n_uniform_offsets))
    offsets = np.concatenate([r_off, u_off])
    rank_idx, unif_idx = np.arange(len(r_off)), np.arange(len(r_off), len(offsets))
    proxy_g = g_all[r_off]

    layers = [int(x) for x in A.layers.split(",")]
    modules = A.modules.split(",")
    dirs = build_directions(model, layers, modules, A.dirs_per_module, gen_dir)
    W0 = {(d["layer"], d["module"]): d["W"].detach().clone() for d in dirs}
    eps_of = {key: A.eps_rel * float(W.double().norm()) for key, W in W0.items()}
    k = len(dirs)
    n_ctx_fwd = len(texts) * (2 * k + 1)
    print(f"k={k} directions, {len(offsets)} offsets ({len(r_off)} ranked + {len(u_off)} uniform), "
          f"{len(texts)} texts: {n_ctx_fwd} context forwards, {n_ctx_fwd * len(offsets)} cue forwards")

    w_ctx = np.ones(a0)
    w_ctx[a0 - 1] = A.w_cue
    w_cue = np.full(b - 1, A.w_cue)
    h = A.m_cue // 2
    G0_t, C_t, Ch1_t, Ch2_t, ptrue, fd = [], [], [], [], [], None
    for t, ids in enumerate(texts):
        tt = time.time()
        h_ctx, cache = ctx_forward(model, ids)
        lab_ctx = sample_labels(model, h_ctx, A.m_ctx, gen_lab, A.chunk)
        lab_cue, pt = [], []
        for D in offsets:
            h_cue = cue_forward(model, cue, cache, a0, int(D))[: b - 1]
            lab_cue.append(sample_labels(model, h_cue, A.m_cue, gen_lab, A.chunk))
            pt.append(true_target_prob(model, h_cue, cue[1:]))
        del cache, h_ctx
        ptrue.append(np.stack(pt))
        S_ctx, S_cue = directional_scores(model, dirs, eps_of, W0, ids, cue, offsets, lab_ctx, lab_cue, A.chunk)
        G0_t.append(ggn_from_scores(S_ctx, w_ctx, A.m_ctx))
        C_t.append(np.stack([ggn_from_scores(S_cue[j], w_cue, A.m_cue) for j in range(len(offsets))]))
        Ch1_t.append(np.stack([ggn_from_scores(S_cue[j][:, :, :h], w_cue, h) for j in range(len(offsets))]))
        Ch2_t.append(np.stack([ggn_from_scores(S_cue[j][:, :, h:], w_cue, A.m_cue - h) for j in range(len(offsets))]))
        if A.fd_check and t == 0:
            sub = dirs[:2]
            _, S_half = directional_scores(model, sub, eps_of, W0, ids, cue, offsets, lab_ctx, lab_cue, A.chunk, 0.5)
            ref = S_cue[:, :2]
            fd = float(np.linalg.norm(S_half - ref) / max(np.linalg.norm(ref), 1e-300))
            print(f"finite-difference check (eps vs eps/2, cue scores): relative diff = {fd:.3e}")
        print(f"text {t + 1}/{len(texts)} done in {time.time() - tt:.1f}s")

    pre = dict(lambda_ratio=A.lambda_ratio, min_reliability=A.min_reliability,
               pass_spearman=A.pass_spearman, pass_auc=A.pass_auc)
    ratios = [float(x) for x in A.lambda_sweep.split(",")]
    res = analyze(np.stack(G0_t), np.stack(C_t), np.stack(Ch1_t), np.stack(Ch2_t),
                  rank_idx, unif_idx, proxy_g, ratios, pre, A.n_boot, rng)
    P = np.stack(ptrue)  # (texts, offsets, b-1)
    res.update(config=vars(A), preregistered=pre, k=k, rank_offsets=r_off.tolist(), uniform_offsets=u_off.tolist(),
               proxy_g=proxy_g.tolist(), proxy_r=proxy.r, fd_relative_diff=fd,
               cue_true_target_prob_mean=float(P.mean()), cue_true_target_prob_min=float(P.min()),
               cue_true_target_prob_by_offset=P.mean(axis=(0, 2))[rank_idx].tolist(), seconds=time.time() - t0)
    print(json.dumps({key: res[key] for key in ("verdict", "spearman", "auc", "split_half_reliability",
                                                 "spearman_boot_ci", "gain_cv", "cue_to_ctx_trace",
                                                 "cue_true_target_prob_mean", "fd_relative_diff")}, indent=1))
    print("lambda sweep:", json.dumps(res["sweep"]))
    with open(A.out, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
