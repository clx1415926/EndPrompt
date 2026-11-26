"""Gate 1: D-optimal integer skip table for ET under the weighted RoPE phase-feature proxy.

Implements Sections 3-5 and 7 of ET_skip采样算法_最终方案.md:
  - exact pair counts m(d), trapezoid kernel n(u)            (Section 3)
  - reference SVD with c_ref(d)=(L-1-d)_+, X_p = diag(sqrt(c_p/c_ref)) U_r   (Section 4)
  - certified interval [f_lo, f_hi] for f*, efficiency verdict for uniform p0 (Section 5)
  - entropy path under f(p) >= f_hi + r*log(eta), final recheck   (Section 5, steps 4-5)

The optimizer for p* is the multiplicative D-optimal update. The bounds do not depend on the
optimizer: f_lo is a value actually attained, f_hi = min_k f(p_k) + G_FW(p_k) by concavity.

Usage:
  python gate1_doptimal.py --a0 8182 --b 10 --wcue 0.5 --solve_eta --out_prefix gate1
"""
import argparse
import json
import resource
import sys
import time

import numpy as np

EPS64 = np.finfo(np.float64).eps


def fftconv(x, y):
    n = len(x) + len(y) - 1
    N = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(y, N), N)[:n]


def peak_rss_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


def entropy(p):
    q = p[p > 0]
    return float(-np.sum(q * np.log(q)))


class Proxy:
    """Weighted RoPE phase-feature Gram proxy for one fixed (a0, b) configuration."""

    def __init__(self, L, s, base, D, a0, b, w_cue, rcond_mult=1.0):
        assert a0 >= 1 and b >= 2 and a0 + b <= L and w_cue > 0
        self.L, self.a0, self.b, self.w = L, a0, b, w_cue
        d = np.arange(L, dtype=np.float64)
        omega = base ** (-2.0 * np.arange(D // 2) / D) / s
        G = np.concatenate([np.cos(np.outer(d, omega)), np.sin(np.outer(d, omega))], axis=1)

        # rows d = 0..L-2 have c_ref > 0; row d = L-1 has zero count on both sides
        self.rows = L - 1
        c_ref = np.maximum(L - 1 - d, 0.0)[: self.rows]
        A = np.sqrt(c_ref)[:, None] * G[: self.rows]
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        self.svd_residual = float(np.linalg.norm(A - (U * S) @ Vt) / np.linalg.norm(A))
        tol = S[0] * max(A.shape) * EPS64 * rcond_mult
        self.r = int((S > tol).sum())
        self.sv_kept_min, self.sv_dropped_max = float(S[self.r - 1]), float(S[self.r]) if self.r < len(S) else 0.0
        self.Ur = U[:, : self.r]
        self.c_ref = c_ref
        self.gt = self.Ur / np.sqrt(c_ref)[:, None]  # rows are g~(d) = Q g(d)
        del G, A, U, Vt

        self.m = (np.maximum(a0 - 1 - d, 0) + w_cue * (d <= a0 - 1) + w_cue * np.maximum(b - 1 - d, 0))[: self.rows]
        u = np.arange(a0 + b - 1, dtype=np.float64)
        n = np.maximum(0, np.minimum(a0 + b - 2, u + a0 - 1) - np.maximum(a0, u) + 1)
        n[0] = 0.0
        self.n = n
        self.Dmax = L - a0 - b
        self.K = self.Dmax + 1
        self.min_c = 0.0

    def coverage(self, p):
        cross = fftconv(p, self.n)[: self.rows]
        c = self.m.copy()
        c[: len(cross)] += self.w * cross
        self.min_c = min(self.min_c, float(c.min()))  # FFT round-off diagnostic
        return np.maximum(c, 0.0)

    def evaluate(self, p):
        """Return f(p) = 2 sum log sigma_i(X_p) and v(d) = g~(d)^T M_p^{-1} g~(d)."""
        c = self.coverage(p)
        X = np.sqrt(c / self.c_ref)[:, None] * self.Ur
        _, sx, Wt = np.linalg.svd(X, full_matrices=False)
        if sx[-1] <= sx[0] * max(X.shape) * EPS64:
            return -np.inf, None
        T = self.gt @ (Wt.T / sx)
        v = np.einsum("ij,ij->i", T, T)
        return float(2.0 * np.sum(np.log(sx))), v

    def grad(self, v):
        """g_Delta = w * sum_u n(u) v(Delta+u) for Delta = 0..Dmax."""
        full = fftconv(v[::-1], self.n)
        return self.w * full[(len(v) - 1) - np.arange(self.K)]

    def fw_gap(self, p, g):
        return float(g.max() - p @ g)

    def uniform(self):
        return np.full(self.K, 1.0 / self.K)


def solve_opt(px, p_start, max_iters, rel_tol=1e-3):
    """Certified interval [f_lo, f_hi] for f* = max_p f(p)."""
    p = p_start.copy()
    f_lo, f_hi = -np.inf, np.inf
    for it in range(1, max_iters + 1):
        f, v = px.evaluate(p)
        g = px.grad(v)
        f_lo = max(f_lo, f)
        f_hi = min(f_hi, f + px.fw_gap(p, g))
        if f_hi - f_lo <= px.r * rel_tol:
            return p, f_lo, f_hi, it, True
        p = p * g / (p @ g)
        p /= p.sum()
    return p, f_lo, f_hi, max_iters, False


def eff_interval(px, f, f_lo, f_hi):
    return float(np.exp((f - f_hi) / px.r)), float(np.exp((f - f_lo) / px.r))


def solve_entropy_tau(px, tau, p_init, inner):
    """Maximize f(p) + tau*H(p) by exponentiated gradient with Armijo backtracking."""
    lp = np.log(np.maximum(p_init, 1e-300))
    f, v = px.evaluate(np.exp(lp))
    obj = f - tau * np.sum(np.exp(lp) * lp)
    alpha = 1.0 / tau
    for _ in range(inner):
        g = px.grad(v)
        step = alpha
        while True:
            nlp = (1 - step * tau) * lp + step * g
            nlp -= nlp.max()
            nlp -= np.log(np.exp(nlp).sum())
            nf, nv = px.evaluate(np.exp(nlp))
            nobj = nf - tau * np.sum(np.exp(nlp) * nlp)
            if nobj >= obj - 1e-12 or step < 1e-8:
                break
            step *= 0.5
        done = abs(nobj - obj) < 1e-9
        lp, f, v, obj = nlp, nf, nv, nobj
        if done:
            break
        alpha = min(step * 2, 1.0 / tau)
    return np.exp(lp)


def entropy_path(px, eta, f_hi, p_start, bisect, inner):
    """Highest-entropy candidate on the tau path satisfying f(p) >= f_hi + r*log(eta)."""
    f_req = f_hi + px.r * np.log(eta)
    lo_t, hi_t = 1e-3, 1e3
    best, pinit = None, p_start.copy()
    for _ in range(bisect):
        tau = np.sqrt(lo_t * hi_t)
        pt = solve_entropy_tau(px, tau, pinit, inner)
        ft, _ = px.evaluate(pt)
        if ft >= f_req:
            if best is None or entropy(pt) > entropy(best[1]):
                best = (tau, pt)
            lo_t = tau
        else:
            hi_t = tau
        pinit = pt
    return best


def final_table(px, p, f_lo, f_hi):
    """Step 5: normalize, rebuild CDF, recompute efficiency interval and entropy."""
    norm_err = float(abs(p.sum() - 1.0))
    p = p / p.sum()
    cdf = np.cumsum(p)
    cdf_err = float(abs(cdf[-1] - 1.0))
    cdf[-1] = 1.0
    f, _ = px.evaluate(p)
    lo, hi = eff_interval(px, f, f_lo, f_hi)
    return p, cdf, dict(norm_err=norm_err, cdf_err=cdf_err, f=f, eff_lo=lo, eff_hi=hi, entropy=entropy(p))


def sample_offsets(cdf, n, rng):
    """Section 7: one uniform draw per sample, searchsorted(side='right')."""
    return np.searchsorted(cdf, rng.random(n), side="right")


def bin_summary(px, p):
    K, a0, Dmax = px.K, px.a0, px.Dmax
    spans = [(0, 1024), (1024, a0), (a0, 2 * a0), (Dmax - a0 + 1, Dmax - 1023), (Dmax - 1023, K)]
    return [dict(lo=lo, hi=hi, mass=float(p[lo:hi].sum()), uniform=(hi - lo) / K) for lo, hi in spans]


def verdict_uniform(px, eta, max_iters):
    """Steps 1-3 for one proxy. Returns summary dict and (p_opt, f_lo, f_hi)."""
    p0 = px.uniform()
    f0, v0 = px.evaluate(p0)
    gap0 = px.fw_gap(p0, px.grad(v0))
    out = dict(r=px.r, K=px.K, f_p0=f0, fw_gap_p0=gap0, step1_lower_bound=float(np.exp(-gap0 / px.r)))
    if out["step1_lower_bound"] >= eta:
        out.update(verdict="uniform (step 1)", eff_lo=out["step1_lower_bound"], eff_hi=1.0)
        return out, None
    p_opt, f_lo, f_hi, iters, conv = solve_opt(px, p0, max_iters)
    lo, hi = eff_interval(px, f0, f_lo, f_hi)
    if lo >= eta:
        verdict = "uniform"
    elif hi < eta:
        verdict = "non-uniform (go to step 4)"
    else:
        verdict = "undetermined"
    out.update(f_lo=f_lo, f_hi=f_hi, iters=iters, converged=conv, eff_lo=lo, eff_hi=hi, verdict=verdict)
    return out, (p_opt, f_lo, f_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=65536)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--base", type=float, default=500000.0)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--a0", type=int, default=8182)
    ap.add_argument("--b", type=int, default=10)
    ap.add_argument("--wcue", type=float, default=0.5)
    ap.add_argument("--eta", type=float, default=0.95)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--solve_eta", action="store_true")
    ap.add_argument("--bisect", type=int, default=14)
    ap.add_argument("--inner", type=int, default=400)
    ap.add_argument("--rank_sensitivity", action="store_true", help="rerun steps 1-3 with rcond x0.1 and x10")
    ap.add_argument("--out_prefix", default="")
    A = ap.parse_args()
    t0 = time.time()

    px = Proxy(A.L, A.s, A.base, A.D, A.a0, A.b, A.wcue)
    summary = dict(config=vars(A), svd_residual=px.svd_residual,
                   sv_kept_min=px.sv_kept_min, sv_dropped_max=px.sv_dropped_max)
    res, opt = verdict_uniform(px, A.eta, A.iters)
    summary["uniform"] = res
    print(json.dumps(res, indent=1))

    if A.rank_sensitivity:
        summary["rank_sensitivity"] = {}
        for mult in (0.1, 10.0):
            alt = Proxy(A.L, A.s, A.base, A.D, A.a0, A.b, A.wcue, rcond_mult=mult)
            r_alt, _ = verdict_uniform(alt, A.eta, A.iters)
            summary["rank_sensitivity"][str(mult)] = {k: r_alt[k] for k in ("r", "eff_lo", "eff_hi", "verdict")}
            print(f"rcond x{mult}:", summary["rank_sensitivity"][str(mult)])
            del alt

    if opt is not None:
        p_opt, f_lo, f_hi = opt
        summary["p_opt_bins"] = bin_summary(px, p_opt / p_opt.sum())
        # the D-optimal table itself (no eta): certified eff in [exp((f-f_hi)/r), 1]
        p_star, cdf_star, chk_star = final_table(px, p_opt, f_lo, f_hi)
        chk_star.update(entropy_uniform=float(np.log(px.K)),
                        density_ratio_max=float(p_star.max() * px.K), argmax=int(p_star.argmax()))
        summary["p_star"] = chk_star
        if A.out_prefix:
            np.save(A.out_prefix + "_p_star.npy", p_star)
            np.save(A.out_prefix + "_p_star_cdf.npy", cdf_star)
        print("p_star:", json.dumps({k: v for k, v in chk_star.items() if k != "bins"}))
        if A.solve_eta and res["verdict"].startswith("non-uniform"):
            best = entropy_path(px, A.eta, f_hi, px.uniform(), A.bisect, A.inner)
            if best is None:
                summary["p_hat"] = dict(status="no candidate passed the conservative constraint")
            else:
                tau, pt = best
                p_hat, cdf, chk = final_table(px, pt, f_lo, f_hi)
                chk.update(tau=tau, status="pass" if chk["eff_lo"] >= A.eta else "undetermined",
                           entropy_uniform=float(np.log(px.K)),
                           density_ratio_min=float(p_hat.min() * px.K), density_ratio_max=float(p_hat.max() * px.K),
                           bins=bin_summary(px, p_hat))
                summary["p_hat"] = chk
                if A.out_prefix:
                    np.save(A.out_prefix + "_p_hat.npy", p_hat)
                    np.save(A.out_prefix + "_cdf.npy", cdf)
                print(json.dumps({k: v for k, v in chk.items() if k != "bins"}, indent=1))

    summary.update(min_coverage_before_clip=px.min_c, peak_rss_mib=peak_rss_mib(), seconds=time.time() - t0)
    print(f"svd_residual={px.svd_residual:.2e}, peak_rss={summary['peak_rss_mib']:.0f} MiB, "
          f"time={summary['seconds']:.1f}s")
    if A.out_prefix:
        with open(A.out_prefix + "_summary.json", "w") as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
