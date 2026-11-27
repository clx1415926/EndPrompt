"""Offline approximately D-optimal offset distribution q_hat for ET (paper Sec. 3.2 and Appendix C).

  - pair counts c_q(d) = (a-d)_+ + (b-1-d)_+ + sum_Delta q(Delta) kappa(d-Delta), unit target weights
  - whitened RoPE features from the SVD of rows sqrt(L-1-d) psi(d), threshold sigma_1 max(L-1, D) eps64
  - multiplicative update q <- q * gamma / (q^T gamma), started from uniform offsets
  - stop when the concavity upper bound and the best attained log-det differ by at most 1e-3 r

Usage:
  python gate1_doptimal.py --a0 8182 --b 10 --rank_sensitivity --out_prefix gate1_w1
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
    """Whitened RoPE feature information matrix for one fixed (a0, b) configuration."""

    def __init__(self, L, s, base, D, a0, b, rcond_mult=1.0):
        assert a0 >= 1 and b >= 2 and a0 + b <= L
        self.L, self.a0, self.b = L, a0, b
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
        self.gt = self.Ur / np.sqrt(c_ref)[:, None]  # rows are psi~(d) = Sigma_r^{-1} V_r^T psi(d)
        del G, A, U, Vt

        # within-segment pairs: (a-d)_+ + (b-1-d)_+
        self.m = (np.maximum(a0 - d, 0) + np.maximum(b - 1 - d, 0))[: self.rows]
        # kappa(t): cue queries a0..a0+b-2 with a context key at physical distance t
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
        c[: len(cross)] += cross
        self.min_c = min(self.min_c, float(c.min()))  # FFT round-off diagnostic
        return np.maximum(c, 0.0)

    def evaluate(self, p):
        """Return Phi(p) = log det Gamma_p and nu(d) = psi~(d)^T Gamma_p^{-1} psi~(d)."""
        c = self.coverage(p)
        X = np.sqrt(c / self.c_ref)[:, None] * self.Ur
        _, sx, Wt = np.linalg.svd(X, full_matrices=False)
        if sx[-1] <= sx[0] * max(X.shape) * EPS64:
            return -np.inf, None
        T = self.gt @ (Wt.T / sx)
        v = np.einsum("ij,ij->i", T, T)
        return float(2.0 * np.sum(np.log(sx))), v

    def grad(self, v):
        """gamma_Delta = sum_t kappa(t) nu(Delta+t) for Delta = 0..Dmax."""
        full = fftconv(v[::-1], self.n)
        return full[(len(v) - 1) - np.arange(self.K)]

    def fw_gap(self, p, g):
        return float(g.max() - p @ g)

    def uniform(self):
        return np.full(self.K, 1.0 / self.K)


def solve_opt(px, p_start, max_iters, rel_tol=1e-3):
    """Bounds [f_lo, f_hi] on Phi* = max_p Phi(p); returns the best attained iterate."""
    p = p_start.copy()
    p_best, f_lo, f_hi = p.copy(), -np.inf, np.inf
    for it in range(1, max_iters + 1):
        f, v = px.evaluate(p)
        g = px.grad(v)
        if f > f_lo:
            p_best, f_lo = p.copy(), f
        f_hi = min(f_hi, f + px.fw_gap(p, g))
        if f_hi - f_lo <= px.r * rel_tol:
            return p_best, f_lo, f_hi, it, True
        p = p * g / (p @ g)
        p /= p.sum()
    return p_best, f_lo, f_hi, max_iters, False


def eff_interval(px, f, f_lo, f_hi):
    return float(np.exp((f - f_hi) / px.r)), float(np.exp((f - f_lo) / px.r))


def final_table(px, p, f_lo, f_hi):
    """Normalize q_hat, build its cumulative table, recompute the D-efficiency interval."""
    norm_err = float(abs(p.sum() - 1.0))
    p = p / p.sum()
    cdf = np.cumsum(p)
    cdf_err = float(abs(cdf[-1] - 1.0))
    cdf[-1] = 1.0
    f, _ = px.evaluate(p)
    lo, hi = eff_interval(px, f, f_lo, f_hi)
    return p, cdf, dict(norm_err=norm_err, cdf_err=cdf_err, f=f, eff_lo=lo, eff_hi=hi, entropy=entropy(p))


def sample_offsets(cdf, n, rng):
    """One uniform draw per example, inverse-CDF lookup by binary search."""
    return np.searchsorted(cdf, rng.random(n), side="right")


def bin_summary(px, p):
    K, a0, Dmax = px.K, px.a0, px.Dmax
    spans = [(0, 1024), (1024, a0), (a0, 2 * a0), (Dmax - a0 + 1, Dmax - 1023), (Dmax - 1023, K)]
    return [dict(lo=lo, hi=hi, mass=float(p[lo:hi].sum()), uniform=(hi - lo) / K) for lo, hi in spans]


def design(px, max_iters):
    """Solve for q_hat from uniform offsets; report the D-efficiency of uniform offsets as a diagnostic."""
    p0 = px.uniform()
    f0, _ = px.evaluate(p0)
    p_opt, f_lo, f_hi, iters, conv = solve_opt(px, p0, max_iters)
    lo, hi = eff_interval(px, f0, f_lo, f_hi)
    out = dict(r=px.r, K=px.K, f_p0=f0, f_lo=f_lo, f_hi=f_hi, iters=iters, converged=conv, eff_lo=lo, eff_hi=hi)
    return out, (p_opt, f_lo, f_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=65536)
    ap.add_argument("--s", type=float, default=8.0)
    ap.add_argument("--base", type=float, default=500000.0)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--a0", type=int, default=8182)
    ap.add_argument("--b", type=int, default=10)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--rank_sensitivity", action="store_true", help="rerun with SVD threshold x0.1 and x10")
    ap.add_argument("--out_prefix", default="")
    A = ap.parse_args()
    t0 = time.time()

    px = Proxy(A.L, A.s, A.base, A.D, A.a0, A.b)
    summary = dict(config=vars(A), svd_residual=px.svd_residual,
                   sv_kept_min=px.sv_kept_min, sv_dropped_max=px.sv_dropped_max)
    res, (p_opt, f_lo, f_hi) = design(px, A.iters)
    summary["uniform"] = res
    print(json.dumps(res, indent=1))

    if A.rank_sensitivity:
        summary["rank_sensitivity"] = {}
        for mult in (0.1, 10.0):
            alt = Proxy(A.L, A.s, A.base, A.D, A.a0, A.b, rcond_mult=mult)
            r_alt, _ = design(alt, A.iters)
            summary["rank_sensitivity"][str(mult)] = {k: r_alt[k] for k in ("r", "eff_lo", "eff_hi")}
            print(f"rcond x{mult}:", summary["rank_sensitivity"][str(mult)])
            del alt

    summary["p_opt_bins"] = bin_summary(px, p_opt / p_opt.sum())
    p_star, cdf_star, chk_star = final_table(px, p_opt, f_lo, f_hi)
    chk_star.update(entropy_uniform=float(np.log(px.K)),
                    density_ratio_max=float(p_star.max() * px.K), argmax=int(p_star.argmax()))
    summary["p_star"] = chk_star
    if A.out_prefix:
        np.save(A.out_prefix + "_p_star.npy", p_star)
        np.save(A.out_prefix + "_p_star_cdf.npy", cdf_star)
    print("p_star:", json.dumps(chk_star))

    summary.update(min_coverage_before_clip=px.min_c, peak_rss_mib=peak_rss_mib(), seconds=time.time() - t0)
    print(f"svd_residual={px.svd_residual:.2e}, peak_rss={summary['peak_rss_mib']:.0f} MiB, "
          f"time={summary['seconds']:.1f}s")
    if A.out_prefix:
        with open(A.out_prefix + "_summary.json", "w") as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
