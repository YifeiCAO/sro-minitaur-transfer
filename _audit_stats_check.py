import numpy as np
from scipy import stats

n = 104
df = n - 1

def invert(mean_d, p, frac, label):
    t = stats.t.ppf(1 - p/2, df)
    dz = t / np.sqrt(n)
    sd_d = abs(mean_d) / dz
    sem = sd_d / np.sqrt(n)
    phi = stats.norm.cdf(dz)
    print(f"== {label} ==")
    print(f"  reported: mean={mean_d}, p={p:.0e}, frac<0={frac}")
    print(f"  implied |t|={t:.3f}, Cohen's dz={dz:.4f}")
    print(f"  implied sd(diff)={sd_d:.5f}, SEM={sem:.6f}")
    print(f"  if diffs~Normal => frac(real<shuf)=Phi(dz)={phi:.3f}")
    k = round(frac * n)
    psign = stats.binomtest(k, n, 0.5, alternative='greater').pvalue
    print(f"  sign test {k}/{n} below 0: one-sided p={psign:.2e}")
    print()

invert(-0.0118, 7e-20, 0.85, "kirby -> discount_titrate")
invert(-0.0059, 6e-6, 0.73, "directed_forgetting -> recent_probes")

# Simulate: is p=7e-20 plausible for a genuine tiny-but-consistent paired effect?
# Build per-person diffs with dz=0.99 (the implied value), see resulting t/p across draws.
print("== simulation: genuine effect with implied dz ~0.99, n=104 ==")
rng = np.random.RandomState(0)
ps = []
for _ in range(2000):
    d = rng.normal(-0.0118, 0.0118/0.99, size=n)
    ps.append(stats.ttest_1samp(d, 0).pvalue)
ps = np.array(ps)
print(f"  median p={np.median(ps):.1e}, 5th pct={np.percentile(ps,5):.1e}, 95th={np.percentile(ps,95):.1e}")
print(f"  frac of sims with p<1e-15: {(ps<1e-15).mean():.2f}")
