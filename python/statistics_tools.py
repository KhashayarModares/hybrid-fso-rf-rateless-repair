"""Failure-aware empirical deadlines and paired uncertainty estimates."""
import math
import numpy as np
from scipy.stats import beta, binom


def quantile(times, p=.99):
    if len(times) == 0:
        raise ValueError('No samples')
    return float(np.partition(times,max(0,math.ceil(p*len(times))-1))[max(0,math.ceil(p*len(times))-1)])


def binomial_interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k < n else 1.]


def quantile_interval(times,p=.99):
    ordered = np.sort(times)
    n = len(times)
    low = int(binom.ppf(.025,n,p))-1
    high = int(binom.ppf(.975,n,p))
    # With too few tail observations a finite upper confidence endpoint is
    # unavailable. Clamping to the sample maximum falsely asserts confidence.
    return [float(ordered[low]) if low >= 0 else 0.,
            float(ordered[high]) if high < n else math.inf]


def paired_gain(base,proposed,reps=1000,seed=801):
    if len(base) != len(proposed):
        raise ValueError('Paired samples must have the same seeds')
    b,q = quantile(base),quantile(proposed)
    estimate = 100*(1-q/b) if math.isfinite(b) and math.isfinite(q) else None
    if estimate is None:
        return estimate,None
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(reps):
        indices = rng.integers(0,len(base),len(base))
        b,q = quantile(base[indices]),quantile(proposed[indices])
        if not math.isfinite(b) or not math.isfinite(q):
            return estimate,None
        bootstrap.append(100*(1-q/b))
    return estimate,list(map(float,np.quantile(bootstrap,[.025,.975])))


def finite_json(value):
    if isinstance(value,dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):
        return None
    return value


def paired_outage_reduction(base, proposed, deadline):
    """Conservative 95% paired interval at a frozen deadline, in percentage points.

    Let b/p be the two discordance probabilities. The reduction is b-p.
    Exact 97.5% binomial intervals for each give a >=95% difference interval
    by the union bound, regardless of their dependence. Unlike rare-event
    bootstrap intervals, zero discordant observations do not force zero width.
    """
    if len(base) != len(proposed) or len(base) == 0 or not math.isfinite(deadline):
        raise ValueError('Need paired samples and a finite frozen deadline')
    b, p = np.asarray(base) > deadline, np.asarray(proposed) > deadline
    groups = 2*b.astype(int) + p.astype(int)
    counts = np.bincount(groups, minlength=4)
    gain = 100*float(b.mean()-p.mean())
    def interval(k):
        n = len(base)
        return (float(beta.ppf(.0125,k,n-k+1)) if k else 0.,
                float(beta.ppf(.9875,k+1,n-k)) if k<n else 1.)
    b_low,b_high = interval(counts[2])
    p_low,p_high = interval(counts[1])
    return gain,[100*(b_low-p_high),100*(b_high-p_low)]
