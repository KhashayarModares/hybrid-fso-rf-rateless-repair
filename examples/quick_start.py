"""Quick start: simulate a few hundred messages per scheme with the settings of the letter.

usage: python examples/quick_start.py [--messages 200] [--eps 0.59] [--seed 1]

All schemes see the same messages and channel realizations (same seeds). With a few hundred
messages the 99th percentile rests on very few samples, so this is a check that the code runs
and behaves sensibly, not a reproduction of the paper (which uses 10^6 messages per scheme).
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from channel import rf_erasure                 # noqa: E402
from coding import Config                      # noqa: E402
from feedback_options import ARCHITECTURES     # noqa: E402
from model import simulate                     # noqa: E402

SIFS_S = 3e-6  # 60 GHz short interframe space, IEEE Std 802.11-2024, Table 20-30


def schemes(eps):
    """The five schemes of the letter, with the frozen (W, theta) of each architecture."""
    base = Config(eps_fso=eps)
    lossy = {'feedback_loss_probability': rf_erasure('clear')}  # reports fail like RF data packets
    return {
        'Soft switching': (base, 'pooling'),
        'Open-loop assignment': (base, 'open_loop'),
        'Proposed, half-duplex': (replace(base, **ARCHITECTURES['hd_real'], window=512,
                                          state_threshold=0.25, turnaround_s=SIFS_S, **lossy), 'bitmap'),
        'Proposed, dedicated': (replace(base, **ARCHITECTURES['ded_real'], window=8, state_threshold=0.6,
                                        feedback_rate_bps=200e6, **lossy), 'bitmap'),
        'Proposed, ideal': (replace(base, **ARCHITECTURES['ded_ideal'], window=16,
                                    state_threshold=0.6), 'bitmap'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--messages', type=int, default=200, help='messages per scheme')
    p.add_argument('--eps', type=float, default=0.59, help='stationary FSO erasure probability')
    p.add_argument('--seed', type=int, default=1, help='seed of the first message')
    args = p.parse_args()
    print(f'{args.messages} messages per scheme, stationary FSO erasure probability {args.eps}')
    print('completion time in slots (one slot = 0.8925 us)\n')
    print(f"{'scheme':24s}{'mean':>8s}{'99th pct.':>11s}{'failed':>8s}{'time (s)':>10s}")
    for name, (cfg, method) in schemes(args.eps).items():
        start = time.perf_counter()
        t = []
        for seed in range(args.seed, args.seed + args.messages):
            result = simulate(cfg, method, seed)
            t.append(result['receiver_completion_slots'] if result['success'] else np.inf)
        t = np.array(t)
        done = np.isfinite(t)
        print(f'{name:24s}{t[done].mean():8.1f}{np.quantile(t, .99, method="higher"):11.1f}'
              f'{int((~done).sum()):8d}{time.perf_counter() - start:10.1f}')


if __name__ == '__main__':
    main()
