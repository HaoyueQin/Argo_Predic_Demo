#!/usr/bin/env python3
"""
分析两个 Argoverse 数据集的构成差异。

用法：
    python data_diff.py <dir1> <dir2> [--sample N] [--city CITY]

示例：
    python data_diff.py train/data_60k val/data
    python data_diff.py sampled_val_2000/ val/data --sample 500
"""

import os, sys, random, json
from collections import defaultdict
import numpy as np

TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME = 0, 1, 2, 3, 4, 5


def parse_csv(filepath):
    """Parse one Argoverse CSV, return basic stats dict or None."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # skip header
    except Exception:
        return None
    if not lines:
        return None

    info = defaultdict(list)
    id2info = {}
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        ts, tid, otype = float(parts[0]), parts[1], parts[2]
        x, y = float(parts[3]), float(parts[4])
        city = parts[5]
        if otype in ('AV', 'AGENT'):
            tid = otype
        if tid not in id2info:
            id2info[tid] = []
        id2info[tid].append((ts, x, y, otype, city))

    if 'AGENT' not in id2info:
        return None

    agent = id2info['AGENT']
    if len(agent) < 20:
        return None

    # Agent trajectory stats
    hist = agent[:20]
    fut = agent[20:] if len(agent) >= 50 else []
    n_future = len(fut)

    # Speed (m/s, based on 0.1s intervals)
    # Argoverse 1 TIMESTAMPs are microseconds (~1e5 apart); normalise to
    # seconds so speeds are real m/s regardless of the source unit.
    speeds = []
    headings = []
    dt_list = []
    for i in range(1, len(agent)):
        dt = agent[i][0] - agent[i-1][0]
        if dt > 0:
            dt_list.append(dt)
    if dt_list:
        median_dt = float(np.median(dt_list))
        unit_scale = 1e-6 if median_dt > 1.0 else 1.0
    else:
        unit_scale = 1.0
    for i in range(1, len(agent)):
        dx = agent[i][1] - agent[i-1][1]
        dy = agent[i][2] - agent[i-1][2]
        dt = (agent[i][0] - agent[i-1][0]) * unit_scale
        if dt > 0:
            spd = np.sqrt(dx*dx + dy*dy) / dt
            speeds.append(spd)
            if spd > 0.1:
                headings.append(np.arctan2(dy, dx))

    # Total displacement
    hist_disp = np.sqrt(
        (hist[-1][1] - hist[0][1])**2 +
        (hist[-1][2] - hist[0][2])**2
    ) if len(hist) >= 2 else 0

    fut_disp = np.sqrt(
        (fut[-1][1] - hist[-1][1])**2 +
        (fut[-1][2] - hist[-1][2])**2
    ) if n_future >= 2 else 0

    # Heading change (last 1s of history vs first 1s of future)
    # Note: arctan2(dy, dx) — y argument comes first.
    h_hist = np.arctan2(
        hist[-1][2] - hist[-6][2],
        hist[-1][1] - hist[-6][1]
    ) if len(hist) >= 6 else 0
    h_fut = np.arctan2(
        fut[9][2] - fut[4][2],
        fut[9][1] - fut[4][1]
    ) if n_future >= 10 else 0
    heading_change = abs(h_fut - h_hist)  # abs() already lands in [0, pi]

    # Other agents count
    n_others = sum(1 for k in id2info if k not in ('AGENT', 'AV'))

    return {
        'file': os.path.basename(filepath),
        'city': agent[0][4] if city_label(str(agent[0][4])) else 'UNK',
        'n_tracks': len(id2info),
        'n_others': n_others,
        'n_agent_frames': len(agent),
        'n_future': n_future,
        'hist_disp': round(hist_disp, 2),
        'fut_disp': round(fut_disp, 2),
        'speed_mean': round(np.mean(speeds), 2) if speeds else 0,
        'speed_max': round(np.max(speeds), 2) if speeds else 0,
        'heading_change': round(np.degrees(heading_change), 1),
        'is_turning': heading_change > np.radians(30),
        'is_low_speed': (np.mean(speeds) if speeds else 99) < 1.0,
    }


def city_label(raw):
    """Map city name to short label."""
    if 'PIT' in raw or 'Pittsburgh' in raw:
        return 'PIT'
    if 'MIA' in raw or 'Miami' in raw:
        return 'MIA'
    return raw[:3] if raw else 'UNK'


def summarize(entries, label):
    """Print summary statistics for a dataset."""
    if not entries:
        print(f'\n{label}: NO VALID ENTRIES\n')
        return

    speeds = [e['speed_mean'] for e in entries if e['speed_mean'] > 0]
    fut_disp = [e['fut_disp'] for e in entries]
    hist_disp = [e['hist_disp'] for e in entries]
    hc = [e['heading_change'] for e in entries]
    n_others = [e['n_others'] for e in entries]
    n_future = [e['n_future'] for e in entries]

    cities = defaultdict(int)
    for e in entries:
        cities[e['city']] += 1

    turning_pct = sum(1 for e in entries if e['is_turning']) / len(entries) * 100
    low_speed_pct = sum(1 for e in entries if e['is_low_speed']) / len(entries) * 100
    full_fut_pct = sum(1 for e in entries if e['n_future'] >= 30) / len(entries) * 100

    print(f'\n{"="*60}')
    print(f'  {label}  (n={len(entries)})')
    print(f'{"="*60}')
    print(f'  City distribution:   {dict(cities)}')
    print(f'  Full future (30fr):  {full_fut_pct:.1f}%')
    print(f'  Turning ratio:       {turning_pct:.1f}% (>30° heading change)')
    print(f'  Low-speed ratio:     {low_speed_pct:.1f}% (<1 m/s avg)')
    print(f'  Tracks per scene:    mean={np.mean(n_others):.1f}  std={np.std(n_others):.1f}  max={np.max(n_others)}')
    print(f'  Speed (m/s):         mean={np.mean(speeds):.2f}  std={np.std(speeds):.2f}  max={np.max(speeds):.1f}')
    print(f'  Hist displ (m):      mean={np.mean(hist_disp):.2f}  std={np.std(hist_disp):.2f}')
    print(f'  Future displ (m):    mean={np.mean(fut_disp):.2f}  std={np.std(fut_disp):.2f}')
    print(f'  Heading change (°):  mean={np.mean(hc):.1f}  std={np.std(hc):.1f}')
    print()

    return {
        'n': len(entries),
        'speed_mean': np.mean(speeds),
        'fut_disp_mean': np.mean(fut_disp),
        'hist_disp_mean': np.mean(hist_disp),
        'hc_mean': np.mean(hc),
        'n_others_mean': np.mean(n_others),
        'turning_pct': turning_pct,
        'low_speed_pct': low_speed_pct,
        'full_fut_pct': full_fut_pct,
        'cities': dict(cities),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python data_diff.py <dir1> <dir2> [--sample N] [--city CITY]")
        sys.exit(1)

    dir1 = sys.argv[1]
    dir2 = sys.argv[2]

    sample_n = None
    city_filter = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--sample':
            sample_n = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == '--city':
            city_filter = sys.argv[i+1]
            i += 2
        else:
            i += 1

    random.seed(42)

    for d, label in [(dir1, 'Dataset 1'), (dir2, 'Dataset 2')]:
        files = [os.path.join(d, f) for f in os.listdir(d)
                 if f.endswith('.csv') and not f.startswith('.')]
        if sample_n and len(files) > sample_n:
            files = random.sample(files, sample_n)

        entries = []
        for fp in files:
            e = parse_csv(fp)
            if e and (city_filter is None or e['city'] == city_filter):
                entries.append(e)

        summarize(entries, f'{label} ({d})')


if __name__ == '__main__':
    main()
