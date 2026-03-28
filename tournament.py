"""
Round-robin tournament runner for TACS PokerBots 2026.

Each pair plays twice (seat swap) to eliminate positional bias.
Results are parsed from engine game logs.
Run with: python3 tournament.py
"""

import os
import re
import subprocess
import sys
import itertools
import json
import tempfile
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
PYTHON    = str(REPO_ROOT / '.venv' / 'bin' / 'python3')
ENGINE    = str(REPO_ROOT / 'engine.py')

# ── Bot registry ─────────────────────────────────────────────────────────────

BOTS = {
    'mc_equity':       './python_skeleton',
    'tight_aggro':     './bot_tight_aggressive',
    'aggro_pressure':  './bot_aggro_pressure',
    'exploiter':       './bot_exploiter',
    'gto_balanced':    './bot_gto_balanced',
    'redraw_hunter':   './bot_redraw_hunter',
}

ROUNDS_PER_MATCH = 100   # reduced from 300 for speed; enough for relative ranking

# ── Config writer ─────────────────────────────────────────────────────────────

def write_config(p1_name, p1_path, p2_name, p2_path, results_dir, rounds):
    cfg = f"""
PLAYER_1_NAME = "{p1_name}"
PLAYER_1_PATH = "{p1_path}"
PLAYER_2_NAME = "{p2_name}"
PLAYER_2_PATH = "{p2_path}"
GAME_LOG_FILENAME = "gamelog"
RESULTS_DIR = "{results_dir}/"
PLAYER_LOG_SIZE_LIMIT = 524288
ENFORCE_GAME_CLOCK = False
STARTING_GAME_CLOCK = 600.0
BUILD_TIMEOUT = 30.0
CONNECT_TIMEOUT = 30.0
NUM_ROUNDS = {rounds}
STARTING_STACK = 400
BIG_BLIND = 2
SMALL_BLIND = 1
PLAYER_TIMEOUT = 120
"""
    config_path = REPO_ROOT / 'config.py'
    config_path.write_text(cfg.strip())


# ── Result parser ─────────────────────────────────────────────────────────────

def parse_result(results_dir, p1_name, p2_name):
    """
    Parse the final bankroll line from the game log.
    Returns (p1_delta, p2_delta) or (None, None) on failure.
    """
    log_path = Path(results_dir) / 'gamelog.txt'
    if not log_path.exists():
        return None, None
    text = log_path.read_text()
    lines = text.strip().splitlines()
    # Last non-empty line: "Final, A (1234), B (-1234)"
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('Final'):
            deltas = {}
            for m in re.finditer(r'(\w+)\s+\((-?\d+)\)', line):
                deltas[m.group(1)] = int(m.group(2))
            d1 = deltas.get(p1_name)
            d2 = deltas.get(p2_name)
            if d1 is not None and d2 is not None:
                return d1, d2
    return None, None


# ── Run one match ─────────────────────────────────────────────────────────────

def run_match(p1_name, p1_path, p2_name, p2_path, match_id):
    results_dir = REPO_ROOT / 'results' / f'match_{match_id}'
    results_dir.mkdir(parents=True, exist_ok=True)

    write_config(p1_name, p1_path, p2_name, p2_path,
                 str(results_dir), ROUNDS_PER_MATCH)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT)   # makes pkrbot.py and config.py findable

    try:
        proc = subprocess.run(
            [PYTHON, ENGINE],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            print(f'  [!] Engine error (rc={proc.returncode}):')
            print(proc.stderr[-800:] if proc.stderr else '(no stderr)')
            return None, None
    except subprocess.TimeoutExpired:
        print(f'  [!] Match timed out: {p1_name} vs {p2_name}')
        return None, None

    return parse_result(str(results_dir), p1_name, p2_name)


# ── Tournament ────────────────────────────────────────────────────────────────

def run_tournament():
    bot_names = list(BOTS.keys())
    pairs     = list(itertools.combinations(bot_names, 2))

    # Each pair plays twice: (A vs B) and (B vs A) to cancel positional bias
    matches = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]

    # Results: {bot_name: {wins, losses, total_delta, matches_played}}
    results = {name: {'wins': 0, 'losses': 0, 'draws': 0,
                      'delta': 0, 'played': 0} for name in bot_names}

    head_to_head = {}   # (a, b): [delta_a, delta_a, ...]

    print(f'\n{"="*60}')
    print(f'TACS PokerBots 2026 — Round Robin Tournament')
    print(f'{len(bot_names)} bots, {len(matches)} matches, {ROUNDS_PER_MATCH} hands each')
    print(f'{"="*60}\n')

    for match_id, (p1, p2) in enumerate(matches, 1):
        print(f'[{match_id:02d}/{len(matches)}] {p1:18s} vs {p2:18s} ... ', end='', flush=True)

        d1, d2 = run_match(p1, BOTS[p1], p2, BOTS[p2], match_id)

        if d1 is None:
            print('FAILED')
            continue

        print(f'{d1:+5d} / {d2:+5d}')

        results[p1]['delta']  += d1
        results[p2]['delta']  += d2
        results[p1]['played'] += 1
        results[p2]['played'] += 1

        if d1 > d2:
            results[p1]['wins']   += 1
            results[p2]['losses'] += 1
        elif d2 > d1:
            results[p2]['wins']   += 1
            results[p1]['losses'] += 1
        else:
            results[p1]['draws'] += 1
            results[p2]['draws'] += 1

        key = tuple(sorted([p1, p2]))
        head_to_head.setdefault(key, {p1: 0, p2: 0})
        head_to_head[key][p1] += d1
        head_to_head[key][p2] += d2

    return results, head_to_head, bot_names


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results, head_to_head, bot_names):
    print(f'\n{"="*60}')
    print('FINAL LEADERBOARD')
    print(f'{"="*60}')
    print(f'{"Rank":<5} {"Bot":<20} {"W":>4} {"L":>4} {"D":>4} {"Total Δ":>10} {"Avg Δ/hand":>12}')
    print('-'*60)

    ranked = sorted(results.items(), key=lambda x: (x[1]['delta'], x[1]['wins']), reverse=True)

    for rank, (name, r) in enumerate(ranked, 1):
        played = r['played']
        hands  = played * ROUNDS_PER_MATCH
        avg    = r['delta'] / hands if hands > 0 else 0
        print(f'{rank:<5} {name:<20} {r["wins"]:>4} {r["losses"]:>4} {r["draws"]:>4} '
              f'{r["delta"]:>+10} {avg:>+12.3f}')

    print(f'\n{"="*60}')
    print('HEAD-TO-HEAD MATRIX (chip delta, row beats column if positive)')
    print(f'{"="*60}')
    # Column headers
    short = {n: n[:8] for n in bot_names}
    header = f'{"":18}' + ''.join(f'{short[n]:>10}' for n in bot_names)
    print(header)
    for a in bot_names:
        row = f'{a:<18}'
        for b in bot_names:
            if a == b:
                row += f'{"---":>10}'
            else:
                key = tuple(sorted([a, b]))
                h = head_to_head.get(key, {})
                da = h.get(a, 0)
                row += f'{da:>+10}'
        print(row)

    # Save JSON for further analysis
    output = {
        'leaderboard': [
            {
                'rank': rank,
                'name': name,
                **r,
                'avg_delta_per_hand': r['delta'] / (r['played'] * ROUNDS_PER_MATCH) if r['played'] > 0 else 0
            }
            for rank, (name, r) in enumerate(ranked, 1)
        ],
        'head_to_head': {f'{a}_vs_{b}': v for (a,b), v in head_to_head.items()},
        'rounds_per_match': ROUNDS_PER_MATCH,
    }
    out_path = REPO_ROOT / 'results' / 'tournament_results.json'
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f'\nFull results saved to results/tournament_results.json')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results, h2h, bot_names = run_tournament()
    print_report(results, h2h, bot_names)
