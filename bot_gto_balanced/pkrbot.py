"""
pkrbot_stub.py — pure-Python drop-in for pkrbot.
Provides: Deck, evaluate(cards).
Used for local testing when the C extension cannot be compiled.
"""
import random
from itertools import combinations
from collections import Counter

RANKS = '23456789TJQKA'
SUITS = 'hdsc'
_RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}   # '2'→2 … 'A'→14


class Deck:
    def __init__(self):
        self._cards = [r + s for r in RANKS for s in SUITS]
        self._pos = 0

    def shuffle(self):
        random.shuffle(self._cards)
        self._pos = 0

    def deal(self, n):
        out = self._cards[self._pos:self._pos + n]
        self._pos += n
        return out


def evaluate(cards):
    """
    Evaluate the best 5-card hand from up to 7 cards.
    Returns an integer where higher = better hand.
    """
    best = -1
    for combo in combinations(cards, 5):
        s = _score5(combo)
        if s > best:
            best = s
    return best


def _score5(cards):
    ranks = sorted([_RANK_VAL[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    flush = len(set(suits)) == 1

    cnt = Counter(ranks)
    counts = sorted(cnt.values(), reverse=True)
    by_freq = sorted(cnt.keys(), key=lambda r: (cnt[r], r), reverse=True)

    # Straight detection
    straight, s_high = False, 0
    uniq = sorted(set(ranks), reverse=True)
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight, s_high = True, uniq[0]
        elif set(uniq) == {14, 2, 3, 4, 5}:      # wheel A-2-3-4-5
            straight, s_high = True, 5

    # Hand categories (8=best … 0=worst)
    if flush and straight:
        return _enc(8, s_high)
    if counts[0] == 4:
        return _enc(7, by_freq[0], by_freq[1])
    if counts[:2] == [3, 2]:
        return _enc(6, by_freq[0], by_freq[1])
    if flush:
        return _enc(5, *ranks)
    if straight:
        return _enc(4, s_high)
    if counts[0] == 3:
        kickers = sorted([r for r in ranks if cnt[r] == 1], reverse=True)
        return _enc(3, by_freq[0], *kickers)
    if counts[:2] == [2, 2]:
        pairs = sorted([r for r in cnt if cnt[r] == 2], reverse=True)
        kicker = max(r for r in cnt if cnt[r] == 1)
        return _enc(2, pairs[0], pairs[1], kicker)
    if counts[0] == 2:
        kickers = sorted([r for r in ranks if cnt[r] == 1], reverse=True)
        return _enc(1, by_freq[0], *kickers)
    return _enc(0, *ranks)


def _enc(*vals):
    """Pack up to 6 values (0-14 each) into a single integer (base-16)."""
    padded = list(vals) + [0] * 6
    result = 0
    for v in padded[:6]:
        result = result * 16 + int(v)
    return result
