"""
v1: Monte Carlo equity bot with smart redraw decisions.

Strategy:
- Preflop: equity-based open/3bet/fold ranges
- Postflop: MC equity vs pot odds for call/bet/fold
- Redraw: compare current equity vs expected equity after swap; take if gain > threshold
- Opponent revealed cards (via redraw) excluded from MC sampling
"""

import random
import socket

import numpy as np
import pkrbot

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, Runner


# ── Card constants ────────────────────────────────────────────────────────────

RANKS = '23456789TJQKA'
SUITS = 'hdsc'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


# ── Monte Carlo equity ────────────────────────────────────────────────────────

def mc_equity(my_cards, board, excluded_extra=None, n_sims=300):
    """
    Estimate win probability via Monte Carlo vs a random opponent range.
    Completes the board to 5 cards and evaluates with pkrbot.
    '??' hole cards (post-redraw unknowns) are sampled randomly each sim.
    Returns a value in [0, 1] (ties count as 0.5).
    """
    known_mine = [c for c in my_cards if c and c != '??']
    n_unknown_mine = len(my_cards) - len(known_mine)

    excluded = set(known_mine) | set(board)
    if excluded_extra:
        excluded |= {c for c in excluded_extra if c and c != '??'}

    avail = [c for c in ALL_CARDS if c not in excluded]
    # need: unknown hole cards + opp 2 hole cards + board completion
    n_need = n_unknown_mine + 2 + (5 - len(board))

    if n_need > len(avail):
        return 0.5

    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        i = 0
        # Fill in any unknown hole cards
        my_full = list(known_mine)
        for _ in range(n_unknown_mine):
            my_full.append(avail[idx[i]]); i += 1
        opp = [avail[idx[i]], avail[idx[i + 1]]]; i += 2
        full_board = board + [avail[idx[i + j]] for j in range(5 - len(board))]

        my_score  = pkrbot.evaluate(my_full + full_board)
        opp_score = pkrbot.evaluate(opp + full_board)

        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    return wins / n_sims


def mc_redraw_equity(my_cards, board, target_type, target_index,
                     excluded_extra=None, n_sims=150):
    """
    Expected equity if we swap target card with a random replacement.
    Old card is excluded from sampling (it's revealed / dead).
    """
    if target_type == 'hole':
        old_card  = my_cards[target_index]
        my_other  = my_cards[1 - target_index]
        excluded  = {my_other, old_card} | set(board)
        if excluded_extra:
            excluded |= {c for c in excluded_extra if c and c != '??'}

        avail  = [c for c in ALL_CARDS if c not in excluded]
        n_need = 1 + 2 + (5 - len(board))   # new_hole + opp_hole + board

        if n_need > len(avail):
            return 0.5

        wins = 0.0
        for _ in range(n_sims):
            idx = np.random.choice(len(avail), n_need, replace=False)
            new_hole   = avail[idx[0]]
            opp        = [avail[idx[1]], avail[idx[2]]]
            full_board = board + [avail[idx[i]] for i in range(3, n_need)]

            my_score  = pkrbot.evaluate([new_hole, my_other] + full_board)
            opp_score = pkrbot.evaluate(opp + full_board)

            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5

        return wins / n_sims

    else:  # board card swap
        old_card       = board[target_index]
        trimmed_board  = [c for i, c in enumerate(board) if i != target_index]
        excluded       = set(my_cards) | set(trimmed_board) | {old_card}
        if excluded_extra:
            excluded |= {c for c in excluded_extra if c and c != '??'}

        avail  = [c for c in ALL_CARDS if c not in excluded]
        # after swap the board still has len(board) cards; still need 5 - len(board) more
        n_need = 1 + 2 + (5 - len(board))

        if n_need > len(avail):
            return 0.5

        wins = 0.0
        for _ in range(n_sims):
            idx           = np.random.choice(len(avail), n_need, replace=False)
            new_bc        = avail[idx[0]]
            opp           = [avail[idx[1]], avail[idx[2]]]
            full_board    = trimmed_board + [new_bc] + [avail[idx[i]] for i in range(3, n_need)]

            my_score  = pkrbot.evaluate(my_cards + full_board)
            opp_score = pkrbot.evaluate(opp + full_board)

            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5

        return wins / n_sims


# ── Enhanced runner (captures opponent's revealed redraw card) ────────────────

class EnhancedRunner(Runner):
    """
    Subclass of Runner that intercepts opponent redraw reveals (X packets)
    and passes them to the bot via on_opponent_reveal().
    """

    def _apply_action_clause(self, round_state, action_clause):
        actor      = round_state.button % 2
        old_card   = self._pending_redraw_old_card.get(actor)
        redraw_info = self._pending_redraw.get(actor)

        result = super()._apply_action_clause(round_state, action_clause)

        # Notify bot after super() has applied (and cleared) the redraw state
        if redraw_info is not None and old_card is not None:
            target_type, target_index = redraw_info
            if hasattr(self.pokerbot, 'on_opponent_reveal'):
                self.pokerbot.on_opponent_reveal(old_card, target_type, target_index)

        return result


def run_bot_enhanced(pokerbot, args):
    assert isinstance(pokerbot, Bot)
    try:
        sock = socket.create_connection((args.host, args.port))
    except OSError:
        print(f'Could not connect to {args.host}:{args.port}')
        return
    socketfile = sock.makefile('rw')
    EnhancedRunner(pokerbot, socketfile).run()
    socketfile.close()
    sock.close()


# ── Player ────────────────────────────────────────────────────────────────────

class Player(Bot):
    """
    MC equity-based bot.  Key edges:
      1. Accurate postflop equity via pkrbot MC simulation
      2. Smart redraw: quantifies EV gain before committing the one-time swap
      3. Opponent's revealed redraw card excluded from MC deck → better equity estimates
    """

    def __init__(self):
        self.opp_dead_cards = set()   # cards revealed by opponent's redraws this hand
        self.hands_seen     = 0

    # ── EnhancedRunner hook ───────────────────────────────────────────────────

    def on_opponent_reveal(self, old_card, target_type, target_index):
        if old_card and old_card != '??':
            self.opp_dead_cards.add(old_card)

    # ── Bot interface ─────────────────────────────────────────────────────────

    def handle_new_round(self, game_state, round_state, active):
        self.opp_dead_cards = set()
        self.hands_seen    += 1

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):
        legal  = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board    = round_state.board

        my_pip  = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack  = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        continue_cost = opp_pip - my_pip
        pot = 2 * STARTING_STACK - my_stack - opp_stack

        # ── Redraw decision (before computing equity for the action) ──────────
        if RedrawAction in legal and not round_state.redraws_used[active] and street > 0:
            rd = self._decide_redraw(my_cards, board, legal, continue_cost)
            if rd is not None:
                return rd

        # ── Equity ────────────────────────────────────────────────────────────
        dead = list(self.opp_dead_cards) if self.opp_dead_cards else None
        n_sims = 100 if street == 0 else 300
        equity = mc_equity(my_cards, board, excluded_extra=dead, n_sims=n_sims)

        # ── Pot odds ──────────────────────────────────────────────────────────
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0

        # ── Raise bounds ──────────────────────────────────────────────────────
        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()
        else:
            min_r = max_r = 0

        def bet(frac):
            """Return raise-to amount for frac * pot."""
            raw = round_state.pips[active] + int(pot * frac)
            return max(min_r, min(raw, max_r))

        # ── Decision ──────────────────────────────────────────────────────────
        if street == 0:
            return self._preflop(legal, equity, continue_cost, pot,
                                 active, round_state.button, min_r, max_r, bet)
        else:
            return self._postflop(legal, equity, pot_odds, continue_cost,
                                  min_r, max_r, bet)

    # ── Redraw ────────────────────────────────────────────────────────────────

    def _decide_redraw(self, my_cards, board, legal, continue_cost):
        """
        Run MC to estimate equity gain from each possible swap.
        Only redraw if gain > THRESHOLD.
        Returns a RedrawAction or None.
        """
        THRESHOLD = 0.03   # minimum equity gain to justify reveal cost

        dead = list(self.opp_dead_cards) if self.opp_dead_cards else None
        current_eq = mc_equity(my_cards, board, excluded_extra=dead, n_sims=150)

        best_gain   = THRESHOLD
        best_target = None   # (target_type, target_index)

        # Evaluate swapping each hole card
        for i in range(2):
            if not my_cards[i] or my_cards[i] == '??':
                continue
            new_eq = mc_redraw_equity(my_cards, board, 'hole', i,
                                      excluded_extra=dead, n_sims=100)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain   = gain
                best_target = ('hole', i)

        # Evaluate swapping board cards
        for i in range(len(board)):
            new_eq = mc_redraw_equity(my_cards, board, 'board', i,
                                      excluded_extra=dead, n_sims=100)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain   = gain
                best_target = ('board', i)

        if best_target is None:
            return None

        tt, ti = best_target
        # Embed the cheapest legal betting action inside the redraw
        if CheckAction in legal:
            inner = CheckAction()
        elif continue_cost == 0:
            inner = CheckAction()
        else:
            inner = CallAction()
        return RedrawAction(tt, ti, inner)

    # ── Preflop ───────────────────────────────────────────────────────────────

    def _preflop(self, legal, equity, continue_cost, pot,
                 active, button, min_r, max_r, bet):
        """
        Heads-up preflop: SB is active=0 at button=0, acts first.
        """
        is_sb = (active == 0)

        if is_sb:
            if button == 0:
                # SB opening: continue_cost = 1 (just the BB)
                if equity >= 0.40 and RaiseAction in legal:
                    amount = max(min_r, min(int(2.5 * BIG_BLIND), max_r))
                    return RaiseAction(amount)
                elif equity >= 0.36 and CallAction in legal:
                    return CallAction()   # limp with playable hands
                else:
                    return FoldAction() if FoldAction in legal else CallAction()
            else:
                # SB facing BB 3-bet
                if equity >= 0.58 and RaiseAction in legal:
                    return RaiseAction(max_r)   # 4-bet shove
                elif equity >= 0.46 and CallAction in legal:
                    return CallAction()
                else:
                    return FoldAction() if FoldAction in legal else CallAction()

        else:  # BB
            if continue_cost == 0:
                # SB limped — BB can raise or check
                if equity >= 0.56 and RaiseAction in legal:
                    amount = max(min_r, min(int(3 * BIG_BLIND), max_r))
                    return RaiseAction(amount)
                return CheckAction()
            else:
                # Facing SB open raise
                if equity >= 0.62 and RaiseAction in legal:
                    amount = max(min_r, min(int(3.5 * continue_cost) + pot, max_r))
                    return RaiseAction(amount)
                elif equity >= 0.43:
                    return CallAction() if CallAction in legal else CheckAction()
                else:
                    return FoldAction() if FoldAction in legal else CheckAction()

    # ── Postflop ──────────────────────────────────────────────────────────────

    def _postflop(self, legal, equity, pot_odds, continue_cost,
                  min_r, max_r, bet):
        """
        Postflop: equity-driven bet/call/fold with light bluffing.
        """
        facing_bet = continue_cost > 0

        if facing_bet:
            # ── Facing a bet ─────────────────────────────────────────────────
            if equity >= pot_odds + 0.15 and RaiseAction in legal:
                # Strong hand vs bet → raise for value
                return RaiseAction(bet(0.7))
            elif equity >= pot_odds + 0.02:
                # Calling range (equity clears pot odds with margin)
                return CallAction() if CallAction in legal else FoldAction()
            else:
                return FoldAction() if FoldAction in legal else CallAction()

        else:
            # ── No bet to face (checking) ─────────────────────────────────────
            if equity >= 0.65 and RaiseAction in legal:
                return RaiseAction(bet(0.70))    # strong value bet
            elif equity >= 0.54 and RaiseAction in legal:
                return RaiseAction(bet(0.50))    # thin value / protection
            elif equity < 0.40 and RaiseAction in legal and random.random() < 0.18:
                return RaiseAction(bet(0.45))    # bluff
            else:
                return CheckAction() if CheckAction in legal else FoldAction()


if __name__ == '__main__':
    run_bot_enhanced(Player(), parse_args())
