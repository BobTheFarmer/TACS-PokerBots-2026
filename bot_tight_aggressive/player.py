import random
import socket

import numpy as np
import pkrbot

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, Runner


RANKS = '23456789TJQKA'
SUITS = 'hdsc'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


def mc_equity(my_cards, board, excluded_extra=None, n_sims=300):
    known_mine = [c for c in my_cards if c and c != '??']
    n_unknown_mine = len(my_cards) - len(known_mine)
    excluded = set(known_mine) | set(board)
    if excluded_extra:
        excluded |= {c for c in excluded_extra if c and c != '??'}
    avail = [c for c in ALL_CARDS if c not in excluded]
    n_need = n_unknown_mine + 2 + (5 - len(board))
    if n_need > len(avail):
        return 0.5
    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        i = 0
        my_full = list(known_mine)
        for _u in range(n_unknown_mine):
            my_full.append(avail[idx[i]]); i += 1
        opp = [avail[idx[i]], avail[idx[i + 1]]]; i += 2
        full_board = board + [avail[idx[i + j]] for j in range(5 - len(board))]
        my_score = pkrbot.evaluate(my_full + full_board)
        opp_score = pkrbot.evaluate(opp + full_board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
    return wins / n_sims


def mc_redraw_equity(my_cards, board, target_type, target_index, excluded_extra=None, n_sims=150):
    if target_type == 'hole':
        old_card = my_cards[target_index]
        my_other = my_cards[1 - target_index]
        excluded = {my_other, old_card} | set(board)
        if excluded_extra:
            excluded |= {c for c in excluded_extra if c and c != '??'}
        avail = [c for c in ALL_CARDS if c not in excluded]
        n_need = 1 + 2 + (5 - len(board))
        if n_need > len(avail):
            return 0.5
        wins = 0.0
        for _ in range(n_sims):
            idx = np.random.choice(len(avail), n_need, replace=False)
            new_hole = avail[idx[0]]
            opp = [avail[idx[1]], avail[idx[2]]]
            full_board = board + [avail[idx[i]] for i in range(3, n_need)]
            my_score = pkrbot.evaluate([new_hole, my_other] + full_board)
            opp_score = pkrbot.evaluate(opp + full_board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
        return wins / n_sims

    old_card = board[target_index]
    trimmed_board = [c for i, c in enumerate(board) if i != target_index]
    excluded = set(my_cards) | set(trimmed_board) | {old_card}
    if excluded_extra:
        excluded |= {c for c in excluded_extra if c and c != '??'}
    avail = [c for c in ALL_CARDS if c not in excluded]
    n_need = 1 + 2 + (5 - len(board))
    if n_need > len(avail):
        return 0.5
    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        new_bc = avail[idx[0]]
        opp = [avail[idx[1]], avail[idx[2]]]
        full_board = trimmed_board + [new_bc] + [avail[idx[i]] for i in range(3, n_need)]
        my_score = pkrbot.evaluate(my_cards + full_board)
        opp_score = pkrbot.evaluate(opp + full_board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
    return wins / n_sims


class EnhancedRunner(Runner):
    def _apply_action_clause(self, round_state, action_clause):
        actor = round_state.button % 2
        old_card = self._pending_redraw_old_card.get(actor)
        redraw_info = self._pending_redraw.get(actor)
        result = super()._apply_action_clause(round_state, action_clause)
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


class Player(Bot):
    def __init__(self):
        self.opp_dead_cards = set()
        self.hands_seen = 0
        self.last_round_delta = 0

    def on_opponent_reveal(self, old_card, target_type, target_index):
        if old_card and old_card != '??':
            self.opp_dead_cards.add(old_card)

    def handle_new_round(self, game_state, round_state, active):
        self.opp_dead_cards = set()
        self.hands_seen += 1

    def handle_round_over(self, game_state, terminal_state, active):
        self.last_round_delta = terminal_state.deltas[active]
        outcome = 'win' if self.last_round_delta > 0 else ('loss' if self.last_round_delta < 0 else 'tie')
        print(
            f'[tight_aggressive] hand={game_state.round_num} outcome={outcome} delta={self.last_round_delta} bankroll={game_state.bankroll}',
            file=__import__('sys').stderr,
        )

    def _safe_raise(self, legal, amount, min_r, max_r):
        if RaiseAction not in legal:
            return None
        return RaiseAction(max(min_r, min(amount, max_r)))

    def _build_bet(self, round_state, active, pot, frac, min_r, max_r):
        raw = round_state.pips[active] + int(pot * frac)
        return max(min_r, min(raw, max_r))

    def _postflop_action(self, legal, equity, pot_odds, continue_cost, pot, round_state, active, min_r, max_r):
        if continue_cost > 0:
            if equity >= pot_odds:
                return CallAction() if CallAction in legal else (FoldAction() if FoldAction in legal else CheckAction())
            return FoldAction() if FoldAction in legal else CallAction()

        if equity >= 0.70 and RaiseAction in legal:
            return RaiseAction(self._build_bet(round_state, active, pot, 0.75, min_r, max_r))
        if equity >= 0.62 and RaiseAction in legal:
            return RaiseAction(self._build_bet(round_state, active, pot, 0.55, min_r, max_r))
        return CheckAction() if CheckAction in legal else (CallAction() if CallAction in legal else FoldAction())

    def _preflop_action(self, legal, equity, continue_cost, pot, round_state, active, button, min_r, max_r):
        is_sb = active == 0
        if is_sb and button == 0:
            if equity >= 0.50 and RaiseAction in legal:
                return RaiseAction(max(min_r, min(3 * BIG_BLIND, max_r)))
            return FoldAction() if FoldAction in legal else CheckAction()

        if is_sb and button > 0:
            if equity >= 0.65 and RaiseAction in legal:
                target = round_state.pips[active] + int(continue_cost * 3.5)
                return RaiseAction(max(min_r, min(target, max_r)))
            if equity >= 0.55 and CallAction in legal:
                return CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if continue_cost == 0:
            if equity >= 0.58 and RaiseAction in legal:
                return RaiseAction(max(min_r, min(4 * BIG_BLIND, max_r)))
            return CheckAction() if CheckAction in legal else CallAction()

        if equity >= 0.65 and RaiseAction in legal:
            target = round_state.pips[active] + int(continue_cost * 3.5)
            return RaiseAction(max(min_r, min(target, max_r)))
        if equity >= 0.48 and CallAction in legal:
            return CallAction()
        return FoldAction() if FoldAction in legal else CallAction()

    def _decide_redraw(self, my_cards, board, legal, current_eq, dead, street, round_state, active, pot, continue_cost, min_r, max_r):
        best_gain = 0.06
        best_target = None
        for i in range(2):
            if not my_cards[i] or my_cards[i] == '??':
                continue
            new_eq = mc_redraw_equity(my_cards, board, 'hole', i, excluded_extra=dead, n_sims=200)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain = gain
                best_target = ('hole', i)
        if best_target is None:
            return None
        if street == 0:
            inner = self._preflop_action(legal, current_eq, continue_cost, pot, round_state, active, round_state.button, min_r, max_r)
        else:
            inner = self._postflop_action(legal, current_eq, continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0, continue_cost, pot, round_state, active, min_r, max_r)
        if isinstance(inner, RedrawAction):
            inner = CheckAction()
        return RedrawAction(best_target[0], best_target[1], inner)

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = max(0, opp_pip - my_pip)
        pot = 2 * STARTING_STACK - my_stack - opp_stack

        if RedrawAction in legal and street < 5 and not round_state.redraws_used[active]:
            dead = list(self.opp_dead_cards) if self.opp_dead_cards else None
            current_eq = mc_equity(my_cards, board, excluded_extra=dead, n_sims=400 if street > 0 else 250)
            redraw = self._decide_redraw(my_cards, board, legal, current_eq, dead, street, round_state, active, pot, continue_cost, *(round_state.raise_bounds() if RaiseAction in legal else (0, 0)))
            if redraw is not None:
                return redraw
        else:
            current_eq = mc_equity(my_cards, board, excluded_extra=list(self.opp_dead_cards) if self.opp_dead_cards else None, n_sims=400 if street > 0 else 250)

        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0
        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()
        else:
            min_r = max_r = 0

        if street == 0:
            return self._preflop_action(legal, current_eq, continue_cost, pot, round_state, active, round_state.button, min_r, max_r)
        return self._postflop_action(legal, current_eq, pot_odds, continue_cost, pot, round_state, active, min_r, max_r)


if __name__ == '__main__':
    run_bot_enhanced(Player(), parse_args())
