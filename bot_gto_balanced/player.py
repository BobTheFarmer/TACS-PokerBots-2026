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
        new_board_card = avail[idx[0]]
        opp = [avail[idx[1]], avail[idx[2]]]
        full_board = trimmed_board + [new_board_card] + [avail[idx[i]] for i in range(3, n_need)]
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
        self._last_round_num = 0

    def on_opponent_reveal(self, old_card, target_type, target_index):
        if old_card and old_card != '??':
            self.opp_dead_cards.add(old_card)

    def handle_new_round(self, game_state, round_state, active):
        self.opp_dead_cards = set()
        self._last_round_num = game_state.round_num

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def _bucket(self, equity):
        if equity >= 0.78:
            return 'NUTS'
        if equity >= 0.62:
            return 'VALUE'
        if equity >= 0.50:
            return 'MEDIUM'
        if equity >= 0.40:
            return 'MARGINAL'
        if equity >= 0.30:
            return 'BLUFF_C'
        return 'TRASH'

    def _raise_bounds(self, round_state, legal):
        if RaiseAction in legal:
            return round_state.raise_bounds()
        return (0, 0)

    def _bet(self, round_state, active, pot, frac, min_r, max_r):
        raw = round_state.pips[active] + int(pot * frac)
        return max(min_r, min(raw, max_r))

    def _random_frac(self, lo, hi):
        return random.uniform(lo, hi)

    def _legalize_raise(self, amount, min_r, max_r):
        return max(min_r, min(int(amount), max_r))

    def _basic_preflop(self, round_state, legal, equity, active, pot, continue_cost, min_r, max_r):
        button = round_state.button
        if active == 0:
            if button == 0:
                if equity >= 0.40 and RaiseAction in legal:
                    return RaiseAction(self._legalize_raise(self._bet(round_state, active, pot, 2.5 * BIG_BLIND / max(pot, 1), min_r, max_r), min_r, max_r))
                if equity >= 0.30 and CallAction in legal:
                    return CallAction()
                return FoldAction() if FoldAction in legal else CheckAction()
            if equity >= 0.62 and RaiseAction in legal:
                return RaiseAction(self._legalize_raise(int(3.5 * continue_cost + round_state.pips[active]), min_r, max_r))
            if equity >= 0.50 and CallAction in legal:
                return CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if continue_cost == 0:
            if equity >= 0.40 and RaiseAction in legal:
                return RaiseAction(self._legalize_raise(round_state.pips[active] + int(4 * BIG_BLIND), min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()

        if equity >= 0.52 and RaiseAction in legal:
            return RaiseAction(self._legalize_raise(round_state.pips[active] + int(3.5 * continue_cost), min_r, max_r))
        if equity >= 0.38 and CallAction in legal:
            return CallAction()
        return FoldAction() if FoldAction in legal else CallAction()

    def _preflop(self, round_state, legal, equity, active, pot, continue_cost, min_r, max_r):
        button = round_state.button
        bucket = self._bucket(equity)

        if active == 0 and button == 0:
            if bucket in ('NUTS', 'VALUE', 'MEDIUM') and RaiseAction in legal:
                return RaiseAction(self._legalize_raise(2.5 * BIG_BLIND, min_r, max_r))
            if bucket == 'MARGINAL':
                if RaiseAction in legal and random.random() < 0.60:
                    return RaiseAction(self._legalize_raise(2.5 * BIG_BLIND, min_r, max_r))
                return FoldAction() if FoldAction in legal else CheckAction()
            if bucket == 'BLUFF_C':
                if RaiseAction in legal and random.random() < 0.25:
                    return RaiseAction(self._legalize_raise(2.5 * BIG_BLIND, min_r, max_r))
                return FoldAction() if FoldAction in legal else CheckAction()
            return FoldAction() if FoldAction in legal else CheckAction()

        if active == 1 and continue_cost == 0:
            if RaiseAction in legal:
                return RaiseAction(self._legalize_raise(4 * BIG_BLIND, min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()

        if active == 1:
            if bucket == 'NUTS':
                if RaiseAction in legal:
                    return RaiseAction(self._legalize_raise(int(3.5 * round_state.pips[1 - active]), min_r, max_r))
                return CallAction() if CallAction in legal else FoldAction()
            if bucket == 'VALUE':
                if RaiseAction in legal and random.random() < 0.40:
                    return RaiseAction(self._legalize_raise(int(3.5 * round_state.pips[1 - active]), min_r, max_r))
                return CallAction() if CallAction in legal else FoldAction()
            if bucket == 'MEDIUM':
                return CallAction() if CallAction in legal else FoldAction()
            if bucket == 'MARGINAL':
                if CallAction in legal and random.random() < 0.50:
                    return CallAction()
                return FoldAction() if FoldAction in legal else CallAction()
            if bucket == 'BLUFF_C':
                if CallAction in legal and random.random() < 0.20:
                    return CallAction()
                return FoldAction() if FoldAction in legal else CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if active == 0:
            if bucket == 'NUTS':
                if RaiseAction in legal and random.random() < 0.55:
                    return RaiseAction(self._legalize_raise(int(2.5 * round_state.pips[1 - active]), min_r, max_r))
                return CallAction() if CallAction in legal else FoldAction()
            if bucket == 'VALUE':
                if RaiseAction in legal and random.random() < 0.25:
                    return RaiseAction(self._legalize_raise(int(2.2 * round_state.pips[1 - active]), min_r, max_r))
                return CallAction() if CallAction in legal else FoldAction()
            if bucket == 'MEDIUM':
                return CallAction() if CallAction in legal else FoldAction()
            if bucket == 'MARGINAL':
                if CallAction in legal and random.random() < 0.50:
                    return CallAction()
                return FoldAction() if FoldAction in legal else CallAction()
            if bucket == 'BLUFF_C':
                return FoldAction() if FoldAction in legal else CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        return CheckAction() if CheckAction in legal else CallAction()

    def _postflop_facing(self, legal, equity, pot_odds, continue_cost, round_state, active, min_r, max_r, pot):
        bucket = self._bucket(equity)
        if bucket == 'NUTS':
            if RaiseAction in legal and random.random() < 0.55:
                return RaiseAction(self._legalize_raise(round_state.pips[active] + int(1.5 * continue_cost + continue_cost), min_r, max_r))
            return CallAction() if CallAction in legal else FoldAction()
        if bucket == 'VALUE':
            if RaiseAction in legal and random.random() < 0.25:
                return RaiseAction(self._legalize_raise(round_state.pips[active] + int(1.2 * continue_cost + continue_cost), min_r, max_r))
            return CallAction() if CallAction in legal else FoldAction()
        if bucket == 'MEDIUM':
            return CallAction() if equity >= pot_odds and CallAction in legal else (FoldAction() if FoldAction in legal else CallAction())
        if bucket == 'MARGINAL':
            return CallAction() if equity >= pot_odds + 0.05 and CallAction in legal else (FoldAction() if FoldAction in legal else CallAction())
        return FoldAction() if FoldAction in legal else CallAction()

    def _postflop_check(self, legal, equity, round_state, active, min_r, max_r, pot):
        bucket = self._bucket(equity)
        if bucket == 'NUTS':
            if RaiseAction in legal:
                return RaiseAction(self._legalize_raise(self._bet(round_state, active, pot, 0.70, min_r, max_r), min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()
        if bucket == 'VALUE':
            if RaiseAction in legal and random.random() < 0.80:
                return RaiseAction(self._legalize_raise(self._bet(round_state, active, pot, self._random_frac(0.65, 0.80), min_r, max_r), min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()
        if bucket == 'MEDIUM':
            if RaiseAction in legal and random.random() < 0.40:
                return RaiseAction(self._legalize_raise(self._bet(round_state, active, pot, 0.45, min_r, max_r), min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()
        if bucket == 'MARGINAL':
            if RaiseAction in legal and random.random() < 0.15:
                return RaiseAction(self._legalize_raise(self._bet(round_state, active, pot, self._random_frac(0.38, 0.52), min_r, max_r), min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()
        if bucket == 'BLUFF_C':
            if RaiseAction in legal and random.random() < 0.22:
                return RaiseAction(self._legalize_raise(self._bet(round_state, active, pot, self._random_frac(0.38, 0.52), min_r, max_r), min_r, max_r))
            return CheckAction() if CheckAction in legal else CallAction()
        return CheckAction() if CheckAction in legal else CallAction()

    def _basic_action(self, game_state, round_state, active, equity, pot_odds, continue_cost, pot, legal, min_r, max_r):
        street = round_state.street
        if street == 0:
            return self._preflop(round_state, legal, equity, active, pot, continue_cost, min_r, max_r)
        if continue_cost > 0:
            return self._postflop_facing(legal, equity, pot_odds, continue_cost, round_state, active, min_r, max_r, pot)
        return self._postflop_check(legal, equity, round_state, active, min_r, max_r, pot)

    def _redraw_decision(self, round_state, active, legal, equity):
        if RedrawAction not in legal or round_state.redraws_used[active] or round_state.street >= 5:
            return None
        best_gain = 0.0
        best_target = None
        dead = list(self.opp_dead_cards) if self.opp_dead_cards else None
        current_eq = equity
        for i in range(2):
            if not round_state.hands[active][i] or round_state.hands[active][i] == '??':
                continue
            new_eq = mc_redraw_equity(round_state.hands[active], round_state.board, 'hole', i, excluded_extra=dead, n_sims=200)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain = gain
                best_target = ('hole', i)
        for i in range(len(round_state.board)):
            new_eq = mc_redraw_equity(round_state.hands[active], round_state.board, 'board', i, excluded_extra=dead, n_sims=200)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain = gain
                best_target = ('board', i)
        if best_target is None:
            return None
        if best_gain > 0.05:
            trigger = 1.0
        elif best_gain > 0.02:
            trigger = 0.50
        elif best_gain > 0.00:
            trigger = 0.15
        else:
            return None
        if random.random() >= trigger:
            return None
        target_type, target_index = best_target
        return (target_type, target_index)

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip
        pot = 2 * STARTING_STACK - my_stack - opp_stack
        min_r, max_r = self._raise_bounds(round_state, legal)
        equity = mc_equity(my_cards, board, excluded_extra=self.opp_dead_cards, n_sims=250)
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0
        basic_action = self._basic_action(game_state, round_state, active, equity, pot_odds, continue_cost, pot, legal, min_r, max_r)
        redraw_choice = self._redraw_decision(round_state, active, legal, equity)
        if redraw_choice is not None:
            target_type, target_index = redraw_choice
            return RedrawAction(target_type, target_index, basic_action)
        return basic_action


if __name__ == '__main__':
    run_bot_enhanced(Player(), parse_args())
