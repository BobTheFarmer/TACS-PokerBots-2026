"""
Redraw specialist bot.

Strategy:
- Standard preflop ranges with equity-based opens and 3-bets
- Postflop equity-driven value betting with conservative bluffs
- Deep redraw analysis that prefers turn redraws, tracks opponent reveals,
  and prices board swaps differently when they appear to complete a draw
"""

import random
import socket
import sys

import numpy as np
import pkrbot

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, Runner


RANKS = '23456789TJQKA'
SUITS = 'hdsc'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]


def _card_is_unknown(card):
    return card == '??'


def _known_cards(*card_groups):
    cards = []
    for group in card_groups:
        for card in group:
            if card and card != '??':
                cards.append(card)
    return cards


def _raise_to(round_state, active, pot, min_r, max_r, frac):
    raw = round_state.pips[active] + int(pot * frac)
    return max(min_r, min(raw, max_r))


def mc_equity(my_cards, board, excluded_extra=None, n_sims=300):
    """
    Estimate win probability via Monte Carlo vs a random opponent range.
    Handles unknown '??' placeholders in our hand or on the board by sampling
    replacements from the unseen deck.
    """
    excluded = set(_known_cards(my_cards, board))
    if excluded_extra:
        excluded |= {c for c in excluded_extra if c and c != '??'}

    avail = [c for c in ALL_CARDS if c not in excluded]
    my_unknown = sum(1 for c in my_cards if c == '??')
    board_unknown = sum(1 for c in board if c == '??')
    n_need = my_unknown + board_unknown + 2 + (5 - len(board))

    if n_need > len(avail):
        return 0.5

    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        cursor = 0

        filled_my_cards = list(my_cards)
        for i, card in enumerate(filled_my_cards):
            if card == '??':
                filled_my_cards[i] = avail[idx[cursor]]
                cursor += 1

        filled_board = list(board)
        for i, card in enumerate(filled_board):
            if card == '??':
                filled_board[i] = avail[idx[cursor]]
                cursor += 1

        opp = [avail[idx[cursor]], avail[idx[cursor + 1]]]
        cursor += 2

        for _ in range(5 - len(board)):
            filled_board.append(avail[idx[cursor]])
            cursor += 1

        my_score = pkrbot.evaluate(filled_my_cards + filled_board)
        opp_score = pkrbot.evaluate(opp + filled_board)

        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    return wins / n_sims


def mc_redraw_equity(my_cards, board, target_type, target_index, excluded_extra=None, n_sims=200):
    """
    Expected equity after swapping one card. The target slot is replaced by
    '??' and then resolved through the general equity sampler.
    """
    if target_type == 'hole':
        candidate_my_cards = list(my_cards)
        old_card = candidate_my_cards[target_index]
        if old_card != '??':
            candidate_my_cards[target_index] = '??'
        extra = [old_card]
        if excluded_extra:
            extra.extend(excluded_extra)
        return mc_equity(candidate_my_cards, board, excluded_extra=extra, n_sims=n_sims)

    candidate_board = list(board)
    old_card = candidate_board[target_index]
    if old_card != '??':
        candidate_board[target_index] = '??'
    extra = [old_card]
    if excluded_extra:
        extra.extend(excluded_extra)
    return mc_equity(my_cards, candidate_board, excluded_extra=extra, n_sims=n_sims)


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
        self.opp_redrawed_this_hand = False
        self.opp_redraw_target_type = None
        self.opp_redraw_target_index = None
        self.opp_redraw_count = 0
        self.opp_redraw_streets = []
        self.current_street = 0
        self.opp_reveal_street = None
        self.opp_revealed_weak_card = False
        self.hands = 0

    def on_opponent_reveal(self, old_card, target_type, target_index):
        if old_card and old_card != '??':
            self.opp_dead_cards.add(old_card)
            self.opp_revealed_weak_card = old_card[0] in '2345678'
        self.opp_redrawed_this_hand = True
        self.opp_redraw_target_type = target_type
        self.opp_redraw_target_index = target_index
        self.opp_redraw_count += 1
        self.opp_redraw_streets.append(self.current_street)
        self.opp_reveal_street = self.current_street

    def handle_new_round(self, game_state, round_state, active):
        self.opp_dead_cards = set()
        self.opp_redrawed_this_hand = False
        self.opp_redraw_target_type = None
        self.opp_redraw_target_index = None
        self.opp_reveal_street = None
        self.opp_revealed_weak_card = False
        self.current_street = 0
        self.hands += 1

    def handle_round_over(self, game_state, terminal_state, active):
        delta = terminal_state.deltas[active]
        print(
            f'round={game_state.round_num} delta={delta} '
            f'opp_redraw={self.opp_redrawed_this_hand} dead={sorted(self.opp_dead_cards)}',
            file=sys.stderr,
            flush=True,
        )

    def _postflop_sims(self, street):
        redraw_rate = self.opp_redraw_count / max(self.hands, 1)
        if redraw_rate > 0.5:
            return 500
        if self.opp_redrawed_this_hand and self.opp_reveal_street is not None and street > self.opp_reveal_street:
            return 500
        return 350

    def _redraw_sims(self):
        redraw_rate = self.opp_redraw_count / max(self.hands, 1)
        if redraw_rate > 0.5 or self.opp_redrawed_this_hand:
            return 300, 250
        return 250, 200

    def _is_targeted_board_swap(self, board, target_index):
        if not self.opp_redrawed_this_hand:
            return False
        target = board[target_index]
        if not target or target == '??':
            return False
        visible_suits = [c[1] for c in board if c and c != '??']
        if not visible_suits:
            return False
        suit_counts = {s: visible_suits.count(s) for s in set(visible_suits)}
        dominant_suit = max(suit_counts, key=suit_counts.get)
        return suit_counts[dominant_suit] >= 3 and target[1] == dominant_suit

    def _betting_action(self, game_state, round_state, active, equity, continue_cost, pot, min_r, max_r):
        legal = round_state.legal_actions()
        street = round_state.street

        if street == 0:
            return self._preflop_action(round_state, active, equity, continue_cost, pot, legal, min_r, max_r)
        return self._postflop_action(round_state, active, equity, continue_cost, pot, legal, min_r, max_r)

    def _preflop_action(self, round_state, active, equity, continue_cost, pot, legal, min_r, max_r):
        if active == 0:
            if round_state.button == 0:
                if equity >= 0.40 and RaiseAction in legal:
                    return RaiseAction(max(min_r, min(int(2.5 * BIG_BLIND), max_r)))
                if equity >= 0.36 and CallAction in legal:
                    return CallAction()
                return FoldAction() if FoldAction in legal else CallAction()

            if equity >= 0.58 and RaiseAction in legal:
                return RaiseAction(max_r)
            if equity >= 0.46 and CallAction in legal:
                return CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if continue_cost == 0:
            if equity >= 0.56 and RaiseAction in legal:
                return RaiseAction(max(min_r, min(int(3 * BIG_BLIND), max_r)))
            return CheckAction() if CheckAction in legal else FoldAction()

        if equity >= 0.60 and RaiseAction in legal:
            raise_to = round_state.pips[active] + int(3.5 * continue_cost) + pot
            return RaiseAction(max(min_r, min(raise_to, max_r)))
        if equity >= 0.43 and CallAction in legal:
            return CallAction()
        return FoldAction() if FoldAction in legal else CallAction()

    def _postflop_action(self, round_state, active, equity, continue_cost, pot, legal, min_r, max_r):
        facing_bet = continue_cost > 0
        street = round_state.street
        weak_reveal_boost = self.opp_revealed_weak_card and self.opp_reveal_street is not None and street > self.opp_reveal_street

        if facing_bet:
            if equity >= 0.83 and RaiseAction in legal:
                return RaiseAction(max(min_r, min(round_state.pips[active] + int(pot * 0.80), max_r)))
            if equity >= (continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0) + 0.03:
                return CallAction() if CallAction in legal else CheckAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if equity >= 0.63 and RaiseAction in legal:
            frac = 0.70 if weak_reveal_boost else 0.65
            return RaiseAction(_raise_to(round_state, active, pot, min_r, max_r, frac))
        if equity >= 0.53 and RaiseAction in legal:
            frac = 0.55 if weak_reveal_boost else 0.50
            return RaiseAction(_raise_to(round_state, active, pot, min_r, max_r, frac))
        if equity < 0.40 and RaiseAction in legal and random.random() < 0.15:
            return RaiseAction(_raise_to(round_state, active, pot, min_r, max_r, 0.40))
        return CheckAction() if CheckAction in legal else FoldAction()

    def _select_redraw_target(self, my_cards, board, current_equity, dead_cards, current_street, round_state, active):
        _, candidate_sims = self._redraw_sims()

        best_target = None
        best_gain = 0.0

        if current_street == 0:
            street_threshold = 0.12
        elif current_street == 3:
            street_threshold = 0.06
        elif current_street == 4:
            street_threshold = 0.025
        else:
            street_threshold = 0.10

        for i, card in enumerate(my_cards):
            if card == '??':
                continue
            new_eq = mc_redraw_equity(my_cards, board, 'hole', i, excluded_extra=dead_cards, n_sims=candidate_sims)
            gain = new_eq - current_equity
            required = max(street_threshold, 0.025)
            if gain > required and gain > best_gain:
                best_gain = gain
                best_target = ('hole', i)

        visible_suits = [c[1] for c in board if c and c != '??']
        suit_counts = {s: visible_suits.count(s) for s in set(visible_suits)}
        dominant_suit = None
        if suit_counts:
            dominant_suit = max(suit_counts, key=suit_counts.get)

        for i, card in enumerate(board):
            if card == '??':
                continue
            new_eq = mc_redraw_equity(my_cards, board, 'board', i, excluded_extra=dead_cards, n_sims=candidate_sims)
            gain = new_eq - current_equity
            targeted = (
                self._is_targeted_board_swap(board, i)
                or (dominant_suit is not None and card[1] == dominant_suit and suit_counts.get(dominant_suit, 0) >= 3)
            )
            target_floor = 0.015 if targeted else 0.035
            required = max(street_threshold, target_floor)
            if gain > required and gain > best_gain:
                best_gain = gain
                best_target = ('board', i)

        if best_target is None:
            return None, current_equity
        return best_target, current_equity

    def _maybe_redraw(self, round_state, active, equity, pot, continue_cost, legal):
        if RedrawAction not in legal or round_state.redraws_used[active] or round_state.street >= 5:
            return None

        my_cards = round_state.hands[active]
        board = round_state.board
        dead_cards = list(self.opp_dead_cards) if self.opp_dead_cards else None
        current_street = round_state.street
        target, current_equity = self._select_redraw_target(
            my_cards,
            board,
            equity,
            dead_cards,
            current_street,
            round_state,
            active,
        )
        if target is None:
            return None

        base_action = self._betting_action(None, round_state, active, current_equity, continue_cost, pot, *(
            round_state.raise_bounds() if RaiseAction in legal else (0, 0)
        ))
        target_type, target_index = target
        return RedrawAction(target_type, target_index, base_action)

    def get_action(self, game_state, round_state, active):
        self.current_street = round_state.street

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
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0

        min_r = max_r = 0
        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()

        dead_cards = list(self.opp_dead_cards) if self.opp_dead_cards else None
        if street >= 3:
            n_sims = self._postflop_sims(street)
        else:
            n_sims = 250
        equity = mc_equity(my_cards, board, excluded_extra=dead_cards, n_sims=n_sims)

        redraw_action = self._maybe_redraw(round_state, active, equity, pot, continue_cost, legal)
        if redraw_action is not None:
            return redraw_action

        if street == 0:
            return self._preflop_action(round_state, active, equity, continue_cost, pot, legal, min_r, max_r)

        return self._postflop_action(round_state, active, equity, continue_cost, pot, legal, min_r, max_r)


if __name__ == '__main__':
    run_bot_enhanced(Player(), parse_args())
