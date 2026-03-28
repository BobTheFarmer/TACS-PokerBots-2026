"""
Hyper-aggressive pressure bot for TACS PokerBots 2026.
"""

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


def _excluded_cards(excluded_extra=None):
    excluded = set()
    if excluded_extra:
        excluded |= {c for c in excluded_extra if c and c != '??'}
    return excluded


def _fill_unknowns(cards, draws):
    filled = list(cards)
    draw_index = 0
    for idx, card in enumerate(filled):
        if card == '??':
            filled[idx] = draws[draw_index]
            draw_index += 1
    return filled, draw_index


def mc_equity(my_cards, board, excluded_extra=None, n_sims=200):
    excluded = {c for c in my_cards + board if c and c != '??'}
    excluded |= _excluded_cards(excluded_extra)

    my_unknown = sum(1 for c in my_cards if c == '??')
    board_unknown = sum(1 for c in board if c == '??')
    avail = [c for c in ALL_CARDS if c not in excluded]
    n_need = my_unknown + board_unknown + 2 + (5 - len(board))

    if n_need > len(avail):
        return 0.5

    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        draws = [avail[i] for i in idx]

        my_filled, used = _fill_unknowns(my_cards, draws)
        board_filled, used_board = _fill_unknowns(board, draws[used:])
        used += used_board
        opp = [draws[used], draws[used + 1]]
        used += 2
        full_board = board_filled + draws[used:]

        my_score = pkrbot.evaluate(my_filled + full_board)
        opp_score = pkrbot.evaluate(opp + full_board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

    return wins / n_sims


def mc_redraw_equity(my_cards, board, target_type, target_index,
                     excluded_extra=None, n_sims=120):
    excluded = {c for c in my_cards + board if c and c != '??'}
    excluded |= _excluded_cards(excluded_extra)

    my_slots = [i for i, c in enumerate(my_cards) if c == '??']
    board_slots = [i for i, c in enumerate(board) if c == '??']

    if target_type == 'hole':
        old_card = my_cards[target_index]
        if old_card != '??':
            excluded.add(old_card)
        if target_index not in my_slots:
            my_slots = my_slots + [target_index]
    else:
        old_card = board[target_index]
        if old_card != '??':
            excluded.add(old_card)
        if target_index not in board_slots:
            board_slots = board_slots + [target_index]

    avail = [c for c in ALL_CARDS if c not in excluded]
    n_need = len(my_slots) + len(board_slots) + 2 + (5 - len(board))

    if n_need > len(avail):
        return 0.5

    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        draws = [avail[i] for i in idx]

        my_filled = list(my_cards)
        board_filled = list(board)
        draw_ptr = 0
        for pos in my_slots:
            my_filled[pos] = draws[draw_ptr]
            draw_ptr += 1
        for pos in board_slots:
            board_filled[pos] = draws[draw_ptr]
            draw_ptr += 1
        opp = [draws[draw_ptr], draws[draw_ptr + 1]]
        draw_ptr += 2
        full_board = board_filled + draws[draw_ptr:]

        my_score = pkrbot.evaluate(my_filled + full_board)
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
        self.rounds_seen = 0

    def on_opponent_reveal(self, old_card, target_type, target_index):
        if old_card and old_card != '??':
            self.opp_dead_cards.add(old_card)

    def handle_new_round(self, game_state, round_state, active):
        self.opp_dead_cards = set()
        self.rounds_seen += 1

    def handle_round_over(self, game_state, terminal_state, active):
        delta = terminal_state.deltas[active]
        print(
            f"[aggro_pressure] round={game_state.round_num} delta={delta} bankroll={game_state.bankroll}",
            file=__import__('sys').stderr,
        )

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
        pot_odds = continue_cost / (pot + continue_cost) if continue_cost > 0 else 0.0
        dead = list(self.opp_dead_cards) if self.opp_dead_cards else None

        if RaiseAction in legal:
            min_r, max_r = round_state.raise_bounds()
        else:
            min_r = max_r = 0

        def bet(frac):
            raw = round_state.pips[active] + int(pot * frac)
            return max(min_r, min(raw, max_r))

        def raise_to(amount):
            return max(min_r, min(amount, max_r))

        current_eq = mc_equity(my_cards, board, excluded_extra=dead, n_sims=200)

        if RedrawAction in legal and not round_state.redraws_used[active] and street < 5:
            redraw_action = self._decide_redraw(my_cards, board, legal, current_eq, dead, pot, continue_cost, bet, raise_to)
            if redraw_action is not None:
                return redraw_action

        if street == 0:
            return self._preflop_action(
                legal,
                current_eq,
                continue_cost,
                pot,
                active,
                round_state.button,
                bet,
                raise_to,
                min_r,
                max_r,
                round_state,
            )
        return self._postflop_action(
            legal,
            current_eq,
            pot_odds,
            continue_cost,
            pot,
            bet,
            raise_to,
        )

    def _decide_redraw(self, my_cards, board, legal, current_eq, dead, pot, continue_cost, bet, raise_to):
        best_gain = 0.02
        best_target = None

        for idx in range(2):
            if my_cards[idx] == '??':
                continue
            new_eq = mc_redraw_equity(my_cards, board, 'hole', idx, excluded_extra=dead, n_sims=120)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain = gain
                best_target = ('hole', idx)

        for idx in range(len(board)):
            new_eq = mc_redraw_equity(my_cards, board, 'board', idx, excluded_extra=dead, n_sims=120)
            gain = new_eq - current_eq
            if gain > best_gain:
                best_gain = gain
                best_target = ('board', idx)

        if best_target is None:
            return None

        target_type, target_index = best_target
        if continue_cost > 0:
            if RaiseAction in legal:
                inner = RaiseAction(raise_to(bet(0.75)))
            elif CallAction in legal:
                inner = CallAction()
            else:
                inner = FoldAction()
        else:
            if RaiseAction in legal:
                inner = RaiseAction(raise_to(bet(0.70)))
            elif CheckAction in legal:
                inner = CheckAction()
            else:
                inner = CallAction() if CallAction in legal else FoldAction()
        return RedrawAction(target_type, target_index, inner)

    def _preflop_action(self, legal, equity, continue_cost, pot, active, button, bet, raise_to, min_r, max_r, round_state):
        is_sb = active == 0
        if is_sb:
            if button == 0:
                if equity >= 0.34 and RaiseAction in legal:
                    return RaiseAction(raise_to(int(3 * BIG_BLIND)))
                return FoldAction() if FoldAction in legal else CheckAction()
            if equity >= 0.55 and RaiseAction in legal:
                return RaiseAction(raise_to(max_r))
            if equity >= 0.45 and CallAction in legal:
                return CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if continue_cost == 0:
            if RaiseAction in legal:
                return RaiseAction(raise_to(int(4 * BIG_BLIND)))
            return CheckAction()

        if equity >= 0.52 and RaiseAction in legal:
            opp_pip = round_state.pips[1 - active]
            amount = int(opp_pip + 2.5 * continue_cost)
            return RaiseAction(raise_to(amount))
        if equity >= 0.38 and CallAction in legal:
            return CallAction()
        return FoldAction() if FoldAction in legal else CallAction()

    def _postflop_action(self, legal, equity, pot_odds, continue_cost, pot, bet, raise_to):
        facing_bet = continue_cost > 0
        if facing_bet:
            if equity >= pot_odds + 0.20 and RaiseAction in legal:
                return RaiseAction(raise_to(bet(0.80)))
            if equity >= pot_odds - 0.02 and CallAction in legal:
                return CallAction()
            return FoldAction() if FoldAction in legal else CallAction()

        if equity >= 0.55 and RaiseAction in legal:
            return RaiseAction(bet(0.80))
        if 0.35 <= equity < 0.55 and RaiseAction in legal and random.random() < 0.70:
            return RaiseAction(bet(0.65))
        if equity < 0.35 and RaiseAction in legal and random.random() < 0.35:
            return RaiseAction(bet(0.50))
        return CheckAction() if CheckAction in legal else FoldAction()


if __name__ == '__main__':
    run_bot_enhanced(Player(), parse_args())
