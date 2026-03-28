import random
import eval7
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction
from skeleton.states import GameState, RoundState
from skeleton.bot import Bot

class Player(Bot):
    '''
    A professional-grade pokerbot using Monte Carlo Simulation and Pot Odds.
    '''

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called at the start of a new round.
        '''
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called at the end of a round.
        '''
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - decision-making logic.
        '''
        legal_actions = round_state.legal_actions
        street = round_state.street # 0=Preflop, 3=Flop, 4=Turn, 5=River
        my_cards = [eval7.Card(c) for c in round_state.hands[active]]
        board_cards = [eval7.Card(c) for c in round_state.board]
        
        # 1. PRE-FLOP (Quick lookup to save time)
        if street == 0:
            return self.get_preflop_action(my_cards, legal_actions, round_state)

        # 2. CALCULATE WIN PROBABILITY (Equity)
        # We run 750 simulations to see our winning chances.
        equity = self.calculate_equity(my_cards, board_cards, iterations=750)
        
        # 3. GET POT ODDS
        pot_total = round_state.pot
        continue_cost = round_state.continue_cost
        pot_odds = continue_cost / (pot_total + continue_cost) if (pot_total + continue_cost) > 0 else 0
        
        # Determine raise limits
        min_raise, max_raise = round_state.raise_bounds() if RaiseAction in legal_actions else (0, 0)

        # 4. ACTION LOGIC
        
        # A. VALUE BETTING: We are very likely to win (>75%)
        if equity > 0.75:
            if RaiseAction in legal_actions:
                # Bet 75% of the pot to build value
                raise_amount = max(min_raise, min(max_raise, int(pot_total * 0.75)))
                return RaiseAction(raise_amount)
            return CallAction()

        # B. SEMI-BLUFFING: Mid-strength (40-60%) + Randomness
        # We bluff 15% of the time to stay unpredictable.
        elif 0.4 < equity < 0.6 and random.random() < 0.15:
            if RaiseAction in legal_actions:
                raise_amount = max(min_raise, min(max_raise, int(pot_total * 0.5)))
                return RaiseAction(raise_amount)

        # C. THE "MATH CALL": Equity is better than the price we are paying
        if equity > pot_odds:
            if CheckAction in legal_actions:
                return CheckAction()
            return CallAction()

        # D. PURE BLUFF: We have a weak hand but try to steal the pot if it's cheap
        if equity < 0.2 and continue_cost == 0 and random.random() < 0.05:
            if RaiseAction in legal_actions:
                raise_amount = max(min_raise, min(max_raise, int(pot_total * 0.4)))
                return RaiseAction(raise_amount)

        # E. DEFAULT: Check if free, otherwise Fold
        return CheckAction() if CheckAction in legal_actions else FoldAction()

    def get_preflop_action(self, my_cards, legal_actions, round_state):
        '''
        Simple rules for the first round of betting.
        '''
        ranks = [c.rank for c in my_cards]
        # Play if we have a Pair, an Ace, or a King
        if len(set(ranks)) < 2 or max(ranks) >= 11: 
            if RaiseAction in legal_actions:
                min_r, _ = round_state.raise_bounds()
                return RaiseAction(min_r)
            return CallAction()
        
        # Fold everything else to save chips
        return CheckAction() if CheckAction in legal_actions else FoldAction()

    def calculate_equity(self, my_cards, board_cards, iterations):
        '''
        Monte Carlo Simulation to find win %
        '''
        deck = eval7.Deck()
        for card in (my_cards + board_cards):
            deck.cards.remove(card)

        wins = 0
        for _ in range(iterations):
            deck.shuffle()
            num_board_needed = 5 - len(board_cards)
            draw = deck.peek(num_board_needed + 2)
            
            opp_cards = draw[:2]
            rest_of_board = draw[2:]
            final_board = board_cards + rest_of_board
            
            my_score = eval7.evaluate(my_cards + final_board)
            opp_score = eval7.evaluate(opp_cards + final_board)
            
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                wins += 0.5 # Tie/Split pot
                
        return wins / iterations