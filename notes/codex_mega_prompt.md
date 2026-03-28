# TACS PokerBots 2026 — Codex Multi-Agent Bot Factory

## YOUR TASK

You are the orchestrator. Spawn **5 worker sub-agents in parallel**, each building exactly one poker bot. Every agent works in its own isolated directory and must not touch any other directory. After all agents complete, verify that each bot directory has the required files.

Spawn all 5 agents at once with:
> "Have worker build bot_tight_aggressive as described below. Have worker build bot_aggro_pressure as described below. Have worker build bot_exploiter as described below. Have worker build bot_gto_balanced as described below. Have worker build bot_redraw_hunter as described below. All five may run in parallel. Each worker must only write files inside its assigned directory."

---

## REPO CONTEXT

**Repo location:** `/path/to/TACS-PokerBots-2026/` (use the actual repo root)

**What already exists — DO NOT TOUCH:**
```
engine.py              — game engine (runs matches)
config.py              — match configuration
requirements.txt       — pkrbot, numpy, numba, cython, pytest
check_call_bot/        — reference bot (passive, do not modify)
python_skeleton/       — THE REFERENCE BOT (already written, do not modify)
  player.py            — full MC equity bot, read this for patterns
  commands.json
  skeleton/
    actions.py
    bot.py
    runner.py
    states.py
```

**What each worker creates (one directory each, nothing else):**
```
bot_tight_aggressive/
bot_aggro_pressure/
bot_exploiter/
bot_gto_balanced/
bot_redraw_hunter/
```

---

## REQUIRED FILE STRUCTURE FOR EVERY BOT

Each bot directory must contain **exactly these files**:

### 1. `commands.json`
```json
{
  "build": [],
  "run": ["python3", "player.py"]
}
```

### 2. `skeleton/` directory — COPY VERBATIM from `python_skeleton/skeleton/`
Copy all four files exactly:
- `skeleton/actions.py`
- `skeleton/bot.py`
- `skeleton/runner.py`
- `skeleton/states.py`

Do not modify them. The engine depends on the exact file content.

### 3. `player.py` — the bot implementation (see per-bot specs below)

---

## COMPLETE API REFERENCE

Every `player.py` must follow this pattern exactly.

### Imports
```python
import random
import socket
import numpy as np
import pkrbot

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, Runner
```

### The `Player` class must extend `Bot` and implement exactly these three methods:
```python
class Player(Bot):
    def handle_new_round(self, game_state, round_state, active): ...
    def handle_round_over(self, game_state, terminal_state, active): ...
    def get_action(self, game_state, round_state, active): ...
```

### `get_action` receives:
```python
street = round_state.street        # 0=preflop, 3=flop, 4=turn, 5=river
my_cards = round_state.hands[active]           # ['Ah', 'Kd'] — always 2 strings
board    = round_state.board                   # [] / ['3h','4d','Jc'] / 4 or 5 cards
my_pip   = round_state.pips[active]            # chips I've put in this street
opp_pip  = round_state.pips[1 - active]        # chips opponent put in
my_stack  = round_state.stacks[active]         # chips remaining
opp_stack = round_state.stacks[1 - active]
continue_cost = opp_pip - my_pip               # chips needed to call
pot = 2 * STARTING_STACK - my_stack - opp_stack   # total pot
legal = round_state.legal_actions()            # set of legal action classes
min_r, max_r = round_state.raise_bounds()      # only when RaiseAction in legal
```

### Position:
- `active == 0` → SB/button (acts FIRST preflop, LAST postflop)
- `active == 1` → BB (acts SECOND preflop, FIRST postflop)
- `round_state.button == 0` means it's SB's opening action (very first preflop decision)

### Actions:
```python
FoldAction()
CallAction()
CheckAction()
RaiseAction(amount)             # amount = total raise-to, must satisfy min_r <= amount <= max_r
RedrawAction(target_type, target_index, inner_action)
  # target_type: 'hole' or 'board'
  # target_index: 0 or 1 for hole; 0..2 on flop, 0..3 on turn for board
  # inner_action: FoldAction|CallAction|CheckAction|RaiseAction (the betting action paired with redraw)
```

Always check `legal_actions()` before returning. `RedrawAction` is in legal_actions only when:
- `street < 5` (not on river)
- `round_state.redraws_used[active]` is False

### `game_state` fields:
```python
game_state.bankroll    # cumulative chip delta so far this match
game_state.game_clock  # seconds remaining for this bot
game_state.round_num   # current hand number (1-indexed)
```

### Card format:
- Rank: `2 3 4 5 6 7 8 9 T J Q K A`
- Suit: `h d s c`
- Example: `'Ah'` = Ace of hearts, `'Td'` = Ten of diamonds
- Unknown card (opponent's redrawed hole card locally): `'??'`

### Hand evaluation:
```python
score = pkrbot.evaluate(cards)   # cards = list of exactly 7 strings (2 hole + 5 board)
# Higher score = better hand. Use for comparison only, not as absolute value.
```

### Bet sizing helper pattern (copy this into every bot):
```python
if RaiseAction in legal:
    min_r, max_r = round_state.raise_bounds()
else:
    min_r = max_r = 0

def bet(frac):
    raw = round_state.pips[active] + int(pot * frac)
    return max(min_r, min(raw, max_r))
```

### Monte Carlo equity function (copy this into every bot):
```python
RANKS = '23456789TJQKA'
SUITS = 'hdsc'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]

def mc_equity(my_cards, board, excluded_extra=None, n_sims=300):
    excluded = set(my_cards) | set(board)
    if excluded_extra:
        excluded |= {c for c in excluded_extra if c and c != '??'}
    avail = [c for c in ALL_CARDS if c not in excluded]
    n_need = 2 + (5 - len(board))
    if n_need > len(avail):
        return 0.5
    wins = 0.0
    for _ in range(n_sims):
        idx = np.random.choice(len(avail), n_need, replace=False)
        opp = [avail[idx[0]], avail[idx[1]]]
        full_board = board + [avail[idx[i]] for i in range(2, n_need)]
        my_score  = pkrbot.evaluate(my_cards + full_board)
        opp_score = pkrbot.evaluate(opp + full_board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
    return wins / n_sims
```

### EnhancedRunner (copy this into every bot — enables revealed card tracking):
```python
class EnhancedRunner(Runner):
    def _apply_action_clause(self, round_state, action_clause):
        actor = round_state.button % 2
        old_card    = self._pending_redraw_old_card.get(actor)
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
```

### Entry point (required at the bottom of every player.py):
```python
if __name__ == '__main__':
    run_bot_enhanced(Player(), parse_args())
```

---

## GAME RULES SUMMARY

- No-limit Texas Hold'em, 2 players, 300 hands per match
- 400 chips per player per hand (resets every hand), SB=1, BB=2
- Streets: 0=preflop, 3=flop, 4=turn, 5=river
- **REDRAW**: once per hand, before river (street < 5), each player may swap one hole card OR one board card. The old card is REVEALED to the opponent.
- Total time budget: 180 seconds for 300 hands (~0.6s average per hand)
- Single-elimination tournament bracket

---

## PER-BOT SPECIFICATIONS

---

### BOT 1: `bot_tight_aggressive/`

**Codename:** Tight-Aggressive Value Machine
**Philosophy:** Win by only playing premium hands and extracting maximum value when strong. Rarely bluff. Punish calling stations by only putting money in with the best of it.

**Preflop strategy:**
- SB opening: only raise with equity >= 0.50 (roughly top 35% of hands: big pairs, broadways, suited connectors). Raise to 3x BB. Fold everything else — do NOT limp.
- BB facing SB raise: 3-bet only with equity >= 0.65 (top 15%). Call with equity >= 0.48. Fold the rest.
- SB facing 3-bet: 4-bet shove only with equity >= 0.65. Call with equity >= 0.55. Fold the rest.
- BB when SB limped: raise to 4x BB with equity >= 0.58. Otherwise check.

**Postflop strategy:**
- Compute equity with n_sims=400 (more accurate, this bot can afford it since it folds often preflop)
- Value bet only when equity >= 0.70: bet 75% pot
- Thin value when equity >= 0.62: bet 55% pot
- When equity 0.50-0.62: check. If opponent bets, call only if equity >= pot_odds + 0.05
- When equity < 0.50: fold to bets. Never bluff (bluff_freq = 0).
- Exception: call any bet with equity >= pot_odds (pure pot-odds calling)

**Redraw strategy:**
- Only use redraw if equity gain > 0.06 (high bar — only clear improvements)
- Only consider hole card swaps, not board card swaps
- Always pair redraw with the appropriate betting action based on equity

**Handle round over:** track win/loss for debugging (just print to stderr)

---

### BOT 2: `bot_aggro_pressure/`

**Codename:** Hyper-Aggressive Pressure Bot
**Philosophy:** Win by applying relentless pressure. Fold equity is free money. Most bots can't handle constant aggression. Never give a free card.

**Preflop strategy:**
- SB opening: raise 95% of hands (equity >= 0.34). Raise to 3x BB. Never limp.
- BB facing SB open: 3-bet with equity >= 0.52 (very frequent 3-betting). Call with equity >= 0.38. Fold the rest.
- SB facing 3-bet: 4-bet with equity >= 0.55. Call with equity >= 0.45. Fold the rest.
- BB when SB limped: always raise to 4x BB (100% raise frequency from BB option).

**Postflop strategy:**
- n_sims=200 (faster decisions, this bot acts on many hands)
- When NO bet to face (checking opportunity):
  - Always bet if equity >= 0.55: bet 80% pot
  - Bet as semi-bluff/bluff if equity 0.35-0.55: bet 65% pot with 70% frequency (random.random() < 0.70)
  - Bet pure bluff if equity < 0.35: bet 50% pot with 35% frequency
  - Otherwise check
- When FACING a bet:
  - Raise (re-raise) with equity >= pot_odds + 0.20: raise to 2.2x their bet
  - Call with equity >= pot_odds - 0.02 (calling slightly wide)
  - Fold otherwise (but this rarely happens given the wide calling range)

**Redraw strategy:**
- Use redraw aggressively: any gain > 0.02 (low threshold)
- Consider both hole cards and board cards
- Pair redraw with a bet when possible (RedrawAction wrapping RaiseAction) to pressure opponent simultaneously

**Timing:** This bot makes fast decisions (low n_sims) to stay well within time budget.

---

### BOT 3: `bot_exploiter/`

**Codename:** The Exploit Engine
**Philosophy:** Start with a solid equity-based foundation. Continuously track opponent patterns. Once enough data is collected (20+ hands), shift strategy to maximally exploit the identified leaks. Every weakness in the opponent's game should be punished systematically.

**This bot has two layers: BASE layer (always running) + EXPLOIT layer (activates after data collection)**

**State to track in `__init__`:**
```python
# Per-hand reset
self.opp_dead_cards = set()
self.preflop_aggressor = False  # did opp raise preflop this hand?
self.we_raised_preflop = False
self.we_cbetted = False         # did we cbet this hand?

# Cumulative stats (never reset)
self.hands = 0
self.opp_vpip = 0           # hands opponent voluntarily entered pot
self.opp_pfr = 0            # hands opponent raised preflop
self.opp_fold_to_3bet = 0   # times opp folded to our 3-bet
self.opp_3bet_opps = 0      # times we 3-bet opponent
self.opp_fold_to_cbet = 0   # times opp folded to our continuation bet
self.opp_cbet_opps = 0      # times we had a cbet opportunity
self.opp_limp_count = 0     # times opp limped (SB call without raise)
self.opp_redraw_count = 0   # times opp used their redraw
self.opp_fold_postflop = 0  # times opp folded postflop to our bet
self.opp_postflop_bets = 0  # times we bet postflop and opp had to respond
```

**Stat update logic:**
- In `handle_new_round`: detect if opponent raised preflop by checking if button>0 and pips show raise (track this via `_opp_preflop_action` state set in get_action)
- In `handle_round_over`:
  - If terminal_state.previous_state.hands[1-active] shows a fold pattern (delta = opponent lost their contribution), opp folded. Update fold stats.
  - Track VPIP: if opp_pip > BIG_BLIND at end of preflop, opp entered voluntarily
- In `get_action`: update running stats when you observe opponent actions (infer from pip changes between calls to get_action)
- Call `on_opponent_reveal` to track redraw count

**BASE strategy (same structure as python_skeleton's bot — copy the mc_equity + EnhancedRunner pattern):**
- SB: open equity >= 0.40, raise 2.5x BB
- BB: 3-bet equity >= 0.60, call equity >= 0.43
- Postflop: value bet equity >= 0.65 at 65% pot, call if equity >= pot_odds + 0.02

**EXPLOIT layer — activated when `self.hands >= 20`:**

Compute these rates:
```python
fold_to_3bet_rate  = self.opp_fold_to_3bet / max(self.opp_3bet_opps, 1)
fold_to_cbet_rate  = self.opp_fold_to_cbet / max(self.opp_cbet_opps, 1)
vpip_rate          = self.opp_vpip / max(self.hands, 1)
pfr_rate           = self.opp_pfr  / max(self.hands, 1)
redraw_rate        = self.opp_redraw_count / max(self.hands, 1)
```

Apply these exploit overrides ON TOP of the base strategy:

1. **If `fold_to_3bet_rate > 0.55`**: Lower 3-bet threshold to equity >= 0.45 from BB. We 3-bet much lighter because opponent folds.
2. **If `fold_to_cbet_rate > 0.60`**: Always cbet any flop we raised preflop (regardless of equity). Bet 60% pot. Pure bluff if needed.
3. **If `vpip_rate < 0.40`** (opponent is very tight): Steal preflop more — SB opens equity >= 0.30. Bluff postflop more (bluff frequency up to 40% when checking with weak hand).
4. **If `vpip_rate > 0.75`** (opponent is loose/passive, calling station): Tighten value betting threshold to equity >= 0.70 only. Never bluff. Bet very large for value (90% pot).
5. **If `pfr_rate < 0.20`** (opponent is passive, rarely raises): Call 3-bets wider. Float their bets with equity >= pot_odds - 0.05.
6. **If `redraw_rate > 0.5`** (opponent redraws frequently): Narrow opponent range on later streets. When opponent redrawed this hand, increase MC sims to 500 and track `opp_dead_cards` (their revealed card) for better equity estimation.

**Redraw strategy for exploiter:**
- Base threshold: gain > 0.03
- If opponent redraws frequently, use their redraw timing as a signal: update range, run more sims
- `on_opponent_reveal` stores card in `opp_dead_cards` AND increments `opp_redraw_count`

**Stat update details for get_action:**
- Store `self._last_opp_pip` between calls. When opp_pip increases without our action, opponent bet/raised.
- Track in `handle_new_round` the initial state of `round_state` for reference.
- After 20 hands of data, the exploit layer can meaningfully differentiate opponent styles.

---

### BOT 4: `bot_gto_balanced/`

**Codename:** GTO-Approximator
**Philosophy:** Be unexploitable. Use mixed strategies based on hand strength buckets so the opponent cannot identify a pattern to exploit. Randomize actions within defined frequency ranges. This bot is designed to NOT leak exploitable tendencies.

**Hand strength bucketing:**
Compute equity at each decision point, then map to a bucket:
```
NUTS    = equity >= 0.78    (top of range)
VALUE   = equity >= 0.62
MEDIUM  = equity >= 0.50
MARGINAL= equity >= 0.40
BLUFF_C = equity >= 0.30    (bluff candidate)
TRASH   = equity < 0.30
```

**Mixed strategy frequencies — preflop:**
- SB open:
  - NUTS/VALUE/MEDIUM: raise 2.5x BB (100%)
  - MARGINAL: raise 2.5x BB (60%) or fold (40%)
  - BLUFF_C: raise 2.5x BB (25%) or fold (75%)
  - TRASH: fold (100%)
- BB vs SB raise:
  - NUTS: 3-bet (100%) to 3.5x SB's raise
  - VALUE: 3-bet (40%) or call (60%)
  - MEDIUM: call (100%)
  - MARGINAL: call (50%) or fold (50%)
  - BLUFF_C: fold (80%) or call (20%)
  - TRASH: fold (100%)

**Mixed strategy frequencies — postflop when no bet (checking opportunity):**
- NUTS: bet 70% pot (100%)
- VALUE: bet 55% pot (80%) or check (20%)
- MEDIUM: bet 45% pot (40%) or check (60%)
- MARGINAL: check (85%) or bet as bluff 40% pot (15%)
- BLUFF_C: bet 40% pot (22%) or check (78%)   ← maintains bluff-to-value ratio
- TRASH: check (100%)

**Mixed strategy frequencies — postflop when facing a bet:**
- NUTS: raise 2.5x (55%) or call (45%)
- VALUE: raise 2.2x (25%) or call (75%)
- MEDIUM: call if equity >= pot_odds, else fold
- MARGINAL: call if equity >= pot_odds + 0.05, else fold
- BLUFF_C/TRASH: fold (100%)

**Implementation:**
Use `random.random()` to select between actions for each bucket. This ensures the bot's actions can't be predicted from a single observation. Example:
```python
bucket = get_bucket(equity)
r = random.random()
if bucket == 'VALUE':
    if r < 0.80:
        return RaiseAction(bet(0.55))
    else:
        return CheckAction()
```

**Redraw:** Run mc_redraw_equity for each hole card. Use mixed frequency:
- If gain > 0.05: redraw (100%)
- If gain > 0.02: redraw (50%)
- If gain > 0.00: redraw (15%)
- Otherwise: don't redraw

**Bet sizing variation:** Add slight randomness to sizes to avoid fixed-pattern exploitation:
- Value bet range: 65-80% pot (random.uniform(0.65, 0.80))
- Bluff range: 38-52% pot (random.uniform(0.38, 0.52))

**n_sims:** 250 (balanced accuracy vs speed)

---

### BOT 5: `bot_redraw_hunter/`

**Codename:** Redraw Specialist
**Philosophy:** Master the unique redraw mechanic. Use deeper MC analysis for redraw decisions. Track what the opponent reveals when they redraw to narrow their range precisely. Also strategically swap board cards to disrupt opponent's likely draws.

**Core differentiator:** This bot does MORE work on redraw decisions than any other bot. It also uses opponent's revealed cards more aggressively to update equity estimates.

**Redraw state tracking:**
```python
self.opp_dead_cards = set()          # revealed by opponent's redraws
self.opp_redrawed_this_hand = False
self.opp_redraw_target_type = None   # 'hole' or 'board'
self.opp_redraw_target_index = None
self.opp_redraw_count = 0
self.opp_redraw_streets = []         # which streets opponent redraws on
```

**`on_opponent_reveal` implementation:**
```python
def on_opponent_reveal(self, old_card, target_type, target_index):
    self.opp_dead_cards.add(old_card)
    self.opp_redrawed_this_hand = True
    self.opp_redraw_target_type = target_type
    self.opp_redraw_target_index = target_index
    self.opp_redraw_count += 1
    self.opp_redraw_streets.append(self.current_street)
```

**Redraw decision engine (deeper than other bots):**
Use n_sims=250 for current equity and n_sims=200 for each redraw candidate.
Evaluate ALL possible targets:
- Both hole cards: indices 0 and 1
- All board cards: indices 0 to len(board)-1
Include `opp_dead_cards` in excluded set for all MC calls.

For board card swaps, add extra logic: if a board card appears to complete the opponent's likely draw (flush card on board after opponent redrawed on flop), prioritize swapping it with higher threshold:
```python
# Detect flush-completing board cards
suits_on_board = [c[1] for c in board]
flush_suit = max(set(suits_on_board), key=suits_on_board.count)
if suits_on_board.count(flush_suit) >= 3:
    # Flush is possible on board — consider swapping the third flush card
    flush_cards = [i for i, c in enumerate(board) if c[1] == flush_suit]
    # These are high-priority swap targets even at lower gain threshold (0.01)
```

**Threshold:** 0.025 for hole cards, 0.015 for targeted board swaps (opponent draws), 0.035 for other board swaps.

**Preflop:** Standard ranges (copy from python_skeleton):
- SB open: equity >= 0.40, raise 2.5x BB
- BB vs raise: 3-bet equity >= 0.60, call equity >= 0.43

**Postflop:** Standard equity-based decisions BUT with higher-quality equity estimates:
- n_sims = 350 for all postflop decisions (more accurate)
- Always pass `opp_dead_cards` to mc_equity (critical — their revealed card is excluded)
- If `opp_redrawed_this_hand`: boost n_sims to 500 on subsequent streets (opponent's range is narrowed by their reveal, making equity calculation more meaningful)
- Value bet: equity >= 0.63, bet 65% pot
- Thin value: equity >= 0.53, bet 50% pot
- Call: equity >= pot_odds + 0.03
- Bluff: 15% frequency with equity < 0.40

**When opponent reveals a weak card (low rank, e.g. rank <= '8'):**
- Inference: opponent was likely weak before the redraw → their hand may be stronger now
- Increase bet sizing slightly on later streets to price out draws they might be on

**Redraw timing strategy:**
- Prefer to redraw on the TURN (street=4) rather than flop (street=3) when possible
- On flop: only redraw if gain > 0.06 (save the redraw for when we have more info)
- On turn: redraw if gain > 0.025 (more board info = better decision)
- Exception: if flop gain is very high (> 0.10), take it immediately

**`handle_new_round`:**
```python
self.opp_dead_cards = set()
self.opp_redrawed_this_hand = False
self.opp_redraw_target_type = None
self.opp_redraw_target_index = None
self.current_street = 0
```

Update `self.current_street = round_state.street` at the start of `get_action`.

---

## IMPORTANT CONSTRAINTS FOR ALL WORKERS

1. **Scope isolation:** Each worker creates files ONLY in its assigned directory. No worker touches `python_skeleton/`, `check_call_bot/`, `engine.py`, `config.py`, or any other worker's directory.

2. **Copy skeleton verbatim:** The `skeleton/` subdirectory must be copied exactly from `python_skeleton/skeleton/`. Do not modify any skeleton file. If the copy command fails, read each file and write it.

3. **No external imports:** Only `random`, `socket`, `numpy`, `pkrbot`, `numba` (optional), and standard library are available. No `sklearn`, `scipy`, `torch`, etc.

4. **Always validate actions:** Check `legal_actions()` before returning any action. Never return an action that is not in the legal set.

5. **Always safe bet sizing:** When returning RaiseAction, always clamp: `max(min_r, min(amount, max_r))`.

6. **Handle `??` cards:** Opponent's hole cards may be `'??'` after they redraw. Skip `'??'` when building excluded sets for MC.

7. **Entry point:** Every player.py must end with:
   ```python
   if __name__ == '__main__':
       run_bot_enhanced(Player(), parse_args())
   ```

8. **Time awareness:** Respect the 180s total budget. Use lower n_sims on simpler decisions. Avoid loops that could exceed 500ms per decision.

9. **Read python_skeleton/player.py first** before writing any bot. It contains the complete reference implementation of mc_equity, mc_redraw_equity, EnhancedRunner, and run_bot_enhanced. Copy these helpers into each bot — don't reinvent them.

---

## VERIFICATION CHECKLIST (orchestrator runs after all workers finish)

After all 5 workers complete, verify each bot directory has:
- [ ] `commands.json` with correct content
- [ ] `skeleton/actions.py`
- [ ] `skeleton/bot.py`
- [ ] `skeleton/runner.py`
- [ ] `skeleton/states.py`
- [ ] `player.py` that syntax-checks cleanly: `python3 -c "import ast; ast.parse(open('player.py').read())"` (run from within each bot dir)
- [ ] `player.py` imports succeed (excluding pkrbot which needs compilation): check all other imports work

Report any missing files or syntax errors.
