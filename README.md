# VOID: Chess Benchmark

VOID is a spatial world-model architecture. Chess is one benchmark variation of it: a compact testbed for latent spatial reasoning, action prediction, and active-inference-style policy learning.

This chess instantiation operates directly on board tensors and moves, not text notation or rendered pixels. In current pre-RL benchmarks, it plays above 1650 Elo; the next stage is reinforcement learning.

![VOID chess playing](chess.gif)

![VOID architecture](void_architecture.svg)

## Architecture

VOID encodes spatial states as tensors and keeps their topology throughout the network. In the chess variant, the state is an `8x8` board tensor. The core block shifts information across the board in the center, orthogonal, and diagonal directions, then contracts those local views into a latent map.

After encoding, a shared recurrent "ghost" loop refines the latent board state for a configurable number of thinking steps. The same weights are reused at every step, so inference can trade compute for stronger decisions by increasing `n_think`.

## World Model + Active Inference

The chess variant is trained not only to imitate strong moves, but also to model the consequences of actions. Given a board and a candidate move, the action-conditioned world-model head predicts the next board state in the same spatial representation. This adds a chess-physics objective alongside policy learning: pieces, turns, captures, and transitions must become part of the latent dynamics.

The policy is active-inference-style: the model refines an internal latent state, evaluates possible action pressure through value and terminal-distance heads, and learns to prefer moves that lead toward better future states. In endgames, the terminal-distance objective adds conversion pressure, encouraging the model to resolve winning positions instead of only predicting that they are good.

The architecture includes:

- policy heads for `from_square` and `to_square`
- a value head for game outcome prediction
- an action-conditioned world-model head for next-board prediction
- a terminal-distance head for endgame conversion pressure
- optional legality and curriculum auxiliary losses

## Curriculum

The chess benchmark is trained as a staged curriculum:

1. Tactics from Lichess puzzles
2. Foundation training on strong human games
3. Opening positions
4. Middlegame positions
5. Endgame positions
6. Optional reinforcement learning against Stockfish

Later phases mix in replay from earlier data so the model does not forget tactics and full-game behavior while specializing. The current checkpoint is pre-RL; reinforcement learning is the next step.

## Files

- `void.py` contains the model architecture.
- `train.py` contains the curriculum trainer, cache builders, held-out evaluation, and benchmark logic.
- `void_architecture.svg` is a compact architecture diagram.
- `chess.gif` shows the chess benchmark variant playing.

## Example

```bash
python train.py \
  --phase 1 \
  --stop-after-phase 1 \
  --all-steps \
  --use-gm-cache \
  --gm-cache-samples 10000000 \
  --phase12-batch 768 \
  --num-workers 4
```

Checkpoints and dataset caches are intentionally not included in this repository.
