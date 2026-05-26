# VOID Chess

VOID is a small spatial chess model that operates directly on board tensors and moves, not text notation or rendered pixels.

![VOID architecture](void_architecture.svg)

## Architecture

The model encodes a chess board as an `8x8` spatial tensor and keeps that topology throughout the network. Its core block shifts information across the board in the center, orthogonal, and diagonal directions, then contracts those local views into a latent map.

After encoding, a shared recurrent "ghost" loop refines the latent board state for a configurable number of thinking steps. The same weights are reused at every step, so inference can trade compute for stronger decisions by increasing `n_think`.

The architecture includes:

- policy heads for `from_square` and `to_square`
- a value head for game outcome prediction
- an action-conditioned world-model head for next-board prediction
- a terminal-distance head for endgame conversion pressure
- optional legality and curriculum auxiliary losses

## Curriculum

Training is organized as a staged curriculum:

1. Tactics from Lichess puzzles
2. Foundation training on strong human games
3. Opening positions
4. Middlegame positions
5. Endgame positions
6. Optional reinforcement learning against Stockfish

Later phases mix in replay from earlier data so the model does not forget tactics and full-game behavior while specializing.

## Files

- `void.py` contains the model architecture.
- `train.py` contains the curriculum trainer, cache builders, held-out evaluation, and benchmark logic.
- `void_architecture.svg` is a compact architecture diagram.

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
