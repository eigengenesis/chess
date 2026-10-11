# Frozen Imagine Versus Direct: Stockfish Pool

Completed October 11, 2026. Local run: `benchmarks/stockfish_parallel_20261010_v2`.
This report summarizes the completed, fixed-sample evaluation; no model or
controller was tuned during the run.

## Results

Each mode played 200 games per Stockfish setting, including both colors from
the same 100 opening families. Score is `(wins + draws / 2) / games`.

| Mode | Stockfish UCI_Elo | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|---:|
| Direct | 1500 | 62 | 35 | 103 | 39.75% |
| Imagine | 1500 | 138 | 21 | 41 | 74.25% |
| Direct | 1800 | 16 | 40 | 144 | 18.00% |
| Imagine | 1800 | 53 | 44 | 103 | 37.50% |
| Direct | 2000 | 14 | 52 | 134 | 20.00% |
| Imagine | 2000 | 51 | 42 | 107 | 36.00% |
| Direct | 2200 | 3 | 32 | 165 | 9.50% |
| Imagine | 2200 | 29 | 58 | 113 | 29.00% |

Direct scored 174.5/800 (21.8125%); Imagine scored 353.5/800 (44.1875%).
The paired difference against the opponent pool was +22.375 percentage points,
with a 95% whole-opening-family bootstrap interval of [18.6234, 26.3750] points.
This is an opponent-pool score difference, not a Direct-versus-Imagine match.

All 1,600 games completed. There were no model exceptions, reference failures,
infrastructure failures, or unfinished games. Imagine lost one game on time
against SF-1500; it was retained. Results, legal game histories, completion
journals, and frozen source/checkpoint hashes were checked after completion.

## Frozen Protocol

- Model: 11,045,201-parameter Phase 4.5 checkpoint, approximately 42 MiB of
  inference weights. Both modes use the same weights and 16 recurrent steps.
  The repository's older `chess.pt` is Phase 4, not this evaluated checkpoint.
- Imagine: `frozen_runtime`, planner `legal_grounded_v2`, root width 6,
  opponent reply width 2, depth 3, transition depth 2, with the validated
  exact-outcome calibration and terminal/tactical/conversion safeguards.
- No MCTS. No Stockfish calls inside the model's action-selection controller.
  Calibrator fitting used offline Stockfish supervision.
- Experimental preference overrides, forcing-reply expansion, and P1/P2
  probabilistic controllers were disabled.
- Both game clocks: 120 seconds initially plus a one-second increment.
  Equal clocks do not imply equal neural computation.
- Reference: Stockfish 19 with `UCI_LimitStrength=true`, settings 1500, 1800,
  2000, and 2200; one engine thread, 32 MiB hash per worker, no pondering.
- Hardware: Apple M2; four simultaneous games; MPS model inference; one
  PyTorch CPU thread and one separate Stockfish process per game worker.
- Starts: one 12-ply seed from each of 100 independently grouped opening
  families, reused across all conditions with colors reversed. Registration
  seed: 20261010. Families were held out from known inspected experiments;
  absence from the model's pretraining corpus is not established.
- Fixed sample, no score adjudication, no outcome-based early stopping,
  no selective deletion of losses. Model errors/timeouts count as forfeits;
  reference failures or unfinished games block final rating estimates.
- Before this registration, four games in an earlier run were interrupted
  without durable results. That aborted run was preserved, its attempted
  opening family excluded, and none of its games pooled into these results.

Mean model decision latency across the four opponent conditions was
0.150-0.153 seconds for Direct and 1.373-1.417 seconds for Imagine. Imagine's
95th-percentile latency ranged from 1.965 to 2.064 seconds. These are measurements
under simultaneous local games, not Hugging Face CPU latency guarantees.

## Rating Interpretation

| Mode | Conditional fitted performance | 97.5% family-bootstrap interval |
|---|---:|---:|
| Direct | 1555.83 | 1515.67-1593.56 |
| Imagine | 1823.41 | 1787.75-1859.68 |

The fit assumes a 400-point logistic expected-score curve anchored to the
configured Stockfish UCI_Elo labels. Those labels are not independently measured
opponent ratings on this hardware. Maximum per-opponent score residuals were
18.22 percentage points for Direct and 18.73 for Imagine. The fit therefore
compresses an irregular score curve; its bootstrap intervals do not resolve
systematic opponent miscalibration or model misspecification.

The robust finding is higher Imagine score at every tested setting and a
positive paired pool-score interval. These results do not certify FIDE/online
Elo, establish 2000+ strength, or justify treating the difference of fitted
ratings as a measured head-to-head rating gap. Simultaneous-game resource
contention is part of this declared protocol; do not merge these results with
runs using different clocks, hardware, or concurrency.

The experiment evaluates the complete controller. It does not separate gains
from learned transitions, exact-state value evaluation, calibration, tactical
safeguards, or additional computation, and it is not evidence for the universal
Free Energy Principle. Future tuning needs development data and a new held-out
evaluation rather than reuse of this completed suite as unseen confirmation.

## Artifact Identity

Checkpoint SHA-256:
`b159fb1cb00e035d5c94ea9efca191a131f8f08c5352502a804f70b1a30cefc9`

Validated outcome calibration SHA-256:
`71036135e2fa2edf2b50d3b6727e28c58a0acf29e6df7a7e3e365e621e7f2114`

The local run retains `manifest.json`, `schedule.json`, `openings.json`,
`attempts.jsonl`, `games.jsonl`, `games.pgn`, per-game durable results, and
`summary.json`. This document publishes the aggregate results and protocol;
it does not redistribute the source opening corpus.
