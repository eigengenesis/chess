#!/usr/bin/env python3
"""
♟️🔥 CHESS GOD V2 — Curriculum Training to 2400 Elo
=====================================================
10M param model, 6-phase curriculum.

6-Phase Curriculum:
  Phase 0: Tactics         — Lichess puzzles (3.5M positions)
  Phase 1: Foundation      — Elo ≥ 1800 full games
  Phase 2: Openings        — First 15 moves of Elo ≥ 2000 games
  Phase 3: Middlegame      — Elo ≥ 2200 full games (moves 10-50)
  Phase 4: Endgames        — Elo ≥ 2000 games with ≤ 10 pieces
  Phase 5: Reinforcement   — Self-play vs Stockfish

Run on Colab (T4/A100) or locally:
  python train.py              # Resume curriculum if checkpoints exist
  python train.py --fresh      # Clean retrain from scratch
  python train.py --all-steps  # Ignore phase time limits
  python train.py --phase 2    # Resume at Phase 2
  python train.py --phase 0 --force-phase --all-steps
  python train.py --benchmark  # Benchmark only
"""

import os, sys, time, io, math, random, gc, json, hashlib, csv, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Auto-install
for pkg, imp in [("python-chess", "chess"), ("datasets", "datasets"), ("zstandard", "zstandard")]:
    try: __import__(imp)
    except ImportError: os.system(f"pip install -q {pkg}")

import chess
import chess.pgn
import zstandard as zstd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from void import ChessGod, board_to_tensor, move_to_indices, N_PLANES

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
PUZZLE_FILE = "lichess_puzzles.csv.zst"
MAX_PUZZLES = 10_000_000


class PuzzleDataset(Dataset):
    """Disk-backed Lichess puzzle dataset used for tactics and replay."""

    record_size = 100 + 4 + 200 + 4

    def __init__(self, bin_path, include_world_target=False):
        self.bin_path = bin_path
        self.include_world_target = include_world_target
        self.file_size = os.path.getsize(bin_path)
        self.length = self.file_size // self.record_size
        print(f"✅ Disk-backed Dataset: {self.length:,} samples found")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if getattr(self, "_file", None) is None:
            self._file = open(self.bin_path, "rb")

        self._file.seek(idx * self.record_size)
        data = self._file.read(self.record_size)

        fen = data[:100].decode("ascii").strip()
        move_idx = struct.unpack("i", data[100:104])[0]
        moves = data[104:304].decode("ascii").strip().split()
        rating = struct.unpack("i", data[304:308])[0]

        try:
            board = chess.Board(fen)
            for i in range(move_idx):
                board.push(chess.Move.from_uci(moves[i]))

            move = chess.Move.from_uci(moves[move_idx])
            tensor = board_to_tensor(board)
            from_sq, to_sq = move_to_indices(move, board)
            sample = (
                tensor,
                torch.tensor(from_sq, dtype=torch.long),
                torch.tensor(to_sq, dtype=torch.long),
                rating,
            )
            if self.include_world_target:
                next_board = board.copy(stack=False)
                next_board.push(move)
                sample = sample + (board_to_tensor(next_board),)
            return sample
        except Exception:
            return self.__getitem__((idx + 1) % self.length)


def download_puzzles():
    if os.path.exists(PUZZLE_FILE):
        return
    print("📥 Downloading Lichess puzzles (~300MB)...")
    ret = os.system(f"wget -q --show-progress '{PUZZLE_URL}' -O '{PUZZLE_FILE}'")
    if ret != 0 or not os.path.exists(PUZZLE_FILE):
        os.system(f"curl -L -o '{PUZZLE_FILE}' '{PUZZLE_URL}'")


def get_dataset(multi_move=False, include_world_target=False):
    """Create or open the compact binary puzzle cache."""
    download_puzzles()
    cache_file = "puzzles_cache.bin"

    if os.path.exists(cache_file) and os.path.getsize(cache_file) < 1000:
        os.remove(cache_file)

    if os.path.exists(cache_file):
        print(f"📂 Found binary cache: {cache_file}")
        return PuzzleDataset(cache_file, include_world_target=include_world_target)

    print(f"📖 Processing CSV (max {MAX_PUZZLES:,}) into binary cache...")
    dctx = zstd.ZstdDecompressor()
    count = 0
    with open(cache_file, "wb") as f_out:
        with open(PUZZLE_FILE, "rb") as f_in:
            reader = dctx.stream_reader(f_in)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            csv_reader = csv.reader(text_stream)
            next(csv_reader)

            for i, row in enumerate(csv_reader):
                if i >= MAX_PUZZLES:
                    break
                try:
                    fen, moves_str, rating = row[1], row[2], int(row[3])
                    moves = moves_str.split()
                    if len(moves) < 2:
                        continue

                    def write_rec(move_idx):
                        f_out.write(fen.encode("ascii")[:100].ljust(100))
                        f_out.write(struct.pack("i", move_idx))
                        f_out.write(moves_str.encode("ascii")[:200].ljust(200))
                        f_out.write(struct.pack("i", rating))

                    write_rec(1)
                    count += 1
                    if multi_move:
                        for move_idx in range(3, len(moves), 2):
                            write_rec(move_idx)
                            count += 1

                    if (i + 1) % 100_000 == 0:
                        print(f"  ... {i + 1:,} read, {count:,} samples stored")
                except Exception:
                    continue

    print(f"✅ Binary cache created: {count:,} samples stored")
    return PuzzleDataset(cache_file, include_world_target=include_world_target)

# ============================================================================
# DDP + DEVICE
# ============================================================================
def setup_ddp():
    """Initialize DDP if launched with torchrun. Returns local_rank or None."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank
    return None

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()

def is_main():
    """True if this is the main process (rank 0 or single-GPU)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

LOCAL_RANK = setup_ddp()

if LOCAL_RANK is not None:
    DEVICE = f'cuda:{LOCAL_RANK}'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

IS_CUDA = str(DEVICE).startswith('cuda')
if IS_CUDA:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ============================================================================
# V2 ARCHITECTURE CONFIG (~10M params, ~42MB)
# ============================================================================
D1 = 192
HIDDEN = 512
N_ENCODE = 5
N_THINK = 16
DROPOUT = 0.05
CURRICULUM_VERSION = "v2-adaptive-thinking-2026-05-21"

# ============================================================================
# PRETTY LOGGING (same system as Pokemon God)
# ============================================================================
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

def can_print():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

def banner(text, color=C.CYAN):
    if not can_print():
        return
    w = 64
    print(f"\n{color}{C.BOLD}{'═'*w}")
    print(f"  {text}")
    print(f"{'═'*w}{C.RESET}")

def section(text, icon="▸"):
    if not can_print():
        return
    print(f"\n{C.YELLOW}{C.BOLD}  {icon} {text}{C.RESET}")

def stat(label, value, color=C.WHITE):
    if not can_print():
        return
    print(f"    {C.DIM}{label:.<35s}{C.RESET} {color}{C.BOLD}{value}{C.RESET}")

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_v2")
os.makedirs(CKPT_DIR, exist_ok=True)

def ckpt_path(phase):
    return os.path.join(CKPT_DIR, f"chess_god_v2_p{phase}.pt")

def checkpoint_config():
    return {
        'd1': D1,
        'hidden': HIDDEN,
        'n_encode': N_ENCODE,
        'n_think': N_THINK,
        'n_planes': N_PLANES,
        'dropout': DROPOUT,
        'curriculum_version': CURRICULUM_VERSION,
        'ghost_gate': 'small_residual',
        'adaptive_thinking': True,
        'world_model_aux': True,
        'terminal_distance_aux': True,
        'inbuilt_policy_pressure': True,
        'legality_aux': True,
    }

def checkpoint_matches_model(ckpt):
    cfg = ckpt.get('config', {})
    return (
        cfg.get('d1') == D1 and
        cfg.get('hidden') == HIDDEN and
        cfg.get('n_encode') == N_ENCODE and
        cfg.get('n_planes', N_PLANES) == N_PLANES
    )

def model_state_dict(model):
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module.state_dict()
    return model.state_dict()

def cloned_model_state_dict(model):
    return {k: v.detach().clone() for k, v in model_state_dict(model).items()}

def base_model(model):
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model

def load_model_state(model, state_dict):
    model_obj = base_model(model)
    target_state = model_obj.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in target_state:
            compatible[key] = value
            continue
        if tuple(target_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    model_obj.load_state_dict(compatible, strict=False)
    if skipped and is_main():
        stat("Checkpoint tensors skipped", f"{len(skipped)} shape mismatches", C.YELLOW)

def save_checkpoint(model, optimizer, phase, step, metrics=None, lr=None, lr_schedule=None):
    if not is_main():
        return  # Only rank 0 saves
    path = ckpt_path(phase)
    payload = {
        'model': model_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'config': checkpoint_config(),
        'phase': phase,
        'step': step,
        'metrics': metrics or {},
    }
    if lr is not None:
        payload['lr'] = lr
    if lr_schedule is not None:
        payload['lr_schedule'] = lr_schedule
    torch.save(payload, path)
    # Also save as "latest"
    torch.save(payload, os.path.join(CKPT_DIR, "chess_god_v2_latest.pt"))
    stat("Checkpoint saved", path, C.GREEN)

def load_checkpoint(phase=None):
    """Load latest checkpoint, or specific phase."""
    if phase is not None:
        path = ckpt_path(phase)
    else:
        path = os.path.join(CKPT_DIR, "chess_god_v2_latest.pt")

    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    return ckpt

def destination_loss(to_logits, f_sq, t_sq):
    """Cross-entropy over the destination logits for each sample's true source."""
    batch_idx = torch.arange(f_sq.size(0), device=f_sq.device)
    return F.cross_entropy(to_logits[batch_idx, f_sq], t_sq)

def int_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace(" ", "").split(",")
    return [int(v) for v in value if str(v) != ""]

def thinking_depths(config):
    depths = sorted(set(d for d in int_list(config.get('think_depths')) if d > 0))
    return depths

def adaptive_thinking_enabled(config):
    return bool(config.get('adaptive_thinking', False) and thinking_depths(config))

def sample_think_depth(config):
    depths = thinking_depths(config)
    if not depths:
        return int(config.get('n_think', N_THINK))

    candidates = depths
    # Depth dropout: sometimes force an early-stop depth so intermediate states
    # become trainable policies, not just transient hidden states.
    dropout_p = float(config.get('depth_dropout', 0.0) or 0.0)
    if len(depths) > 1 and random.random() < dropout_p:
        candidates = depths[:-1]
    return int(random.choice(candidates))

def supervision_steps_for_depth(config, n_think):
    steps = [s for s in int_list(config.get('aux_steps')) if 0 < s <= n_think]
    if n_think not in steps:
        steps.append(n_think)
    return sorted(set(steps))

def scheduled_world_model_weight(config, step):
    """Gentle half-cosine ramp so transition learning grows after policy warms up."""
    final_w = float(config.get('world_model_weight', 0.0) or 0.0)
    if final_w <= 0:
        return 0.0
    start_w = float(config.get('world_model_start_weight', final_w * 0.25) or 0.0)
    start_w = min(start_w, final_w)
    max_steps = int(config.get('max_steps', 0) or 0)
    if max_steps <= 1:
        return final_w

    progress = min(1.0, max(0.0, int(step) / max(1, max_steps - 1)))
    cosine_up = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_w + (final_w - start_w) * cosine_up

def scheduled_distance_weight(config, step):
    final_w = float(config.get('distance_weight', 0.0) or 0.0)
    if final_w <= 0:
        return 0.0
    start_w = float(config.get('distance_start_weight', final_w) or 0.0)
    start_w = min(start_w, final_w)
    max_steps = int(config.get('max_steps', 0) or 0)
    if max_steps <= 1:
        return final_w
    progress = min(1.0, max(0.0, int(step) / max(1, max_steps - 1)))
    cosine_up = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_w + (final_w - start_w) * cosine_up

def scheduled_policy_pressure_weight(config, step):
    final_w = float(config.get('policy_pressure_weight', 0.0) or 0.0)
    if final_w <= 0:
        return 0.0
    start_w = float(config.get('policy_pressure_start_weight', final_w) or 0.0)
    start_w = min(start_w, final_w)
    max_steps = int(config.get('max_steps', 0) or 0)
    if max_steps <= 1:
        return final_w
    progress = min(1.0, max(0.0, int(step) / max(1, max_steps - 1)))
    cosine_up = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start_w + (final_w - start_w) * cosine_up

def phase_thinking_config_for_step(config, step):
    """Use broad variable-depth training, then a deep-only sharpening tail."""
    cfg = dict(config)
    sharpen_steps = int(cfg.get('sharpen_steps', 0) or 0)
    max_steps = int(cfg.get('max_steps', 0) or 0)
    if (
        adaptive_thinking_enabled(cfg)
        and sharpen_steps > 0
        and max_steps > 0
        and int(step) >= max(0, max_steps - sharpen_steps)
    ):
        final_depth = int(cfg.get('sharpen_n_think', cfg.get('n_think', N_THINK)))
        cfg['think_depths'] = [final_depth]
        cfg['n_think'] = final_depth
        cfg['aux_steps'] = int_list(cfg.get('sharpen_aux_steps', [8, final_depth]))
        cfg['aux_loss_weight'] = float(cfg.get('sharpen_aux_loss_weight', 0.12))
        cfg['consistency_anchor'] = int(cfg.get('sharpen_consistency_anchor', 8))
        cfg['consistency_weight'] = float(cfg.get('sharpen_consistency_weight', 0.01))
        cfg['value_consistency_weight'] = float(cfg.get('sharpen_value_consistency_weight', 0.01))
        cfg['confidence_reg_weight'] = float(cfg.get('sharpen_confidence_reg_weight', 0.01))
        cfg['depth_dropout'] = float(cfg.get('sharpen_depth_dropout', 0.0))
        cfg['sharpening'] = True
    else:
        cfg['sharpening'] = False
    cfg['world_model_weight'] = scheduled_world_model_weight(config, step)
    cfg['distance_weight'] = scheduled_distance_weight(config, step)
    cfg['policy_pressure_weight'] = scheduled_policy_pressure_weight(config, step)
    return cfg

def value_loss_for(value, vals=None, value_mask=None):
    value_pred = value.squeeze(-1)
    if vals is None:
        return value_pred.sum() * 0.0
    if value_mask is None:
        return F.mse_loss(value_pred, vals)
    value_loss_per = F.mse_loss(value_pred, vals, reduction='none')
    return (value_loss_per * value_mask).sum() / value_mask.sum().clamp_min(1.0)

def policy_value_loss(from_logits, to_logits, value, f_sq, t_sq,
                      vals=None, value_mask=None, value_weight=0.5,
                      distance_targets=None, policy_pressure_weight=0.0,
                      pressure_win_threshold=0.25,
                      pressure_survival_weight=0.5,
                      pressure_loss_downweight=0.25,
                      pressure_max_multiplier=1.5):
    batch_idx = torch.arange(f_sq.size(0), device=f_sq.device)
    loss_from_per = F.cross_entropy(from_logits, f_sq, reduction='none')
    loss_to_per = F.cross_entropy(to_logits[batch_idx, f_sq], t_sq, reduction='none')
    policy_per = loss_from_per + loss_to_per

    pressure_w = float(policy_pressure_weight or 0.0)
    if pressure_w > 0 and vals is not None and distance_targets is not None:
        vals_f = vals.to(device=policy_per.device, dtype=torch.float32).view(-1)
        dist_f = distance_targets.to(device=policy_per.device, dtype=torch.float32).view(-1)
        if vals_f.numel() == policy_per.numel() and dist_f.numel() == policy_per.numel():
            threshold = float(pressure_win_threshold)
            denom = max(1e-6, 1.0 - threshold)
            winning = torch.clamp((vals_f - threshold) / denom, 0.0, 1.0)
            losing = torch.clamp((-vals_f - threshold) / denom, 0.0, 1.0)
            close_to_terminal = torch.clamp(1.0 - dist_f, 0.0, 1.0)
            far_from_terminal = torch.clamp(dist_f, 0.0, 1.0)

            conversion = winning * close_to_terminal
            survival = losing * far_from_terminal * float(pressure_survival_weight)
            quick_loss = losing * close_to_terminal

            pressure = conversion + survival
            multiplier = 1.0 + pressure_w * pressure
            multiplier = multiplier * (1.0 - float(pressure_loss_downweight) * quick_loss)
            multiplier = torch.clamp(multiplier, min=0.65)
            multiplier = torch.clamp(multiplier, max=float(pressure_max_multiplier))
            loss_policy = (policy_per * multiplier).mean()
        else:
            loss_policy = policy_per.mean()
    else:
        loss_policy = policy_per.mean()

    loss_value = value_loss_for(value, vals, value_mask)
    return loss_policy + value_weight * loss_value, loss_policy, loss_value

def policy_kl_consistency(anchor, current, f_sq):
    anchor_from, anchor_to, _anchor_value = anchor
    current_from, current_to, _current_value = current
    batch_idx = torch.arange(f_sq.size(0), device=f_sq.device)

    from_kl = F.kl_div(
        F.log_softmax(current_from, dim=-1),
        F.softmax(anchor_from.detach(), dim=-1),
        reduction='batchmean',
    )
    to_kl = F.kl_div(
        F.log_softmax(current_to[batch_idx, f_sq], dim=-1),
        F.softmax(anchor_to[batch_idx, f_sq].detach(), dim=-1),
        reduction='batchmean',
    )
    return from_kl + to_kl

def target_move_probability(outputs, f_sq, t_sq):
    from_logits, to_logits, _value = outputs
    batch_idx = torch.arange(f_sq.size(0), device=f_sq.device)
    from_prob = F.softmax(from_logits, dim=-1)[batch_idx, f_sq]
    to_prob = F.softmax(to_logits[batch_idx, f_sq], dim=-1)[batch_idx, t_sq]
    return from_prob * to_prob

def approximate_next_board_targets(boards, f_sq, t_sq):
    """Fallback next-board tensor when exact python-chess targets are absent.

    This keeps old caches usable, but exact world-model training uses
    board_to_tensor(board_after_move) supplied by the streamer/dataset/cache.
    """
    pieces = (boards[:, :12] > 0.5).float()
    cur = pieces[:, :6].clone()
    opp = pieces[:, 6:12].clone()

    batch_idx = torch.arange(boards.size(0), device=boards.device)
    fr, fc = f_sq // 8, f_sq % 8
    tr, tc = t_sq // 8, t_sq % 8

    moving = cur[batch_idx, :, fr, fc]
    cur[batch_idx, :, fr, fc] = 0.0
    opp[batch_idx, :, tr, tc] = 0.0
    cur[batch_idx, :, tr, tc] = moving

    # Side-to-move flips after the move. Since tensors are always oriented from
    # the side-to-move perspective, mirror ranks and swap own/opponent planes.
    next_pieces = torch.cat([
        torch.flip(opp, dims=[2]),
        torch.flip(cur, dims=[2]),
    ], dim=1)

    targets = torch.zeros_like(boards)
    targets[:, :12] = next_pieces
    if boards.size(1) > 12:
        targets[:, 12] = 1.0 - boards[:, 12]
    if boards.size(1) > 13:
        targets[:, 13] = torch.flip(boards[:, 13], dims=[1])
    return targets

def world_model_loss_weights(targets, f_sq, t_sq, config):
    """Favor active and changed cells so empty-board accuracy cannot dominate."""
    empty_w = float(config.get('world_empty_weight', 0.15))
    active_w = float(config.get('world_piece_weight', 1.0))
    changed_w = float(config.get('world_changed_weight', 3.0))

    weights = torch.full(targets.shape, empty_w, dtype=torch.float32, device=targets.device)
    weights = torch.where(targets > 0.5, torch.full_like(weights, active_w), weights)

    batch_idx = torch.arange(targets.size(0), device=targets.device)
    fr, fc = f_sq // 8, f_sq % 8
    tr, tc = t_sq // 8, t_sq % 8
    weights[batch_idx, :, 7 - fr, fc] = changed_w
    weights[batch_idx, :, 7 - tr, tc] = changed_w
    return weights

def world_model_training_loss(model, boards, f_sq, t_sq, config, world_targets=None):
    weight = float(config.get('world_model_weight', 0.0) or 0.0)
    if weight <= 0:
        zero = boards.sum() * 0.0
        return zero, zero

    max_world_batch = int(config.get('world_model_max_batch', 0) or 0)
    if max_world_batch > 0 and boards.size(0) > max_world_batch:
        idx = torch.randperm(boards.size(0), device=boards.device)[:max_world_batch]
        boards = boards.index_select(0, idx)
        f_sq = f_sq.index_select(0, idx)
        t_sq = t_sq.index_select(0, idx)
        if world_targets is not None:
            world_targets = world_targets.index_select(0, idx)

    n_world = int(config.get('world_n_think', min(4, config.get('n_think', N_THINK))))
    transition_logits = model(
        boards,
        n_think=n_world,
        transition_from=f_sq,
        transition_to=t_sq,
    )
    if world_targets is None:
        targets = approximate_next_board_targets(boards, f_sq, t_sq)
    else:
        targets = world_targets.to(device=boards.device, dtype=torch.float32)
    weights = world_model_loss_weights(targets, f_sq, t_sq, config)
    loss_map = F.binary_cross_entropy_with_logits(
        transition_logits.float(),
        targets.float(),
        reduction='none',
    )
    raw_loss = (loss_map * weights).sum() / weights.sum().clamp_min(1.0)
    return weight * raw_loss, raw_loss

def distance_training_loss(model, boards, config, distance_targets=None, distance_mask=None):
    weight = float(config.get('distance_weight', 0.0) or 0.0)
    if weight <= 0 or distance_targets is None:
        zero = boards.sum() * 0.0
        return zero, zero

    targets = distance_targets.to(device=boards.device, dtype=torch.float32).view(-1)
    if distance_mask is None:
        mask = torch.ones_like(targets)
    else:
        mask = distance_mask.to(device=boards.device, dtype=torch.float32).view(-1)
    if mask.sum() <= 0:
        zero = boards.sum() * 0.0
        return zero, zero

    max_distance_batch = int(config.get('distance_model_max_batch', 0) or 0)
    if max_distance_batch > 0 and boards.size(0) > max_distance_batch:
        valid_idx = torch.nonzero(mask > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 0:
            zero = boards.sum() * 0.0
            return zero, zero
        perm = torch.randperm(valid_idx.numel(), device=boards.device)[:max_distance_batch]
        idx = valid_idx.index_select(0, perm)
        boards = boards.index_select(0, idx)
        targets = targets.index_select(0, idx)
        mask = mask.index_select(0, idx)

    n_distance = int(config.get('distance_n_think', config.get('n_think', N_THINK)))
    logits = model(boards, n_think=n_distance, distance_only=True).squeeze(-1)
    pred = torch.sigmoid(logits.float())
    raw_per = F.smooth_l1_loss(pred, targets.float(), reduction='none')
    raw_loss = (raw_per * mask).sum() / mask.sum().clamp_min(1.0)
    return weight * raw_loss, raw_loss

def legality_training_loss(model, boards, config, legal_targets=None, legal_mask=None):
    weight = float(config.get('legality_weight', 0.0) or 0.0)
    if weight <= 0 or legal_targets is None:
        zero = boards.sum() * 0.0
        return zero, zero

    targets = legal_targets.to(device=boards.device, dtype=torch.float32)
    if legal_mask is None:
        mask = torch.ones(targets.size(0), device=boards.device, dtype=torch.float32)
    else:
        mask = legal_mask.to(device=boards.device, dtype=torch.float32).view(-1)
    if mask.sum() <= 0:
        zero = boards.sum() * 0.0
        return zero, zero

    max_legal_batch = int(config.get('legality_model_max_batch', 0) or 0)
    if max_legal_batch > 0 and boards.size(0) > max_legal_batch:
        valid_idx = torch.nonzero(mask > 0, as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 0:
            zero = boards.sum() * 0.0
            return zero, zero
        perm = torch.randperm(valid_idx.numel(), device=boards.device)[:max_legal_batch]
        idx = valid_idx.index_select(0, perm)
        boards = boards.index_select(0, idx)
        targets = targets.index_select(0, idx)
        mask = mask.index_select(0, idx)

    n_legal = int(config.get('legality_n_think', config.get('n_think', N_THINK)))
    logits = model(boards, n_think=n_legal, legality_only=True).float()
    pos_weight = torch.tensor(float(config.get('legality_pos_weight', 16.0)), device=boards.device)
    loss_map = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        pos_weight=pos_weight,
        reduction='none',
    ).mean(dim=(1, 2))
    raw_loss = (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
    return weight * raw_loss, raw_loss

def adaptive_training_loss(model, boards, f_sq, t_sq, config,
                           vals=None, value_mask=None, value_weight=0.5,
                           world_targets=None, distance_targets=None, distance_mask=None,
                           legal_targets=None, legal_mask=None):
    """Policy/value loss with optional variable-depth recurrent supervision."""
    pressure_kwargs = {
        'distance_targets': distance_targets,
        'policy_pressure_weight': float(config.get('policy_pressure_weight', 0.0) or 0.0),
        'pressure_win_threshold': float(config.get('policy_pressure_win_threshold', 0.25)),
        'pressure_survival_weight': float(config.get('policy_pressure_survival_weight', 0.5)),
        'pressure_loss_downweight': float(config.get('policy_pressure_loss_downweight', 0.25)),
        'pressure_max_multiplier': float(config.get('policy_pressure_max_multiplier', 1.5)),
    }
    if not adaptive_thinking_enabled(config):
        n_think = int(config.get('n_think', N_THINK))
        from_logits, to_logits, value = model(boards, n_think=n_think)
        loss, loss_policy, loss_value = policy_value_loss(
            from_logits, to_logits, value, f_sq, t_sq,
            vals=vals, value_mask=value_mask, value_weight=value_weight,
            **pressure_kwargs,
        )
        world_weighted, world_raw = world_model_training_loss(
            model, boards, f_sq, t_sq, config, world_targets=world_targets,
        )
        distance_weighted, distance_raw = distance_training_loss(
            model, boards, config,
            distance_targets=distance_targets,
            distance_mask=distance_mask,
        )
        legality_weighted, legality_raw = legality_training_loss(
            model, boards, config,
            legal_targets=legal_targets,
            legal_mask=legal_mask,
        )
        loss = loss + world_weighted + distance_weighted + legality_weighted
        zero = loss.detach() * 0.0
        return {
            'loss': loss,
            'policy': loss_policy,
            'value': loss_value,
            'aux': zero,
            'reg': zero,
            'world': world_raw.detach(),
            'distance': distance_raw.detach(),
            'legality': legality_raw.detach(),
            'from_logits': from_logits,
            'to_logits': to_logits,
            'value_pred': value,
            'n_think': n_think,
        }

    n_think = sample_think_depth(config)
    supervision_steps = supervision_steps_for_depth(config, n_think)
    from_logits, to_logits, value, step_outputs = model(
        boards,
        n_think=n_think,
        return_steps=True,
        supervision_steps=supervision_steps,
    )

    final_outputs = (from_logits, to_logits, value)
    loss, loss_policy, loss_value = policy_value_loss(
        from_logits, to_logits, value, f_sq, t_sq,
        vals=vals, value_mask=value_mask, value_weight=value_weight,
        **pressure_kwargs,
    )

    aux_terms = []
    aux_weights = []
    for step in supervision_steps:
        if step == n_think or step not in step_outputs:
            continue
        aux_loss, _aux_policy, _aux_value = policy_value_loss(
            *step_outputs[step], f_sq, t_sq,
            vals=vals, value_mask=value_mask, value_weight=value_weight,
            **pressure_kwargs,
        )
        aux_terms.append(aux_loss)
        aux_weights.append((step / max(1, n_think)) ** 1.5)

    aux_loss = loss.detach() * 0.0
    if aux_terms:
        total_weight = sum(aux_weights)
        aux_loss = sum((w / total_weight) * term for w, term in zip(aux_weights, aux_terms))
        loss = loss + float(config.get('aux_loss_weight', 0.35)) * aux_loss

    reg_loss = loss.detach() * 0.0
    anchor_target = min(int(config.get('consistency_anchor', 8)), n_think)
    anchor_candidates = [s for s in supervision_steps if s <= anchor_target and s in step_outputs]
    anchor_step = max(anchor_candidates) if anchor_candidates else min(step_outputs.keys())
    anchor_outputs = step_outputs[anchor_step]
    if anchor_step != n_think:
        consistency_w = float(config.get('consistency_weight', 0.0) or 0.0)
        value_consistency_w = float(config.get('value_consistency_weight', 0.0) or 0.0)
        confidence_w = float(config.get('confidence_reg_weight', 0.0) or 0.0)

        if consistency_w:
            kl = policy_kl_consistency(anchor_outputs, final_outputs, f_sq)
            reg_loss = reg_loss + consistency_w * kl
        if value_consistency_w:
            reg_loss = reg_loss + value_consistency_w * F.mse_loss(
                value.squeeze(-1),
                anchor_outputs[2].detach().squeeze(-1),
            )
        if confidence_w:
            anchor_conf = target_move_probability(anchor_outputs, f_sq, t_sq).detach()
            final_conf = target_move_probability(final_outputs, f_sq, t_sq)
            reg_loss = reg_loss + confidence_w * F.relu(anchor_conf - final_conf).mean()

        loss = loss + reg_loss

    world_weighted, world_raw = world_model_training_loss(
        model, boards, f_sq, t_sq, config, world_targets=world_targets,
    )
    distance_weighted, distance_raw = distance_training_loss(
        model, boards, config,
        distance_targets=distance_targets,
        distance_mask=distance_mask,
    )
    legality_weighted, legality_raw = legality_training_loss(
        model, boards, config,
        legal_targets=legal_targets,
        legal_mask=legal_mask,
    )
    loss = loss + world_weighted + distance_weighted + legality_weighted

    return {
        'loss': loss,
        'policy': loss_policy,
        'value': loss_value,
        'aux': aux_loss,
        'reg': reg_loss,
        'world': world_raw.detach(),
        'distance': distance_raw.detach(),
        'legality': legality_raw.detach(),
        'from_logits': from_logits,
        'to_logits': to_logits,
        'value_pred': value,
        'n_think': n_think,
    }

def lr_schedule_config(config):
    base_lr = float(config['lr'])
    max_steps = int(config['max_steps'])
    warmup_steps = int(config.get('warmup_steps', min(1000, max(1, max_steps // 40))))
    warmup_steps = max(0, min(warmup_steps, max_steps - 1))
    min_lr = float(config.get('min_lr', base_lr * 0.02))
    warmup_start_lr = float(config.get('warmup_start_lr', max(min_lr, base_lr * 0.10)))
    return {
        'name': 'warmup_cosine_decay',
        'base_lr': base_lr,
        'min_lr': min_lr,
        'warmup_start_lr': warmup_start_lr,
        'warmup_steps': warmup_steps,
        'max_steps': max_steps,
    }

def scheduled_lr(config, step):
    """Deterministic warmup + one-way cosine decay; safe across checkpoint resumes."""
    sched = lr_schedule_config(config)
    step = max(0, min(int(step), sched['max_steps'] - 1))

    if sched['warmup_steps'] > 0 and step < sched['warmup_steps']:
        pct = (step + 1) / sched['warmup_steps']
        return sched['warmup_start_lr'] + pct * (sched['base_lr'] - sched['warmup_start_lr'])

    decay_steps = max(1, sched['max_steps'] - sched['warmup_steps'])
    progress = min(1.0, max(0.0, (step - sched['warmup_steps']) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return sched['min_lr'] + (sched['base_lr'] - sched['min_lr']) * cosine

def set_optimizer_lr(optimizer, lr, base_lr=None):
    for group in optimizer.param_groups:
        group['lr'] = lr
        if base_lr is not None:
            group['initial_lr'] = base_lr
    return lr

def lr_schedule_summary(config):
    sched = lr_schedule_config(config)
    return (
        f"warmup {sched['warmup_steps']} steps "
        f"{sched['warmup_start_lr']:.1e}->{sched['base_lr']:.1e}, "
        f"cosine -> {sched['min_lr']:.1e}"
    )

def apply_runtime_options(all_steps=False):
    if all_steps:
        for cfg in PHASES.values():
            cfg['max_minutes'] = float('inf')

def apply_batch_overrides(phase0_batch_size=None, gm_batch_size=None, phase12_batch_size=None):
    if phase0_batch_size is not None:
        PHASES[0]['batch_size'] = phase0_batch_size
    if gm_batch_size is not None:
        for phase in (1, 2, 3, 4):
            PHASES[phase]['batch_size'] = gm_batch_size
    if phase12_batch_size is not None:
        for phase in (1, 2):
            PHASES[phase]['batch_size'] = phase12_batch_size

def apply_loader_overrides(num_workers=None):
    if num_workers is not None:
        for cfg in PHASES.values():
            cfg['num_workers'] = max(0, int(num_workers))

def apply_thinking_overrides(disable_adaptive=False, think_depths=None):
    depths = thinking_depths({'think_depths': think_depths}) if think_depths else None
    for cfg in PHASES.values():
        if disable_adaptive:
            cfg['adaptive_thinking'] = False
        if depths and cfg.get('adaptive_thinking', False):
            cfg['think_depths'] = depths
            cfg['n_think'] = max(depths)
            cfg['aux_steps'] = [s for s in int_list(cfg.get('aux_steps')) if s <= max(depths)]
            if not cfg['aux_steps']:
                cfg['aux_steps'] = depths

def apply_thinking_loss_overrides(aux_loss_weight=None, consistency_weight=None,
                                  value_consistency_weight=None,
                                  confidence_reg_weight=None, depth_dropout=None,
                                  world_model_weight=None, world_model_start_weight=None,
                                  world_model_max_batch=None, world_n_think=None,
                                  distance_weight=None, distance_start_weight=None,
                                  distance_model_max_batch=None, distance_n_think=None,
                                  policy_pressure_weight=None, policy_pressure_start_weight=None,
                                  legality_weight=None, legality_model_max_batch=None,
                                  legality_n_think=None):
    overrides = {
        'aux_loss_weight': aux_loss_weight,
        'consistency_weight': consistency_weight,
        'value_consistency_weight': value_consistency_weight,
        'confidence_reg_weight': confidence_reg_weight,
        'depth_dropout': depth_dropout,
        'world_model_weight': world_model_weight,
        'world_model_start_weight': world_model_start_weight,
        'world_model_max_batch': world_model_max_batch,
        'world_n_think': world_n_think,
        'distance_weight': distance_weight,
        'distance_start_weight': distance_start_weight,
        'distance_model_max_batch': distance_model_max_batch,
        'distance_n_think': distance_n_think,
        'policy_pressure_weight': policy_pressure_weight,
        'policy_pressure_start_weight': policy_pressure_start_weight,
        'legality_weight': legality_weight,
        'legality_model_max_batch': legality_model_max_batch,
        'legality_n_think': legality_n_think,
    }
    for cfg in PHASES.values():
        if not cfg.get('adaptive_thinking', False):
            continue
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = int(value) if key in (
                    'world_n_think',
                    'world_model_max_batch',
                    'distance_n_think',
                    'distance_model_max_batch',
                    'legality_n_think',
                    'legality_model_max_batch',
                ) else float(value)

def apply_phase_max_steps(start_phase, stop_after_phase=None, phase_max_steps=None):
    if phase_max_steps is None:
        return
    end_phase = 5 if stop_after_phase is None else max(0, min(5, int(stop_after_phase)))
    for phase in range(max(0, int(start_phase)), end_phase + 1):
        if phase in PHASES:
            PHASES[phase]['max_steps'] = int(phase_max_steps)

def replay_batch_counts(total_batch_size, replay_cfg):
    puzzle_frac = float(replay_cfg.get('puzzle', 0.0) or 0.0)
    gm_frac = float(replay_cfg.get('gm', 0.0) or 0.0)
    low_elo_frac = float(replay_cfg.get('low_elo', 0.0) or 0.0)
    puzzle_n = int(round(total_batch_size * puzzle_frac))
    gm_n = int(round(total_batch_size * gm_frac))
    low_elo_n = int(round(total_batch_size * low_elo_frac))
    main_n = total_batch_size - puzzle_n - gm_n - low_elo_n
    if main_n < 1:
        main_n = 1
        overflow = puzzle_n + gm_n + low_elo_n + main_n - total_batch_size
        while overflow > 0 and gm_n > 0:
            gm_n -= 1
            overflow -= 1
        while overflow > 0 and low_elo_n > 0:
            low_elo_n -= 1
            overflow -= 1
        while overflow > 0 and puzzle_n > 0:
            puzzle_n -= 1
            overflow -= 1
    return {
        'main': main_n,
        'puzzle': max(0, puzzle_n),
        'gm': max(0, gm_n),
        'low_elo': max(0, low_elo_n),
    }

def replay_mix_summary(total_batch_size, replay_cfg):
    counts = replay_batch_counts(total_batch_size, replay_cfg)
    if counts['puzzle'] == 0 and counts['gm'] == 0 and counts['low_elo'] == 0:
        return "disabled"
    parts = [f"main {counts['main']}"]
    if counts['puzzle']:
        parts.append(f"puzzle {counts['puzzle']}")
    if counts['gm']:
        parts.append(f"full-game {counts['gm']}")
    if counts['low_elo']:
        parts.append(f"low-elo {counts['low_elo']}")
    return ", ".join(parts)

class SliceDataset(Dataset):
    def __init__(self, dataset, start, length):
        self.dataset = dataset
        self.start = int(start)
        self.length = int(length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.dataset[self.start + idx]

def puzzle_split_sizes(dataset_len, cfg):
    requested = int(cfg.get('val_size', 50_000))
    max_reasonable = max(1, dataset_len // 20)
    val_size = min(requested, max_reasonable)
    train_size = max(1, dataset_len - val_size)
    return train_size, val_size

def get_puzzle_train_val(cfg):
    dataset = get_dataset(
        multi_move=False,
        include_world_target=float(cfg.get('world_model_weight', 0.0) or 0.0) > 0,
    )
    train_size, val_size = puzzle_split_sizes(len(dataset), cfg)
    train_dataset = SliceDataset(dataset, 0, train_size)
    val_dataset = SliceDataset(dataset, train_size, val_size)
    return train_dataset, val_dataset, len(dataset)

# ============================================================================
# GM GAME STREAMER
# ============================================================================
HF_DATASET = "Lichess/standard-chess-games"
GM_CACHE_DIR = os.path.join(SCRIPT_DIR, "gm_cache_v2")
REPLAY_GM_CACHE_DIR = os.path.join(SCRIPT_DIR, "gm_replay_cache_v2")
GM_CACHE_BOARD_DTYPE = np.uint8 if N_PLANES <= 17 else np.float16
GM_CACHE_SHARD_SAMPLES = max(100_000, int(os.getenv("GM_CACHE_SHARD_SAMPLES", "1000000")))
WORLD_BINARY_PLANES = min(N_PLANES, 17)
WORLD_PACKED_BITS = WORLD_BINARY_PLANES * 64
WORLD_PACKED_BYTES = (WORLD_PACKED_BITS + 7) // 8
DISTANCE_TARGET_FORMAT = 'terminal_distance_log_v1'
DISTANCE_TARGET_MAX_PLIES = 200
LEGALITY_TARGET_FORMAT = 'legal_moves_packed_v1'
LEGALITY_PACKED_BITS = 64 * 64
LEGALITY_PACKED_BYTES = (LEGALITY_PACKED_BITS + 7) // 8

def pack_world_targets_np(targets):
    binary = np.rint(np.clip(targets[:, :WORLD_BINARY_PLANES], 0, 1)).astype(np.uint8)
    packed = np.packbits(binary.reshape(binary.shape[0], -1), axis=1)
    halfmove = None
    if N_PLANES > 17:
        halfmove = np.rint(np.clip(targets[:, 17, 0, 0], 0, 1) * 100).astype(np.uint8)
    return packed, halfmove

def unpack_world_target_np(packed, halfmove=None):
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[:WORLD_PACKED_BITS]
    target = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    target[:WORLD_BINARY_PLANES] = bits.reshape(WORLD_BINARY_PLANES, 8, 8).astype(np.float32)
    if N_PLANES > 17 and halfmove is not None:
        target[17] = min(float(halfmove) / 100.0, 1.0)
    return target

def legal_move_target(board):
    target = np.zeros((64, 64), dtype=np.uint8)
    for move in board.legal_moves:
        f_idx, t_idx = move_to_indices(move, board)
        target[f_idx, t_idx] = 1
    return target

def pack_legal_targets_np(targets):
    legal = np.rint(np.clip(targets, 0, 1)).astype(np.uint8)
    return np.packbits(legal.reshape(legal.shape[0], -1), axis=1)

def unpack_legal_target_np(packed):
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[:LEGALITY_PACKED_BITS]
    return bits.reshape(64, 64).astype(np.float32)

def terminal_distance_target(plies_remaining, max_plies=DISTANCE_TARGET_MAX_PLIES):
    """Map plies remaining to [0, 1], with log scaling for long games."""
    plies = max(1, int(plies_remaining))
    return min(1.0, math.log1p(plies) / math.log1p(max_plies))

class GMStreamer:
    """Streams full games from HuggingFace Lichess dataset with Elo filtering."""

    def __init__(self, min_elo=1800, max_elo=None, move_range=None,
                 min_pieces=None, max_pieces=None, skip_games=0,
                 include_legality=False):
        """
        Args:
            min_elo: Minimum Elo for both players
            max_elo: Maximum Elo for both players
            move_range: (start, end) tuple — only sample moves in this range
                        e.g., (0, 15) for openings, (10, 50) for middlegame
            min_pieces: Only include positions with >= this many pieces
            max_pieces: Only include positions with <= this many pieces (endgame)
        """
        self.min_elo = int(min_elo or 0)
        self.max_elo = int(max_elo) if max_elo is not None else None
        self.move_range = move_range
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces
        self.games_seen = int(skip_games or 0)
        self.include_legality = bool(include_legality)

        elo_label = f"{self.min_elo}-{self.max_elo}" if self.max_elo is not None else f"≥ {self.min_elo}"
        section(f"Connecting to Lichess stream (Elo {elo_label})", "📡")
        self.dataset = load_dataset(HF_DATASET, name="default", streaming=True, split="train")
        if skip_games:
            stat("Skipping source games", f"{int(skip_games):,}", C.YELLOW)
            self.dataset = self.dataset.skip(int(skip_games))
        self.iterator = iter(self.dataset)
        self._reset_game()

        filters = []
        if self.max_elo is not None: filters.append(f"Elo {self.min_elo}-{self.max_elo}")
        if move_range: filters.append(f"moves {move_range[0]}-{move_range[1]}")
        if max_pieces: filters.append(f"≤{max_pieces} pieces")
        if min_pieces: filters.append(f"≥{min_pieces} pieces")
        if filters:
            stat("Filters", ", ".join(filters))

    def _reset_game(self):
        self.board = chess.Board()
        self.moves = []
        self.result = 0.0
        self.move_idx = 0

    def _load_next_game(self):
        """Pull next valid game matching Elo filter."""
        while True:
            try:
                ex = next(self.iterator)
                self.games_seen += 1
                try:
                    w_elo = int(ex.get('WhiteElo', 0) or 0)
                    b_elo = int(ex.get('BlackElo', 0) or 0)
                except ValueError:
                    continue

                if w_elo < self.min_elo or b_elo < self.min_elo:
                    continue
                if self.max_elo is not None and (w_elo > self.max_elo or b_elo > self.max_elo):
                    continue

                movetext = ex.get('movetext', '')
                if not movetext:
                    continue

                game = chess.pgn.read_game(io.StringIO(movetext))
                if game is None:
                    continue

                moves = list(game.mainline_moves())
                if len(moves) < 6:
                    continue

                # Parse result
                res = ex.get('Result', '*')
                if res == '1-0': result = 1.0
                elif res == '0-1': result = -1.0
                else: result = 0.0

                self.board = game.board()
                self.moves = moves
                self.result = result
                self.move_idx = 0
                return True

            except StopIteration:
                return False
            except Exception:
                continue

    def get_batch(self, batch_size=512):
        """Get a batch with exact next-board world-model targets."""
        tensors, f_sqs, t_sqs, values = [], [], [], []
        world_targets, distance_targets, legal_targets = [], [], []

        while len(tensors) < batch_size:
            # Need a new game?
            if self.move_idx >= len(self.moves) - 1:
                if not self._load_next_game():
                    break
                continue

            move_num = self.move_idx // 2  # Full move number
            board = self.board

            # Apply move range filter
            if self.move_range:
                lo, hi = self.move_range
                if move_num < lo:
                    self.board.push(self.moves[self.move_idx])
                    self.move_idx += 1
                    continue
                if move_num > hi:
                    # Skip to next game
                    self._reset_game()
                    continue

            # Apply piece count filter
            n_pieces = len(board.piece_map())
            if self.max_pieces and n_pieces > self.max_pieces:
                self.board.push(self.moves[self.move_idx])
                self.move_idx += 1
                continue
            if self.min_pieces and n_pieces < self.min_pieces:
                self.board.push(self.moves[self.move_idx])
                self.move_idx += 1
                continue

            # Extract training sample
            move = self.moves[self.move_idx]
            try:
                f_idx, t_idx = move_to_indices(move, board)
                tensor = board_to_tensor(board)
                next_board = board.copy(stack=False)
                next_board.push(move)
                next_tensor = board_to_tensor(next_board)
            except Exception:
                self.board.push(move)
                self.move_idx += 1
                continue

            tensors.append(tensor)
            f_sqs.append(torch.tensor(f_idx, dtype=torch.long))
            t_sqs.append(torch.tensor(t_idx, dtype=torch.long))
            world_targets.append(next_tensor)
            if self.include_legality:
                legal_targets.append(torch.from_numpy(legal_move_target(board)).float())

            # Value from current player's perspective
            val = self.result if board.turn == chess.WHITE else -self.result
            values.append(torch.tensor(val, dtype=torch.float32))
            distance_targets.append(torch.tensor(
                terminal_distance_target(len(self.moves) - self.move_idx),
                dtype=torch.float32,
            ))

            self.board.push(move)
            self.move_idx += 1

        if not tensors:
            return None

        out = {
            'boards': torch.stack(tensors),
            'from_sq': torch.stack(f_sqs),
            'to_sq': torch.stack(t_sqs),
            'values': torch.stack(values),
            'value_mask': torch.ones(len(tensors), dtype=torch.float32),
            'world_targets': torch.stack(world_targets),
            'distance_targets': torch.stack(distance_targets),
            'distance_mask': torch.ones(len(tensors), dtype=torch.float32),
        }
        if self.include_legality:
            out['legal_targets'] = torch.stack(legal_targets)
            out['legal_mask'] = torch.ones(len(tensors), dtype=torch.float32)
        return out

def gm_cache_key(phase, cfg, max_samples):
    payload = {
        'phase': phase,
        'samples': int(max_samples),
        'n_planes': N_PLANES,
        'min_elo': cfg.get('min_elo'),
        'move_range': cfg.get('move_range'),
        'min_pieces': cfg.get('min_pieces'),
        'max_pieces': cfg.get('max_pieces'),
        'curriculum_version': CURRICULUM_VERSION,
    }
    if float(cfg.get('distance_weight', 0.0) or 0.0) > 0:
        payload['distance_target_format'] = DISTANCE_TARGET_FORMAT
    if float(cfg.get('legality_weight', 0.0) or 0.0) > 0:
        payload['legality_target_format'] = LEGALITY_TARGET_FORMAT
    if cfg.get('max_elo') is not None:
        payload['max_elo'] = cfg.get('max_elo')
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
    return f"phase{phase}_{int(max_samples)}_{digest}"

def gm_cache_paths(phase, cfg, max_samples, cache_dir=None):
    cache_dir = cache_dir or GM_CACHE_DIR
    key = gm_cache_key(phase, cfg, max_samples)
    return {
        'dir': cache_dir,
        'key': key,
        'meta': os.path.join(cache_dir, f"{key}.meta.json"),
        'shard_dir': os.path.join(cache_dir, f"{key}.shards"),
        'boards': os.path.join(cache_dir, f"{key}.boards.npy"),
        'from_sq': os.path.join(cache_dir, f"{key}.from.npy"),
        'to_sq': os.path.join(cache_dir, f"{key}.to.npy"),
        'values': os.path.join(cache_dir, f"{key}.values.npy"),
        'distance': os.path.join(cache_dir, f"{key}.distance.npy"),
        'legal_bits': os.path.join(cache_dir, f"{key}.legal_bits.npy"),
        'world_bits': os.path.join(cache_dir, f"{key}.world_bits.npy"),
        'world_halfmove': os.path.join(cache_dir, f"{key}.world_halfmove.npy"),
    }

def gm_cache_shard_paths(paths, shard_idx):
    prefix = os.path.join(paths['shard_dir'], f"shard_{int(shard_idx):05d}")
    return {
        'meta': f"{prefix}.meta.json",
        'boards': f"{prefix}.boards.npy",
        'from_sq': f"{prefix}.from.npy",
        'to_sq': f"{prefix}.to.npy",
        'values': f"{prefix}.values.npy",
        'distance': f"{prefix}.distance.npy",
        'legal_bits': f"{prefix}.legal_bits.npy",
        'world_bits': f"{prefix}.world_bits.npy",
        'world_halfmove': f"{prefix}.world_halfmove.npy",
    }

def gm_cache_required_arrays(cfg=None):
    required = ['boards', 'from_sq', 'to_sq', 'values', 'world_bits']
    if N_PLANES > 17:
        required.append('world_halfmove')
    needs_distance = bool(cfg and float(cfg.get('distance_weight', 0.0) or 0.0) > 0)
    if needs_distance:
        required.append('distance')
    needs_legality = bool(cfg and float(cfg.get('legality_weight', 0.0) or 0.0) > 0)
    if needs_legality:
        required.append('legal_bits')
    return required, needs_distance, needs_legality

def gm_cache_shard_ready(shard_paths, expected_samples, cfg=None):
    required, needs_distance, needs_legality = gm_cache_required_arrays(cfg)
    if not os.path.exists(shard_paths['meta']):
        return False
    if not all(os.path.exists(shard_paths[k]) for k in required):
        return False
    try:
        with open(shard_paths['meta'], 'r') as f:
            meta = json.load(f)
        ready = (
            int(meta.get('samples', 0)) == int(expected_samples)
            and int(meta.get('n_planes', -1)) == N_PLANES
            and meta.get('world_target_format') == 'packed_full_tensor_v1'
            and int(meta.get('world_binary_planes', -1)) == WORLD_BINARY_PLANES
        )
        if needs_distance:
            ready = ready and meta.get('distance_target_format') == DISTANCE_TARGET_FORMAT
        if needs_legality:
            ready = ready and meta.get('legality_target_format') == LEGALITY_TARGET_FORMAT
        return ready
    except Exception:
        return False

def gm_cache_ready(paths, expected_samples, cfg=None):
    required, needs_distance, needs_legality = gm_cache_required_arrays(cfg)
    if not os.path.exists(paths['meta']):
        return False
    try:
        with open(paths['meta'], 'r') as f:
            meta = json.load(f)
        if bool(meta.get('sharded', False)):
            if int(meta.get('samples', 0)) < int(expected_samples):
                return False
            shards = meta.get('shards', [])
            if not shards:
                return False
            total = 0
            for shard in shards:
                shard_idx = int(shard['index'])
                shard_samples = int(shard['samples'])
                shard_paths = gm_cache_shard_paths(paths, shard_idx)
                if not gm_cache_shard_ready(shard_paths, shard_samples, cfg):
                    return False
                total += shard_samples
            return total >= int(expected_samples)

        if not all(os.path.exists(paths[k]) for k in required):
            return False
        ready = (
            int(meta.get('samples', 0)) >= int(expected_samples)
            and int(meta.get('n_planes', -1)) == N_PLANES
            and meta.get('world_target_format') == 'packed_full_tensor_v1'
            and int(meta.get('world_binary_planes', -1)) == WORLD_BINARY_PLANES
        )
        if needs_distance:
            ready = ready and meta.get('distance_target_format') == DISTANCE_TARGET_FORMAT
        if needs_legality:
            ready = ready and meta.get('legality_target_format') == LEGALITY_TARGET_FORMAT
        return ready
    except Exception:
        return False

def build_gm_cache(phase, cfg, max_samples, cache_dir=None):
    """Build a compact, resumable shard cache of GM positions for one phase."""
    paths = gm_cache_paths(phase, cfg, max_samples, cache_dir)
    os.makedirs(paths['dir'], exist_ok=True)
    os.makedirs(paths['shard_dir'], exist_ok=True)

    if gm_cache_ready(paths, max_samples, cfg=cfg):
        stat("GM cache", f"Using {paths['key']} ({max_samples:,} samples)", C.GREEN)
        return paths

    section(f"Building GM cache for phase {phase}", "🧱")
    stat("Target samples", f"{max_samples:,}")
    stat("Cache key", paths['key'])
    stat("Shard size", f"{GM_CACHE_SHARD_SAMPLES:,}")
    needs_legality = float(cfg.get('legality_weight', 0.0) or 0.0) > 0

    shard_size = min(int(max_samples), GM_CACHE_SHARD_SAMPLES)
    n_shards = int(math.ceil(int(max_samples) / shard_size))
    completed = []
    skip_games = 0
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        expected = min(shard_size, int(max_samples) - start)
        shard_paths = gm_cache_shard_paths(paths, shard_idx)
        if not gm_cache_shard_ready(shard_paths, expected, cfg=cfg):
            break
        with open(shard_paths['meta'], 'r') as f:
            shard_meta = json.load(f)
        skip_games = int(shard_meta.get('games_seen_after', skip_games))
        completed.append({
            'index': shard_idx,
            'start': start,
            'samples': expected,
            'games_seen_before': int(shard_meta.get('games_seen_before', 0)),
            'games_seen_after': skip_games,
        })

    if completed:
        cached = sum(s['samples'] for s in completed)
        stat("Resuming GM cache", f"{cached:,}/{max_samples:,} samples ({len(completed)}/{n_shards} shards)", C.GREEN)

    streamer = GMStreamer(
        min_elo=cfg['min_elo'],
        max_elo=cfg.get('max_elo'),
        move_range=cfg.get('move_range'),
        min_pieces=cfg.get('min_pieces'),
        max_pieces=cfg.get('max_pieces'),
        skip_games=skip_games,
        include_legality=needs_legality,
    )

    t0 = time.time()
    for shard_idx in range(len(completed), n_shards):
        start = shard_idx * shard_size
        expected = min(shard_size, int(max_samples) - start)
        shard_paths = gm_cache_shard_paths(paths, shard_idx)
        games_before = int(streamer.games_seen)

        stat("Building shard", f"{shard_idx + 1}/{n_shards} ({expected:,} samples)", C.CYAN)
        boards_mm = np.lib.format.open_memmap(
            shard_paths['boards'], mode='w+', dtype=GM_CACHE_BOARD_DTYPE,
            shape=(expected, N_PLANES, 8, 8),
        )
        from_mm = np.lib.format.open_memmap(shard_paths['from_sq'], mode='w+', dtype=np.int16, shape=(expected,))
        to_mm = np.lib.format.open_memmap(shard_paths['to_sq'], mode='w+', dtype=np.int16, shape=(expected,))
        val_mm = np.lib.format.open_memmap(shard_paths['values'], mode='w+', dtype=np.float32, shape=(expected,))
        dist_mm = np.lib.format.open_memmap(shard_paths['distance'], mode='w+', dtype=np.float16, shape=(expected,))
        legal_bits_mm = None
        if needs_legality:
            legal_bits_mm = np.lib.format.open_memmap(
                shard_paths['legal_bits'], mode='w+', dtype=np.uint8,
                shape=(expected, LEGALITY_PACKED_BYTES),
            )
        world_bits_mm = np.lib.format.open_memmap(
            shard_paths['world_bits'], mode='w+', dtype=np.uint8,
            shape=(expected, WORLD_PACKED_BYTES),
        )
        world_halfmove_mm = None
        if N_PLANES > 17:
            world_halfmove_mm = np.lib.format.open_memmap(
                shard_paths['world_halfmove'], mode='w+', dtype=np.uint8, shape=(expected,)
            )

        written = 0
        while written < expected:
            batch = streamer.get_batch(min(2048, expected - written))
            if batch is None:
                break

            n = batch['boards'].shape[0]
            end = written + n
            boards = batch['boards'].numpy()
            if GM_CACHE_BOARD_DTYPE == np.uint8:
                boards = np.rint(np.clip(boards, 0, 1)).astype(np.uint8)
            else:
                boards = boards.astype(GM_CACHE_BOARD_DTYPE)

            boards_mm[written:end] = boards
            from_mm[written:end] = batch['from_sq'].numpy().astype(np.int16)
            to_mm[written:end] = batch['to_sq'].numpy().astype(np.int16)
            val_mm[written:end] = batch['values'].numpy().astype(np.float32)
            dist_mm[written:end] = batch['distance_targets'].numpy().astype(np.float16)
            world_targets = batch['world_targets'].numpy()
            packed_targets, halfmove_targets = pack_world_targets_np(world_targets)
            world_bits_mm[written:end] = packed_targets
            if legal_bits_mm is not None:
                legal_bits_mm[written:end] = pack_legal_targets_np(batch['legal_targets'].numpy())
            if world_halfmove_mm is not None and halfmove_targets is not None:
                world_halfmove_mm[written:end] = halfmove_targets
            written = end

            global_written = start + written
            if written % 100_000 < n or written == expected:
                rate = max(1, global_written - sum(s['samples'] for s in completed)) / max(1.0, time.time() - t0)
                stat("Cached", f"{global_written:,}/{max_samples:,} ({rate:,.0f}/s)", C.CYAN)

        flush_arrays = [boards_mm, from_mm, to_mm, val_mm, dist_mm, world_bits_mm]
        if legal_bits_mm is not None:
            flush_arrays.append(legal_bits_mm)
        if world_halfmove_mm is not None:
            flush_arrays.append(world_halfmove_mm)
        for arr in flush_arrays:
            arr.flush()

        if written < expected:
            raise RuntimeError(f"GM cache shard {shard_idx + 1}/{n_shards} exhausted at {written:,}/{expected:,} samples")

        shard_meta = {
            'key': paths['key'],
            'phase': phase,
            'shard_index': int(shard_idx),
            'start': int(start),
            'samples': int(expected),
            'n_planes': N_PLANES,
            'board_dtype': str(np.dtype(GM_CACHE_BOARD_DTYPE)),
            'world_target_format': 'packed_full_tensor_v1',
            'world_binary_planes': WORLD_BINARY_PLANES,
            'world_packed_bytes': WORLD_PACKED_BYTES,
            'distance_target_format': DISTANCE_TARGET_FORMAT,
            'distance_target_max_plies': DISTANCE_TARGET_MAX_PLIES,
            'legality_target_format': LEGALITY_TARGET_FORMAT if needs_legality else None,
            'legality_packed_bytes': LEGALITY_PACKED_BYTES if needs_legality else None,
            'games_seen_before': int(games_before),
            'games_seen_after': int(streamer.games_seen),
            'created_at': time.time(),
        }
        tmp_meta = shard_paths['meta'] + ".tmp"
        with open(tmp_meta, 'w') as f:
            json.dump(shard_meta, f, indent=2, sort_keys=True)
        os.replace(tmp_meta, shard_paths['meta'])
        completed.append({
            'index': shard_idx,
            'start': start,
            'samples': expected,
            'games_seen_before': int(games_before),
            'games_seen_after': int(streamer.games_seen),
        })

    meta = {
        'key': paths['key'],
        'phase': phase,
        'samples': int(max_samples),
        'sharded': True,
        'shard_size': int(shard_size),
        'shards': completed,
        'n_planes': N_PLANES,
        'board_dtype': str(np.dtype(GM_CACHE_BOARD_DTYPE)),
        'world_target_format': 'packed_full_tensor_v1',
        'world_binary_planes': WORLD_BINARY_PLANES,
        'world_packed_bytes': WORLD_PACKED_BYTES,
        'distance_target_format': DISTANCE_TARGET_FORMAT,
        'distance_target_max_plies': DISTANCE_TARGET_MAX_PLIES,
        'legality_target_format': LEGALITY_TARGET_FORMAT if needs_legality else None,
        'legality_packed_bytes': LEGALITY_PACKED_BYTES if needs_legality else None,
        'config': {
            'min_elo': cfg.get('min_elo'),
            'max_elo': cfg.get('max_elo'),
            'move_range': cfg.get('move_range'),
            'min_pieces': cfg.get('min_pieces'),
            'max_pieces': cfg.get('max_pieces'),
        },
        'created_at': time.time(),
    }
    tmp_meta = paths['meta'] + ".tmp"
    with open(tmp_meta, 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    os.replace(tmp_meta, paths['meta'])
    stat("GM cache ready", f"{int(meta['samples']):,} samples", C.GREEN)
    return paths

class GMCacheDataset(Dataset):
    def __init__(self, paths, start=0, end=None):
        with open(paths['meta'], 'r') as f:
            self.meta = json.load(f)
        total_samples = int(self.meta['samples'])
        self.start = max(0, int(start))
        self.end = total_samples if end is None else min(total_samples, int(end))
        self.samples = max(0, self.end - self.start)
        self.sharded = bool(self.meta.get('sharded', False))
        self.shards = []
        if self.sharded:
            for shard in self.meta.get('shards', []):
                shard_idx = int(shard['index'])
                shard_paths = gm_cache_shard_paths(paths, shard_idx)
                self.shards.append({
                    'start': int(shard['start']),
                    'end': int(shard['start']) + int(shard['samples']),
                    'boards': np.load(shard_paths['boards'], mmap_mode='r'),
                    'from_sq': np.load(shard_paths['from_sq'], mmap_mode='r'),
                    'to_sq': np.load(shard_paths['to_sq'], mmap_mode='r'),
                    'values': np.load(shard_paths['values'], mmap_mode='r'),
                    'distance': np.load(shard_paths['distance'], mmap_mode='r') if os.path.exists(shard_paths['distance']) else None,
                    'legal_bits': np.load(shard_paths['legal_bits'], mmap_mode='r') if os.path.exists(shard_paths['legal_bits']) else None,
                    'world_bits': np.load(shard_paths['world_bits'], mmap_mode='r') if os.path.exists(shard_paths['world_bits']) else None,
                    'world_halfmove': (
                        np.load(shard_paths['world_halfmove'], mmap_mode='r')
                        if N_PLANES > 17 and os.path.exists(shard_paths['world_halfmove'])
                        else None
                    ),
                })
        else:
            self.boards = np.load(paths['boards'], mmap_mode='r')
            self.from_sq = np.load(paths['from_sq'], mmap_mode='r')
            self.to_sq = np.load(paths['to_sq'], mmap_mode='r')
            self.values = np.load(paths['values'], mmap_mode='r')
            self.distance = np.load(paths['distance'], mmap_mode='r') if os.path.exists(paths['distance']) else None
            self.legal_bits = np.load(paths['legal_bits'], mmap_mode='r') if os.path.exists(paths['legal_bits']) else None
            self.world_bits = np.load(paths['world_bits'], mmap_mode='r') if os.path.exists(paths['world_bits']) else None
            self.world_halfmove = (
                np.load(paths['world_halfmove'], mmap_mode='r')
                if N_PLANES > 17 and os.path.exists(paths['world_halfmove'])
                else None
            )

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        real_idx = self.start + idx
        if self.sharded:
            shard_obj = None
            for shard in self.shards:
                if shard['start'] <= real_idx < shard['end']:
                    shard_obj = shard
                    break
            if shard_obj is None:
                raise IndexError(real_idx)
            local_idx = real_idx - shard_obj['start']
            boards = shard_obj['boards']
            from_sq = shard_obj['from_sq']
            to_sq = shard_obj['to_sq']
            values = shard_obj['values']
            distance = shard_obj['distance']
            legal_bits = shard_obj['legal_bits']
            world_bits = shard_obj['world_bits']
            world_halfmove = shard_obj['world_halfmove']
        else:
            local_idx = real_idx
            boards = self.boards
            from_sq = self.from_sq
            to_sq = self.to_sq
            values = self.values
            distance = self.distance
            legal_bits = self.legal_bits
            world_bits = self.world_bits
            world_halfmove = self.world_halfmove

        board = torch.from_numpy(np.array(boards[local_idx], dtype=np.float32, copy=True))
        sample = (
            board,
            torch.tensor(int(from_sq[local_idx]), dtype=torch.long),
            torch.tensor(int(to_sq[local_idx]), dtype=torch.long),
            torch.tensor(float(values[local_idx]), dtype=torch.float32),
        )
        if world_bits is None:
            return sample
        world_target = torch.from_numpy(unpack_world_target_np(
            world_bits[local_idx],
            None if world_halfmove is None else world_halfmove[local_idx],
        ))
        if distance is None:
            if legal_bits is None:
                return sample + (world_target,)
            legal_target = torch.from_numpy(unpack_legal_target_np(legal_bits[local_idx]))
            distance_target = torch.tensor(0.0, dtype=torch.float32)
            return sample + (world_target, distance_target, legal_target)
        distance_target = torch.tensor(float(distance[local_idx]), dtype=torch.float32)
        if legal_bits is None:
            return sample + (world_target, distance_target)
        legal_target = torch.from_numpy(unpack_legal_target_np(legal_bits[local_idx]))
        return sample + (world_target, distance_target, legal_target)

def gm_cache_train_val_counts(cfg, total_samples):
    val_size = int(cfg.get('val_size', cfg.get('gm_val_size', 0)) or 0)
    val_size = max(0, min(val_size, max(0, int(total_samples) - 1)))
    train_size = int(total_samples) - val_size
    return train_size, val_size

class GMCacheBatcher:
    def __init__(self, phase, cfg, cache_paths, batch_size=None):
        with open(cache_paths['meta'], 'r') as f:
            meta = json.load(f)
        train_size, val_size = gm_cache_train_val_counts(cfg, int(meta['samples']))
        dataset = GMCacheDataset(cache_paths, start=0, end=train_size)
        sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
        num_workers = int(cfg.get('num_workers', min(4, os.cpu_count() or 2)))
        loader_kwargs = {
            'batch_size': int(batch_size or cfg['batch_size']),
            'shuffle': sampler is None,
            'sampler': sampler,
            'num_workers': num_workers,
            'pin_memory': IS_CUDA,
            'drop_last': True,
        }
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 4

        self.phase = phase
        self.cfg = cfg
        self.sampler = sampler
        self.loader = DataLoader(dataset, **loader_kwargs)
        self.iterator = iter(self.loader)
        self.epoch = 0
        stat("GM cache samples", f"{len(dataset):,}", C.GREEN)
        if val_size:
            stat("GM held-out samples", f"{val_size:,}", C.CYAN)

    def get_batch(self, batch_size=None):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        boards, f_sq, t_sq, values = batch[:4]

        out = {
            'boards': boards,
            'from_sq': f_sq,
            'to_sq': t_sq,
            'values': values,
            'value_mask': torch.ones(values.shape[0], dtype=torch.float32),
        }
        if len(batch) > 4:
            out['world_targets'] = batch[4]
        if len(batch) > 5:
            out['distance_targets'] = batch[5]
            out['distance_mask'] = torch.ones_like(batch[5], dtype=torch.float32)
        if len(batch) > 6:
            out['legal_targets'] = batch[6]
            out['legal_mask'] = torch.ones(batch[6].shape[0], dtype=torch.float32)
        return out

class PuzzleReplayBatcher:
    def __init__(self, cfg, batch_size):
        dataset = get_dataset(
            multi_move=False,
            include_world_target=float(cfg.get('world_model_weight', 0.0) or 0.0) > 0,
        )
        sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
        num_workers = int(cfg.get('num_workers', 2))
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=IS_CUDA,
            drop_last=True,
        )
        self.sampler = sampler
        self.loader = loader
        self.iterator = iter(loader)
        self.epoch = 0
        stat("Puzzle replay", f"{len(dataset):,} samples", C.GREEN)

    def get_batch(self, batch_size=None):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            if self.sampler is not None:
                self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        boards, f_sq, t_sq, _ratings = batch[:4]
        values = torch.zeros(f_sq.shape[0], dtype=torch.float32)
        out = {
            'boards': boards,
            'from_sq': f_sq,
            'to_sq': t_sq,
            'values': values,
            'value_mask': torch.zeros_like(values),
        }
        if len(batch) > 4:
            out['world_targets'] = batch[4]
        return out

def concat_training_batches(batches):
    boards = torch.cat([b['boards'] for b in batches], dim=0)
    from_sq = torch.cat([b['from_sq'] for b in batches], dim=0)
    to_sq = torch.cat([b['to_sq'] for b in batches], dim=0)
    values = torch.cat([b['values'] for b in batches], dim=0)
    world_targets = None
    if all('world_targets' in b for b in batches):
        world_targets = torch.cat([b['world_targets'] for b in batches], dim=0)
    distance_targets = torch.cat([
        b.get('distance_targets', torch.zeros(b['values'].shape[0], dtype=torch.float32))
        for b in batches
    ], dim=0)
    distance_mask = torch.cat([
        b.get('distance_mask', torch.zeros(b['values'].shape[0], dtype=torch.float32))
        for b in batches
    ], dim=0)
    legal_targets = None
    if any('legal_targets' in b for b in batches):
        legal_targets = torch.cat([
            b.get('legal_targets', torch.zeros(b['values'].shape[0], 64, 64, dtype=torch.float32))
            for b in batches
        ], dim=0)
    legal_mask = torch.cat([
        b.get('legal_mask', torch.zeros(b['values'].shape[0], dtype=torch.float32))
        for b in batches
    ], dim=0)
    masks = [
        b.get('value_mask', torch.ones(b['values'].shape[0], dtype=torch.float32))
        for b in batches
    ]
    value_mask = torch.cat(masks, dim=0)
    perm = torch.randperm(boards.shape[0])
    out = {
        'boards': boards[perm],
        'from_sq': from_sq[perm],
        'to_sq': to_sq[perm],
        'values': values[perm],
        'value_mask': value_mask[perm],
        'distance_targets': distance_targets[perm],
        'distance_mask': distance_mask[perm],
        'legal_mask': legal_mask[perm],
    }
    if world_targets is not None:
        out['world_targets'] = world_targets[perm]
    if legal_targets is not None:
        out['legal_targets'] = legal_targets[perm]
    return out

class ReplayMixedBatcher:
    def __init__(self, main_source, replay_sources):
        self.main_source = main_source
        self.replay_sources = replay_sources

    def get_batch(self, batch_size=None):
        batches = []
        main_batch = self.main_source.get_batch()
        if main_batch is None:
            return None
        batches.append(main_batch)

        for _name, source in self.replay_sources:
            batch = source.get_batch()
            if batch is not None:
                batches.append(batch)

        if len(batches) == 1:
            return batches[0]
        return concat_training_batches(batches)

class FixedBatchSource:
    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = int(batch_size)

    def get_batch(self, batch_size=None):
        return self.source.get_batch(self.batch_size)

# ============================================================================
# TRAINING ENGINE
# ============================================================================
def train_phase(phase, model, optimizer, streamer, config):
    """Run one phase of curriculum training."""
    max_steps = config['max_steps']
    batch_size = config['batch_size']
    save_every = config.get('save_every', 500)
    max_minutes = config.get('max_minutes', 120)

    scaler = torch.amp.GradScaler('cuda', enabled=IS_CUDA)
    model.train()

    t0 = time.time()
    step = config.get('start_step', 0)
    ema = {'loss': None, 'pol': None, 'val': None,
           'aux': None, 'reg': None, 'world': None, 'dist': None,
           'legal': None, 'depth': None, 'acc': None}
    alpha = 0.05
    sharpening_announced = False
    epoch = 0

    if is_main():
        print()
    while step < max_steps:
        # Wall-clock safety
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > max_minutes:
            if is_main():
                print(f"\n    {C.YELLOW}⏰ Time limit ({max_minutes}m) reached at step {step}{C.RESET}")
            break

        batch = streamer.get_batch(batch_size)
        if batch is None:
            if is_main():
                print(f"\n    {C.RED}⚠ Stream exhausted at step {step}{C.RESET}")
            break

        lr_now = set_optimizer_lr(optimizer, scheduled_lr(config, step), config['lr'])

        boards = batch['boards'].to(DEVICE)
        f_sq = batch['from_sq'].to(DEVICE)
        t_sq = batch['to_sq'].to(DEVICE)
        vals = batch['values'].to(DEVICE)
        value_mask = batch.get('value_mask')
        if value_mask is not None:
            value_mask = value_mask.to(DEVICE)
        world_targets = batch.get('world_targets')
        if world_targets is not None:
            world_targets = world_targets.to(DEVICE)
        distance_targets = batch.get('distance_targets')
        if distance_targets is not None:
            distance_targets = distance_targets.to(DEVICE)
        distance_mask = batch.get('distance_mask')
        if distance_mask is not None:
            distance_mask = distance_mask.to(DEVICE)
        legal_targets = batch.get('legal_targets')
        if legal_targets is not None:
            legal_targets = legal_targets.to(DEVICE)
        legal_mask = batch.get('legal_mask')
        if legal_mask is not None:
            legal_mask = legal_mask.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        step_cfg = phase_thinking_config_for_step(config, step)
        if is_main() and step_cfg.get('sharpening') and not sharpening_announced:
            print(f"\n    {C.GREEN}🔎 Final 10k sharpening: depth 16 only, lighter aux, small consistency{C.RESET}")
            sharpening_announced = True

        with torch.amp.autocast('cuda', enabled=IS_CUDA):
            loss_pack = adaptive_training_loss(
                model, boards, f_sq, t_sq, step_cfg,
                vals=vals, value_mask=value_mask, value_weight=0.5,
                world_targets=world_targets,
                distance_targets=distance_targets,
                distance_mask=distance_mask,
                legal_targets=legal_targets,
                legal_mask=legal_mask,
            )
            loss = loss_pack['loss']
            loss_policy = loss_pack['policy']
            loss_value = loss_pack['value']
            from_logits = loss_pack['from_logits']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # EMA tracking
        pred_f = from_logits.argmax(dim=-1)
        acc = (pred_f == f_sq).float().mean().item() * 100
        for k, v in [('loss', loss.item()), ('pol', loss_policy.item()),
                      ('val', loss_value.item()), ('aux', loss_pack['aux'].detach().item()),
                      ('reg', loss_pack['reg'].detach().item()),
                      ('world', loss_pack['world'].detach().item()),
                      ('dist', loss_pack['distance'].detach().item()),
                      ('legal', loss_pack['legality'].detach().item()),
                      ('depth', float(loss_pack['n_think'])), ('acc', acc)]:
            ema[k] = v if ema[k] is None else (1 - alpha) * ema[k] + alpha * v

        # Live progress
        if is_main() and step % 20 == 0:
            p_color = C.GREEN if ema['pol'] < 3.0 else C.YELLOW if ema['pol'] < 4.0 else C.RED
            a_color = C.GREEN if ema['acc'] > 30 else C.YELLOW if ema['acc'] > 20 else C.RED
            bar_filled = int(30 * step / max_steps)
            bar = f"{'█' * bar_filled}{'░' * (30 - bar_filled)}"
            thought_stats = ""
            if adaptive_thinking_enabled(step_cfg):
                mode = f"{C.GREEN}Sharp{C.RESET} " if step_cfg.get('sharpening') else ""
                thought_stats = (
                    f"{mode}"
                    f"T:{C.BLUE}{ema['depth']:.1f}{C.RESET} "
                    f"Aux:{C.BLUE}{ema['aux']:.3f}{C.RESET} "
                    f"Reg:{C.BLUE}{ema['reg']:.3f}{C.RESET} "
                    f"WM:{C.BLUE}{ema['world']:.3f}{C.RESET} "
                    f"DT:{C.BLUE}{ema['dist']:.3f}{C.RESET} "
                    f"LG:{C.BLUE}{ema['legal']:.3f}{C.RESET} "
                    f"WW:{C.BLUE}{step_cfg.get('world_model_weight', 0.0):.3f}{C.RESET} "
                    f"DW:{C.BLUE}{step_cfg.get('distance_weight', 0.0):.3f}{C.RESET} "
                    f"PP:{C.BLUE}{step_cfg.get('policy_pressure_weight', 0.0):.3f}{C.RESET} "
                )

            print(f"    {C.DIM}Step {step:05d}{C.RESET} {C.CYAN}{bar}{C.RESET} "
                  f"Loss:{p_color}{ema['loss']:.3f}{C.RESET} "
                  f"Pol:{p_color}{ema['pol']:.3f}{C.RESET} "
                  f"Val:{C.MAGENTA}{ema['val']:.3f}{C.RESET} "
                  f"Acc:{a_color}{ema['acc']:.1f}%{C.RESET} "
                  f"{thought_stats}"
                  f"LR:{lr_now:.1e}",
                  end="\r" if step + 20 < max_steps else "\n")

        # Checkpoint
        if step > 0 and step % save_every == 0:
            save_checkpoint(
                model, optimizer, phase, step, ema,
                lr=lr_now,
                lr_schedule=lr_schedule_config(config),
            )
            ddp_barrier()

        # Mid-training spot-check: quick 10-game Stockfish benchmark every 5000 steps
        if step > 0 and step % 5000 == 0:
            if is_main():
                print(f"\n    {C.CYAN}{'─'*50}{C.RESET}")
                print(f"    {C.BOLD}📊 SPOT CHECK @ step {step}{C.RESET}")
                safe_benchmark_vs_stockfish(model, [1500, 1800, 2000], n_games=10)
                print(f"    {C.CYAN}{'─'*50}{C.RESET}\n")
            ddp_barrier()
            model.train()  # Resume training mode

        step += 1

    # Final save
    save_checkpoint(
        model, optimizer, phase, step, ema,
        lr=scheduled_lr(config, min(step, max_steps - 1)),
        lr_schedule=lr_schedule_config(config),
    )
    ddp_barrier()
    return step, ema


# ============================================================================
# STOCKFISH BENCHMARK
# ============================================================================
def benchmark_vs_stockfish(model, elo_levels=None, n_games=30):
    """Quick benchmark against Stockfish at various Elos."""
    try:
        import chess.engine
    except ImportError:
        print(f"    {C.RED}chess.engine not available{C.RESET}")
        return {}

    if elo_levels is None:
        elo_levels = [1800, 2000, 2200]

    # Find Stockfish
    engine = None
    for path in ["stockfish", "/usr/games/stockfish", "/opt/homebrew/bin/stockfish"]:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(path)
            break
        except FileNotFoundError:
            continue
        except Exception:
            continue

    if engine is None:
        print(f"    {C.RED}Stockfish not found — skipping benchmark{C.RESET}")
        return {}

    was_training = model.training
    bench_model = base_model(model)
    model.eval()
    bench_model.eval()
    results = {}

    try:
        engine.configure({"UCI_LimitStrength": True})

        for elo in elo_levels:
            engine.configure({"UCI_Elo": elo})
            wins, draws = 0, 0

            for i in range(n_games):
                board = chess.Board()
                white_is_model = (i % 2 == 0)
                moves_played = 0

                while not board.is_game_over() and moves_played < 300:
                    if (board.turn == chess.WHITE) == white_is_model:
                        # Use the unwrapped model here: DataParallel breaks on batch-size-1 eval.
                        bt = board_to_tensor(board).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            fl, tl, _ = bench_model(bt, n_think=N_THINK)
                        fp = torch.softmax(fl[0] / 0.2, dim=0)
                        legal = list(board.legal_moves)
                        if not legal: break
                        scored = []
                        for m in legal:
                            f, t = move_to_indices(m, board)
                            tp = torch.softmax(tl[0, f] / 0.2, dim=0)
                            scored.append((m, fp[f].item() * tp[t].item()))
                        scored.sort(key=lambda x: -x[1])
                        board.push(scored[0][0])
                    else:
                        res = engine.play(board, chess.engine.Limit(time=0.05))
                        if res.move is None: break
                        board.push(res.move)
                    moves_played += 1

                r = board.result()
                if r == '1-0' and white_is_model: wins += 1
                elif r == '0-1' and not white_is_model: wins += 1
                elif '1/2' in r: draws += 1

            wr = (wins + 0.5 * draws) / n_games * 100
            results[elo] = {'wins': wins, 'draws': draws, 'score': wr}

            wr_color = C.GREEN if wr >= 40 else C.YELLOW if wr >= 20 else C.RED
            bar_filled = int(20 * wr / 100)
            bar = f"{'█' * bar_filled}{'░' * (20 - bar_filled)}"
            print(f"      vs SF-{elo} {wr_color}{bar} {wr:5.1f}%{C.RESET} "
                  f"({wins}W {draws}D/{n_games})")
    finally:
        engine.quit()
        model.train(was_training)
    return results

def safe_benchmark_vs_stockfish(model, elo_levels=None, n_games=30):
    """Benchmark helper that can never kill an overnight training run."""
    try:
        return benchmark_vs_stockfish(model, elo_levels, n_games)
    except Exception as exc:
        print(f"    {C.YELLOW}Benchmark skipped after error: {type(exc).__name__}: {exc}{C.RESET}")
        model.train()
        return {}


# ============================================================================
# MAIN CURRICULUM
# ============================================================================
ADAPTIVE_THINKING = {
    'adaptive_thinking': True,
    'think_depths': [2, 4, 6, 8, 12, 16],
    'aux_steps': [2, 4, 6, 8, 12, 16],
    'aux_loss_weight': 0.35,
    'consistency_anchor': 8,
    'consistency_weight': 0.015,
    'value_consistency_weight': 0.02,
    'confidence_reg_weight': 0.03,
    'depth_dropout': 0.25,
    # Automatic final-phase sharpening: every supervised phase spends its
    # final 10k steps specializing the trained loop at full depth.
    'sharpen_steps': 10000,
    'sharpen_n_think': 16,
    'sharpen_aux_steps': [8, 16],
    'sharpen_aux_loss_weight': 0.12,
    'sharpen_consistency_anchor': 8,
    'sharpen_consistency_weight': 0.01,
    'sharpen_value_consistency_weight': 0.01,
    'sharpen_confidence_reg_weight': 0.01,
    'sharpen_depth_dropout': 0.0,
    # Small explicit world-model auxiliary: board + chosen move -> exact next
    # board tensor. It ramps up gently so policy/value remain primary early.
    'world_model_start_weight': 0.01,
    'world_model_weight': 0.04,
    'world_n_think': 2,
    'world_model_max_batch': 256,
    'world_empty_weight': 0.15,
    'world_piece_weight': 1.0,
    'world_changed_weight': 3.0,
    # Optional terminal-distance auxiliary. Kept off by default and enabled
    # only for endgames so policy/value are not dragged toward "finish fast"
    # before the model can evaluate whether it is winning.
    'distance_start_weight': 0.0,
    'distance_weight': 0.0,
    'distance_n_think': 16,
    'distance_model_max_batch': 256,
    # Inbuilt conversion pressure: winning, near-terminal examples get a
    # slightly stronger policy gradient, so urgency is distilled into the
    # policy itself instead of being applied as a move-time trick.
    'policy_pressure_start_weight': 0.0,
    'policy_pressure_weight': 0.0,
    'policy_pressure_win_threshold': 0.25,
    'policy_pressure_survival_weight': 0.5,
    'policy_pressure_loss_downweight': 0.25,
    'policy_pressure_max_multiplier': 1.5,
    # Optional legality auxiliary. Keep off for current phase caches unless a
    # future cache/RL stream provides exact legal move maps.
    'legality_weight': 0.0,
    'legality_n_think': 16,
    'legality_model_max_batch': 256,
    'legality_pos_weight': 16.0,
}

PHASES = {
    0: {
        'name': '🧩 TACTICS (Lichess Puzzles)',
        'type': 'puzzle',
        'lr': 1e-3, 'batch_size': 1024, 'max_steps': 40000, 'n_think': 16,
        'warmup_steps': 1000, 'min_lr': 2e-5, 'warmup_start_lr': 1e-4,
        'val_size': 50000, 'val_batch_size': 1024,
        **ADAPTIVE_THINKING,
        'max_minutes': 240, 'save_every': 2000,
    },
    1: {
        'name': '📘 FOUNDATION (Elo ≥ 1800)',
        'type': 'gm',
        'min_elo': 1800, 'move_range': None, 'max_pieces': None, 'min_pieces': None,
        'lr': 5e-4, 'batch_size': 768, 'max_steps': 40000, 'n_think': 16,
        'warmup_steps': 1000, 'min_lr': 1e-5, 'warmup_start_lr': 5e-5,
        'val_size': 50000, 'val_batch_size': 1024,
        'replay': {'puzzle': 0.10, 'low_elo': 0.20},
        'low_elo_replay': {'min_elo': 1300, 'max_elo': 1799},
        **ADAPTIVE_THINKING,
        'max_minutes': 240, 'save_every': 2000,
    },
    2: {
        'name': '📖 OPENINGS (Elo ≥ 2000, moves 1-15)',
        'type': 'gm',
        'min_elo': 2000, 'move_range': (0, 15), 'max_pieces': None, 'min_pieces': None,
        'lr': 3e-4, 'batch_size': 768, 'max_steps': 30000, 'n_think': 16,
        'warmup_steps': 750, 'min_lr': 6e-6, 'warmup_start_lr': 3e-5,
        'val_size': 50000, 'val_batch_size': 1024,
        'replay': {'puzzle': 0.10, 'gm': 0.10},
        'replay_phases': [1],
        **ADAPTIVE_THINKING,
        'max_minutes': 180, 'save_every': 2000,
    },
    3: {
        'name': '⚔️  MIDDLEGAME (Elo ≥ 2200, moves 10-50)',
        'type': 'gm',
        'min_elo': 2200, 'move_range': (10, 50), 'max_pieces': None, 'min_pieces': None,
        'lr': 2e-4, 'batch_size': 256, 'max_steps': 50000, 'n_think': 16,
        'warmup_steps': 1000, 'min_lr': 4e-6, 'warmup_start_lr': 2e-5,
        'val_size': 50000, 'val_batch_size': 1024,
        'replay': {'puzzle': 0.10, 'gm': 0.10},
        'replay_phases': [1, 2],
        **ADAPTIVE_THINKING,
        'max_minutes': 300, 'save_every': 2000,
    },
    4: {
        'name': '♔ ENDGAMES (Elo ≥ 2000, ≤ 10 pieces)',
        'type': 'gm',
        'min_elo': 2000, 'move_range': None, 'max_pieces': 10, 'min_pieces': None,
        'lr': 1e-4, 'batch_size': 256, 'max_steps': 30000, 'n_think': 16,
        'warmup_steps': 750, 'min_lr': 2e-6, 'warmup_start_lr': 1e-5,
        'val_size': 50000, 'val_batch_size': 1024,
        'replay': {'puzzle': 0.15, 'gm': 0.15},
        'replay_phases': [1, 2, 3],
        **ADAPTIVE_THINKING,
        'distance_start_weight': 0.01,
        'distance_weight': 0.05,
        'policy_pressure_start_weight': 0.05,
        'policy_pressure_weight': 0.35,
        'max_minutes': 180, 'save_every': 2000,
    },
    5: {
        'name': '🔥 REINFORCEMENT (vs Stockfish)',
        'type': 'rl',
        'lr': 3e-5, 'batch_size': 64, 'max_steps': 10000, 'n_think': 16,
        'warmup_steps': 300, 'min_lr': 5e-7, 'warmup_start_lr': 3e-6,
        'legality_weight': 0.03,
        'max_minutes': 120, 'save_every': 1000,
    },
}


def train_puzzle_phase(model, optimizer, cfg):
    """Phase 0: Train on Lichess puzzles for tactical foundation."""
    if is_main():
        section("Loading Lichess puzzle dataset", "📥")
    dataset, val_dataset, full_len = get_puzzle_train_val(cfg)

    # Use DistributedSampler if DDP is active
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None
    num_workers = int(cfg.get('num_workers', 2))
    loader = DataLoader(dataset, batch_size=cfg['batch_size'],
                        shuffle=(sampler is None), sampler=sampler,
                        num_workers=num_workers, pin_memory=IS_CUDA, drop_last=True)

    max_steps = cfg['max_steps']
    save_every = cfg.get('save_every', 500)
    max_minutes = cfg.get('max_minutes', 120)

    scaler = torch.amp.GradScaler('cuda', enabled=IS_CUDA)
    model.train()

    t0 = time.time()
    step = cfg.get('start_step', 0)
    ema = {'loss': None, 'pol': None, 'val': None,
           'aux': None, 'reg': None, 'world': None, 'dist': None,
           'legal': None, 'depth': None, 'acc': None}
    alpha = 0.05
    sharpening_announced = False

    stat("Puzzle samples", f"{full_len:,}")
    stat("Puzzle train samples", f"{len(dataset):,}")
    stat("Puzzle held-out samples", f"{len(val_dataset):,}")
    stat("Max steps", str(max_steps))
    if is_main():
        print()

    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= max_steps:
                break
            elapsed_min = (time.time() - t0) / 60
            if elapsed_min > max_minutes:
                if is_main():
                    print(f"\n    {C.YELLOW}⏰ Time limit ({max_minutes}m) reached{C.RESET}")
                save_checkpoint(
                    model, optimizer, 0, step, ema,
                    lr=scheduled_lr(cfg, min(step, max_steps - 1)),
                    lr_schedule=lr_schedule_config(cfg),
                )
                ddp_barrier()
                return step, ema

            lr_now = set_optimizer_lr(optimizer, scheduled_lr(cfg, step), cfg['lr'])

            boards, f_sq, t_sq, ratings = batch[:4]
            world_targets = batch[4] if len(batch) > 4 else None
            boards = boards.to(DEVICE)
            f_sq = f_sq.to(DEVICE)
            t_sq = t_sq.to(DEVICE)
            if world_targets is not None:
                world_targets = world_targets.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            step_cfg = phase_thinking_config_for_step(cfg, step)
            if is_main() and step_cfg.get('sharpening') and not sharpening_announced:
                print(f"\n    {C.GREEN}🔎 Final 10k sharpening: depth 16 only, lighter aux, small consistency{C.RESET}")
                sharpening_announced = True

            with torch.amp.autocast('cuda', enabled=IS_CUDA):
                loss_pack = adaptive_training_loss(
                    model, boards, f_sq, t_sq, step_cfg,
                    vals=None, value_mask=None, value_weight=0.0,
                    world_targets=world_targets,
                )
                loss = loss_pack['loss']
                loss_policy = loss_pack['policy']
                from_logits = loss_pack['from_logits']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA
            acc = (from_logits.argmax(dim=-1) == f_sq).float().mean().item() * 100
            for k, v in [('loss', loss.item()), ('pol', loss_policy.item()),
                         ('aux', loss_pack['aux'].detach().item()),
                         ('reg', loss_pack['reg'].detach().item()),
                         ('world', loss_pack['world'].detach().item()),
                         ('dist', loss_pack['distance'].detach().item()),
                         ('legal', loss_pack['legality'].detach().item()),
                         ('depth', float(loss_pack['n_think'])), ('acc', acc)]:
                ema[k] = v if ema[k] is None else (1 - alpha) * ema[k] + alpha * v
            ema['val'] = 0.0  # No value target in puzzles

            if is_main() and step % 20 == 0:
                p_color = C.GREEN if ema['pol'] < 3.0 else C.YELLOW if ema['pol'] < 4.0 else C.RED
                a_color = C.GREEN if ema['acc'] > 30 else C.YELLOW if ema['acc'] > 20 else C.RED
                bar_filled = int(30 * step / max_steps)
                bar = f"{'█' * bar_filled}{'░' * (30 - bar_filled)}"
                thought_stats = ""
                if adaptive_thinking_enabled(step_cfg):
                    mode = f"{C.GREEN}Sharp{C.RESET} " if step_cfg.get('sharpening') else ""
                    thought_stats = (
                        f"{mode}"
                        f"T:{C.BLUE}{ema['depth']:.1f}{C.RESET} "
                        f"Aux:{C.BLUE}{ema['aux']:.3f}{C.RESET} "
                        f"Reg:{C.BLUE}{ema['reg']:.3f}{C.RESET} "
                        f"WM:{C.BLUE}{ema['world']:.3f}{C.RESET} "
                        f"DT:{C.BLUE}{ema['dist']:.3f}{C.RESET} "
                        f"LG:{C.BLUE}{ema['legal']:.3f}{C.RESET} "
                        f"WW:{C.BLUE}{step_cfg.get('world_model_weight', 0.0):.3f}{C.RESET} "
                        f"DW:{C.BLUE}{step_cfg.get('distance_weight', 0.0):.3f}{C.RESET} "
                        f"PP:{C.BLUE}{step_cfg.get('policy_pressure_weight', 0.0):.3f}{C.RESET} "
                    )
                print(f"    {C.DIM}Step {step:05d}{C.RESET} {C.CYAN}{bar}{C.RESET} "
                      f"Loss:{p_color}{ema['loss']:.3f}{C.RESET} "
                      f"Acc:{a_color}{ema['acc']:.1f}%{C.RESET} "
                      f"{thought_stats}"
                      f"LR:{lr_now:.1e}",
                      end="\r" if step + 20 < max_steps else "\n")

            if step > 0 and step % save_every == 0:
                save_checkpoint(
                    model, optimizer, 0, step, ema,
                    lr=lr_now,
                    lr_schedule=lr_schedule_config(cfg),
                )
                ddp_barrier()

            # Mid-training spot-check
            if step > 0 and step % 5000 == 0:
                if is_main():
                    print(f"\n    {C.CYAN}{'─'*50}{C.RESET}")
                    print(f"    {C.BOLD}📊 SPOT CHECK @ step {step}{C.RESET}")
                    safe_benchmark_vs_stockfish(model, [1500, 1800, 2000], n_games=10)
                    print(f"    {C.CYAN}{'─'*50}{C.RESET}\n")
                ddp_barrier()
                model.train()

            step += 1
        epoch += 1

    save_checkpoint(
        model, optimizer, 0, step, ema,
        lr=scheduled_lr(cfg, min(step, max_steps - 1)),
        lr_schedule=lr_schedule_config(cfg),
    )
    ddp_barrier()
    return step, ema

@torch.no_grad()
def evaluate_puzzle_heldout(model, cfg):
    """Evaluate Phase 0 on the fixed held-out puzzle tail split."""
    _, val_dataset, full_len = get_puzzle_train_val(cfg)
    batch_size = int(cfg.get('val_batch_size', min(cfg.get('batch_size', 1024), 1024)))
    num_workers = int(cfg.get('num_workers', 2))
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=IS_CUDA,
        drop_last=False,
    )

    was_training = model.training
    model.eval()
    n_think = cfg.get('n_think', N_THINK)
    total = 0
    loss_sum = 0.0
    from_correct = 0
    to_true_correct = 0
    full_correct = 0

    section("PHASE 0 HELD-OUT VALIDATION", "🧪")
    stat("Dataset samples", f"{full_len:,}")
    stat("Held-out samples", f"{len(val_dataset):,}")

    for batch in loader:
        boards, f_sq, t_sq, _ratings = batch[:4]
        boards = boards.to(DEVICE)
        f_sq = f_sq.to(DEVICE)
        t_sq = t_sq.to(DEVICE)

        with torch.amp.autocast('cuda', enabled=IS_CUDA):
            from_logits, to_logits, _value = model(boards, n_think=n_think)
            loss_from = F.cross_entropy(from_logits, f_sq, reduction='sum')
            batch_idx = torch.arange(f_sq.size(0), device=f_sq.device)
            true_from_to_logits = to_logits[batch_idx, f_sq]
            loss_to = F.cross_entropy(true_from_to_logits, t_sq, reduction='sum')

        pred_f = from_logits.argmax(dim=-1)
        pred_t_true_from = true_from_to_logits.argmax(dim=-1)
        pred_t_pred_from = to_logits[torch.arange(pred_f.size(0), device=pred_f.device), pred_f].argmax(dim=-1)

        n = f_sq.size(0)
        total += n
        loss_sum += (loss_from + loss_to).item()
        from_correct += (pred_f == f_sq).sum().item()
        to_true_correct += (pred_t_true_from == t_sq).sum().item()
        full_correct += ((pred_f == f_sq) & (pred_t_pred_from == t_sq)).sum().item()

    metrics = {
        'samples': total,
        'loss': loss_sum / max(1, total),
        'from_acc': 100.0 * from_correct / max(1, total),
        'to_acc_true_from': 100.0 * to_true_correct / max(1, total),
        'full_move_acc': 100.0 * full_correct / max(1, total),
    }

    stat("Held-out policy loss", f"{metrics['loss']:.4f}", C.GREEN if metrics['loss'] < 1.5 else C.YELLOW)
    stat("Held-out from acc", f"{metrics['from_acc']:.2f}%")
    stat("Held-out to acc", f"{metrics['to_acc_true_from']:.2f}%")
    stat("Held-out full move acc", f"{metrics['full_move_acc']:.2f}%", C.GREEN if metrics['full_move_acc'] > 50 else C.YELLOW)
    model.train(was_training)
    return metrics

@torch.no_grad()
def evaluate_gm_heldout(model, phase_num, cfg, cache_paths):
    """Evaluate a GM phase on the held-out tail of its disk cache."""
    if not os.path.exists(cache_paths['meta']):
        stat("GM held-out skipped", "cache meta not found", C.YELLOW)
        return None

    with open(cache_paths['meta'], 'r') as f:
        meta = json.load(f)
    cache_samples = int(meta.get('samples', 0))
    if cache_samples <= 0 or not gm_cache_ready(cache_paths, cache_samples, cfg=cfg):
        stat("GM held-out skipped", "cache incomplete or incompatible", C.YELLOW)
        return None

    train_size, val_size = gm_cache_train_val_counts(cfg, cache_samples)
    if val_size <= 0:
        stat("GM held-out skipped", "val_size is 0", C.YELLOW)
        return None

    dataset = GMCacheDataset(cache_paths, start=train_size, end=train_size + val_size)
    batch_size = int(cfg.get('val_batch_size', min(cfg.get('batch_size', 1024), 1024)))
    num_workers = int(cfg.get('num_workers', 2))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=IS_CUDA,
        drop_last=False,
    )

    was_training = model.training
    model.eval()
    n_think = cfg.get('n_think', N_THINK)
    total = 0
    policy_loss_sum = 0.0
    value_mse_sum = 0.0
    value_mae_sum = 0.0
    from_correct = 0
    to_true_correct = 0
    full_correct = 0

    section(f"PHASE {phase_num} HELD-OUT GM VALIDATION", "🧪")
    stat("Cache samples", f"{int(meta['samples']):,}")
    stat("Train samples", f"{train_size:,}")
    stat("Held-out samples", f"{len(dataset):,}")

    for batch in loader:
        boards, f_sq, t_sq, vals = batch[:4]
        boards = boards.to(DEVICE)
        f_sq = f_sq.to(DEVICE)
        t_sq = t_sq.to(DEVICE)
        vals = vals.to(DEVICE)

        with torch.amp.autocast('cuda', enabled=IS_CUDA):
            from_logits, to_logits, value = model(boards, n_think=n_think)
            loss_from = F.cross_entropy(from_logits, f_sq, reduction='sum')
            batch_idx = torch.arange(f_sq.size(0), device=f_sq.device)
            true_from_to_logits = to_logits[batch_idx, f_sq]
            loss_to = F.cross_entropy(true_from_to_logits, t_sq, reduction='sum')
            value_pred = value.squeeze(-1)
            value_mse = F.mse_loss(value_pred, vals, reduction='sum')
            value_mae = F.l1_loss(value_pred, vals, reduction='sum')

        pred_f = from_logits.argmax(dim=-1)
        pred_t_true_from = true_from_to_logits.argmax(dim=-1)
        pred_t_pred_from = to_logits[torch.arange(pred_f.size(0), device=pred_f.device), pred_f].argmax(dim=-1)

        n = f_sq.size(0)
        total += n
        policy_loss_sum += (loss_from + loss_to).item()
        value_mse_sum += value_mse.item()
        value_mae_sum += value_mae.item()
        from_correct += (pred_f == f_sq).sum().item()
        to_true_correct += (pred_t_true_from == t_sq).sum().item()
        full_correct += ((pred_f == f_sq) & (pred_t_pred_from == t_sq)).sum().item()

    metrics = {
        'samples': total,
        'policy_loss': policy_loss_sum / max(1, total),
        'value_mse': value_mse_sum / max(1, total),
        'value_mae': value_mae_sum / max(1, total),
        'from_acc': 100.0 * from_correct / max(1, total),
        'to_acc_true_from': 100.0 * to_true_correct / max(1, total),
        'full_move_acc': 100.0 * full_correct / max(1, total),
    }

    stat("Held-out policy loss", f"{metrics['policy_loss']:.4f}", C.GREEN if metrics['policy_loss'] < 3.0 else C.YELLOW)
    stat("Held-out value MSE", f"{metrics['value_mse']:.4f}")
    stat("Held-out value MAE", f"{metrics['value_mae']:.4f}")
    stat("Held-out from acc", f"{metrics['from_acc']:.2f}%")
    stat("Held-out to acc", f"{metrics['to_acc_true_from']:.2f}%")
    stat("Held-out full move acc", f"{metrics['full_move_acc']:.2f}%", C.GREEN if metrics['full_move_acc'] > 35 else C.YELLOW)
    model.train(was_training)
    return metrics


def make_gm_source(phase_num, cfg, batch_size, use_gm_cache=False,
                   gm_cache_samples=1_000_000, gm_cache_dir=None):
    source_cfg = dict(cfg)
    source_cfg['batch_size'] = int(batch_size)
    if use_gm_cache:
        if is_main():
            build_gm_cache(
                phase_num, cfg,
                max_samples=gm_cache_samples,
                cache_dir=gm_cache_dir,
            )
        if dist.is_initialized():
            dist.barrier()
        cache_paths = gm_cache_paths(
            phase_num, cfg,
            max_samples=gm_cache_samples,
            cache_dir=gm_cache_dir,
        )
        return GMCacheBatcher(phase_num, source_cfg, cache_paths)

    stream = GMStreamer(
        min_elo=cfg['min_elo'],
        max_elo=cfg.get('max_elo'),
        move_range=cfg.get('move_range'),
        min_pieces=cfg.get('min_pieces'),
        max_pieces=cfg.get('max_pieces'),
        include_legality=float(cfg.get('legality_weight', 0.0) or 0.0) > 0,
    )
    return FixedBatchSource(stream, batch_size)

def low_elo_replay_config(cfg):
    low_cfg = dict(cfg)
    spec = cfg.get('low_elo_replay', {})
    low_cfg['name'] = 'LOW-ELO REPLAY (Elo 1300-1799)'
    low_cfg['min_elo'] = int(spec.get('min_elo', 1300))
    low_cfg['max_elo'] = int(spec.get('max_elo', 1799))
    low_cfg['move_range'] = spec.get('move_range', cfg.get('move_range'))
    low_cfg['min_pieces'] = spec.get('min_pieces', cfg.get('min_pieces'))
    low_cfg['max_pieces'] = spec.get('max_pieces', cfg.get('max_pieces'))
    low_cfg['val_size'] = 0
    low_cfg['replay'] = {}
    return low_cfg

def replay_cache_identity_config(cfg):
    """Keep replay cache names stable across training-only loss overrides."""
    replay_cfg = dict(cfg)
    replay_cfg['distance_weight'] = 0.0
    replay_cfg['distance_start_weight'] = 0.0
    replay_cfg['policy_pressure_weight'] = 0.0
    replay_cfg['policy_pressure_start_weight'] = 0.0
    return replay_cfg

def split_replay_counts(total, phases):
    phases = [int(p) for p in phases]
    if not phases or total <= 0:
        return []
    base = total // len(phases)
    rem = total % len(phases)
    return [
        (phase, count)
        for i, phase in enumerate(phases)
        for count in [base + (1 if i < rem else 0)]
        if count > 0
    ]

def make_phase_streamer(phase_num, cfg, use_gm_cache=False,
                        gm_cache_samples=1_000_000, gm_cache_dir=None,
                        replay_gm_cache_samples=2_000_000, replay_gm_cache_dir=None,
                        enable_replay=True, low_elo_cache_samples=2_000_000):
    replay_cfg = cfg.get('replay', {}) if enable_replay else {}
    counts = replay_batch_counts(cfg['batch_size'], replay_cfg)
    replay_gm_cache_dir = replay_gm_cache_dir or REPLAY_GM_CACHE_DIR

    main_source = make_gm_source(
        phase_num, cfg, counts['main'],
        use_gm_cache=use_gm_cache,
        gm_cache_samples=gm_cache_samples,
        gm_cache_dir=gm_cache_dir,
    )

    replay_sources = []
    if counts['puzzle'] > 0:
        replay_sources.append(('puzzle', PuzzleReplayBatcher(cfg, counts['puzzle'])))

    if counts['gm'] > 0:
        replay_phases = cfg.get('replay_phases', [1])
        for replay_phase, replay_batch in split_replay_counts(counts['gm'], replay_phases):
            replay_phase_cfg = dict(PHASES[replay_phase])
            replay_phase_cfg['replay'] = {}
            replay_phase_cfg['val_size'] = 0
            replay_phase_cfg = replay_cache_identity_config(replay_phase_cfg)
            replay_sources.append((
                f"phase-{replay_phase}",
                make_gm_source(
                    replay_phase, replay_phase_cfg, replay_batch,
                    use_gm_cache=use_gm_cache,
                    gm_cache_samples=replay_gm_cache_samples,
                    gm_cache_dir=replay_gm_cache_dir,
                ),
            ))

    if counts['low_elo'] > 0:
        low_cfg = replay_cache_identity_config(low_elo_replay_config(cfg))
        replay_sources.append((
            'low-elo',
            make_gm_source(
                f"{phase_num}_lowelo", low_cfg, counts['low_elo'],
                use_gm_cache=use_gm_cache,
                gm_cache_samples=low_elo_cache_samples,
                gm_cache_dir=replay_gm_cache_dir,
            ),
        ))

    if not replay_sources:
        return main_source
    return ReplayMixedBatcher(main_source, replay_sources)

def build_phase_caches_only(phase_num, cfg, gm_cache_samples=1_000_000,
                            gm_cache_dir=None, replay_gm_cache_samples=2_000_000,
                            replay_gm_cache_dir=None, low_elo_cache_samples=2_000_000,
                            enable_replay=True):
    """Build the on-disk caches a phase needs, without creating/training a model."""
    if cfg['type'] == 'puzzle':
        section(f"Building puzzle cache for phase {phase_num}", "🧱")
        get_dataset(
            multi_move=False,
            include_world_target=float(cfg.get('world_model_weight', 0.0) or 0.0) > 0,
        )
        return

    if cfg['type'] != 'gm':
        stat("Cache build skipped", f"Phase {phase_num} type={cfg['type']}", C.YELLOW)
        return

    section(f"Cache-only phase {phase_num}: {cfg['name']}", "🧱")
    stat("Main cache samples", f"{int(gm_cache_samples):,}")
    stat("Main cache dir", gm_cache_dir or GM_CACHE_DIR)
    build_gm_cache(
        phase_num, cfg,
        max_samples=gm_cache_samples,
        cache_dir=gm_cache_dir,
    )

    if not enable_replay:
        stat("Replay caches", "disabled", C.YELLOW)
        return

    replay_cfg = cfg.get('replay', {})
    replay_gm_cache_dir = replay_gm_cache_dir or REPLAY_GM_CACHE_DIR

    if float(replay_cfg.get('puzzle', 0.0) or 0.0) > 0:
        section("Building puzzle replay cache", "🧩")
        get_dataset(
            multi_move=False,
            include_world_target=float(cfg.get('world_model_weight', 0.0) or 0.0) > 0,
        )

    if float(replay_cfg.get('gm', 0.0) or 0.0) > 0:
        replay_phases = cfg.get('replay_phases', [1])
        stat("Replay GM cache samples", f"{int(replay_gm_cache_samples):,}")
        stat("Replay GM cache dir", replay_gm_cache_dir)
        for replay_phase in replay_phases:
            replay_phase_cfg = dict(PHASES[int(replay_phase)])
            replay_phase_cfg['replay'] = {}
            replay_phase_cfg['val_size'] = 0
            replay_phase_cfg = replay_cache_identity_config(replay_phase_cfg)
            build_gm_cache(
                int(replay_phase), replay_phase_cfg,
                max_samples=replay_gm_cache_samples,
                cache_dir=replay_gm_cache_dir,
            )

    if float(replay_cfg.get('low_elo', 0.0) or 0.0) > 0:
        low_cfg = replay_cache_identity_config(low_elo_replay_config(cfg))
        stat("Low-Elo replay cache samples", f"{int(low_elo_cache_samples):,}")
        stat("Low-Elo replay cache dir", replay_gm_cache_dir)
        build_gm_cache(
            f"{phase_num}_lowelo", low_cfg,
            max_samples=low_elo_cache_samples,
            cache_dir=replay_gm_cache_dir,
        )

def build_cache_only(start_phase=0, stop_after_phase=None, gm_cache_samples=1_000_000,
                     replay_gm_cache_samples=2_000_000, low_elo_cache_samples=2_000_000,
                     gm_cache_dir=None, replay_gm_cache_dir=None, enable_replay=True):
    start_phase = max(0, min(5, int(start_phase)))
    end_phase = start_phase if stop_after_phase is None else max(0, min(5, int(stop_after_phase)))
    if start_phase > end_phase:
        stat("Cache build skipped", f"start phase {start_phase} after stop phase {end_phase}", C.YELLOW)
        return

    if is_main():
        banner("🧱 CACHE-ONLY BUILD", C.BLUE)
        stat("Phase range", f"{start_phase}-{end_phase}")
        stat("Model training", "skipped", C.GREEN)

    for phase_num in range(start_phase, end_phase + 1):
        cfg = dict(PHASES[phase_num])
        if is_main():
            build_phase_caches_only(
                phase_num, cfg,
                gm_cache_samples=gm_cache_samples,
                gm_cache_dir=gm_cache_dir,
                replay_gm_cache_samples=replay_gm_cache_samples,
                replay_gm_cache_dir=replay_gm_cache_dir,
                low_elo_cache_samples=low_elo_cache_samples,
                enable_replay=enable_replay,
            )
        if dist.is_initialized():
            dist.barrier()

    if is_main():
        banner("CACHE BUILD COMPLETE", C.GREEN)


def run_curriculum(start_phase=0, fresh=False, all_steps=False, force_phase=False,
                   use_gm_cache=False, gm_cache_samples=1_000_000,
                   replay_gm_cache_samples=2_000_000,
                   low_elo_cache_samples=2_000_000,
                   gm_cache_dir=None, gm_batch_size=None, phase12_batch_size=None,
                   replay_gm_cache_dir=None,
                   phase0_batch_size=None, num_workers=None, enable_replay=True,
                   eval_phase0_before_start=False, disable_adaptive_thinking=False,
                   think_depths=None, stop_after_phase=None, phase_max_steps=None,
                   aux_loss_weight=None, consistency_weight=None,
                   value_consistency_weight=None, confidence_reg_weight=None,
                   depth_dropout=None, world_model_weight=None,
                   world_model_start_weight=None, world_model_max_batch=None,
                   world_n_think=None,
                   distance_weight=None, distance_start_weight=None,
                   distance_model_max_batch=None, distance_n_think=None,
                   policy_pressure_weight=None, policy_pressure_start_weight=None,
                   legality_weight=None, legality_model_max_batch=None,
                   legality_n_think=None,
                   init_from_phase=None, ignore_phase_checkpoint=False):
    apply_runtime_options(all_steps=all_steps)
    apply_thinking_overrides(
        disable_adaptive=disable_adaptive_thinking,
        think_depths=think_depths,
    )
    apply_thinking_loss_overrides(
        aux_loss_weight=aux_loss_weight,
        consistency_weight=consistency_weight,
        value_consistency_weight=value_consistency_weight,
        confidence_reg_weight=confidence_reg_weight,
        depth_dropout=depth_dropout,
        world_model_weight=world_model_weight,
        world_model_start_weight=world_model_start_weight,
        world_model_max_batch=world_model_max_batch,
        world_n_think=world_n_think,
        distance_weight=distance_weight,
        distance_start_weight=distance_start_weight,
        distance_model_max_batch=distance_model_max_batch,
        distance_n_think=distance_n_think,
        policy_pressure_weight=policy_pressure_weight,
        policy_pressure_start_weight=policy_pressure_start_weight,
        legality_weight=legality_weight,
        legality_model_max_batch=legality_model_max_batch,
        legality_n_think=legality_n_think,
    )
    apply_phase_max_steps(start_phase, stop_after_phase, phase_max_steps)
    apply_batch_overrides(
        phase0_batch_size=phase0_batch_size,
        gm_batch_size=gm_batch_size,
        phase12_batch_size=phase12_batch_size,
    )
    apply_loader_overrides(num_workers=num_workers)

    if is_main():
        banner("♟️🔥 CHESS GOD V2 — CURRICULUM TRAINING", C.RED)
        print(f"  {C.DIM}10M param Geometric Manifold | 6-Phase Curriculum{C.RESET}")
        print(f"  {C.DIM}Target: 2000-2400 Elo | Device: {DEVICE}{C.RESET}")

    # Create or load model
    model = ChessGod(d1=D1, hidden=HIDDEN, n_encode=N_ENCODE,
                     n_think=N_THINK, dropout=DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main():
        stat("Parameters", f"{total_params:,}")
        stat("Model size", f"{total_params * 4 / 1024 / 1024:.1f} MB")

    # Wrap with DDP or DataParallel
    if dist.is_initialized():
        model = DDP(model, device_ids=[LOCAL_RANK])
        if is_main():
            stat("DDP", f"Enabled ({dist.get_world_size()} GPUs)", C.GREEN)
    elif torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if is_main():
            stat("DataParallel", f"{torch.cuda.device_count()} GPUs (use torchrun for true 2x)", C.YELLOW)
    else:
        if is_main():
            stat("Device", DEVICE)

    # Try to resume
    if init_from_phase is not None:
        init_ckpt = load_checkpoint(int(init_from_phase))
        if init_ckpt is None:
            stat("Init checkpoint missing", f"Phase {init_from_phase}", C.RED)
            return
        if checkpoint_matches_model(init_ckpt):
            load_model_state(model, init_ckpt['model'])
            stat("Initialized from", f"Phase {init_from_phase} checkpoint; starting Phase {start_phase}", C.CYAN)
        else:
            stat("Init checkpoint skipped", "architecture/config mismatch", C.YELLOW)
            return
        ckpt = None
    elif fresh:
        ckpt = None
        stat("Fresh run", "Ignoring existing checkpoints", C.YELLOW)
    elif force_phase:
        ckpt = None
        stat("Forced phase", f"Starting from Phase {start_phase} checkpoint", C.YELLOW)
    else:
        ckpt = load_checkpoint()
    if ckpt and start_phase < ckpt.get('phase', 0):
        if checkpoint_matches_model(ckpt):
            load_model_state(model, ckpt['model'])
            start_phase = ckpt.get('phase', 0)
            stat("Resumed from", f"Phase {start_phase}, step {ckpt.get('step', 0)}", C.CYAN)
        else:
            stat("Checkpoint skipped", "architecture/config mismatch; use --fresh for clean V2", C.YELLOW)
    elif ckpt and start_phase == ckpt.get('phase', 0):
        if checkpoint_matches_model(ckpt):
            load_model_state(model, ckpt['model'])
            stat("Resumed from", f"Phase {start_phase}, step {ckpt.get('step', 0)}", C.CYAN)
        else:
            stat("Checkpoint skipped", "architecture/config mismatch; use --fresh for clean V2", C.YELLOW)
    elif ckpt and start_phase > ckpt.get('phase', 0):
        if checkpoint_matches_model(ckpt):
            load_model_state(model, ckpt['model'])
            stat("Initialized from", f"Phase {ckpt.get('phase', 0)} checkpoint; starting Phase {start_phase}", C.CYAN)
        else:
            stat("Checkpoint skipped", "architecture/config mismatch; use --fresh for clean V2", C.YELLOW)

    if eval_phase0_before_start and start_phase > 0:
        if is_main():
            phase0_ckpt = load_checkpoint(0)
            if phase0_ckpt is None:
                stat("Phase 0 eval skipped", "no Phase 0 checkpoint found", C.YELLOW)
            elif checkpoint_matches_model(phase0_ckpt):
                saved_state = cloned_model_state_dict(model)
                load_model_state(model, phase0_ckpt['model'])
                evaluate_puzzle_heldout(base_model(model), PHASES[0])
                load_model_state(model, saved_state)
            else:
                stat("Phase 0 eval skipped", "checkpoint config mismatch", C.YELLOW)
        ddp_barrier()

    if stop_after_phase is not None:
        stop_after_phase = max(0, min(5, int(stop_after_phase)))
        if start_phase > stop_after_phase:
            stat("Stop requested", f"start phase {start_phase} is after stop phase {stop_after_phase}", C.YELLOW)
            return

    end_phase = 5 if stop_after_phase is None else stop_after_phase

    for phase_num in range(start_phase, end_phase + 1):
        cfg = dict(PHASES[phase_num])
        banner(f"PHASE {phase_num}/5: {cfg['name']}", C.WHITE)

        optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.01)

        # Try loading phase-specific checkpoint
        phase_ckpt = None if (fresh or ignore_phase_checkpoint or init_from_phase is not None) else load_checkpoint(phase_num)
        if phase_ckpt:
            if checkpoint_matches_model(phase_ckpt):
                load_model_state(model, phase_ckpt['model'])
                stat("Model resumed", f"Phase {phase_num}, step {phase_ckpt.get('step', 0)}", C.CYAN)
                try:
                    optimizer.load_state_dict(phase_ckpt['optimizer'])
                    stat("Optimizer resumed", f"Phase {phase_num}", C.CYAN)
                except Exception:
                    pass
                set_optimizer_lr(optimizer, scheduled_lr(cfg, phase_ckpt.get('step', 0)), cfg['lr'])
                cfg['start_step'] = phase_ckpt.get('step', 0)
            else:
                stat("Phase checkpoint skipped", f"Phase {phase_num} config mismatch", C.YELLOW)

        # Phase 5 is reinforcement (separate logic)
        if cfg['type'] == 'rl':
            section("REINFORCEMENT — Self-Play vs Stockfish", "🔥")
            stat("Status", "Coming soon — run phases 0-4 first", C.YELLOW)
            break

        stat("LR", f"{cfg['lr']}")
        stat("LR schedule", lr_schedule_summary(cfg))
        stat("Batch size", str(cfg['batch_size']))
        if dist.is_initialized():
            stat("DDP effective batch", f"{cfg['batch_size']} per GPU, {cfg['batch_size'] * dist.get_world_size()} global", C.YELLOW)
        stat("Max steps", str(cfg['max_steps']))
        stat("n_think", str(cfg.get('n_think', N_THINK)))
        if adaptive_thinking_enabled(cfg):
            stat("Adaptive depths", ",".join(map(str, thinking_depths(cfg))), C.GREEN)
            stat("Aux losses", ",".join(map(str, int_list(cfg.get('aux_steps')))))
            stat("Thought regularizers",
                 f"aux={cfg.get('aux_loss_weight')} kl={cfg.get('consistency_weight')} "
                 f"value={cfg.get('value_consistency_weight')} conf={cfg.get('confidence_reg_weight')} "
                 f"dropout={cfg.get('depth_dropout')}")
            stat("World model aux",
                 f"cosine {cfg.get('world_model_start_weight')}->{cfg.get('world_model_weight')} "
                 f"n_think={cfg.get('world_n_think')} max_batch={cfg.get('world_model_max_batch')}",
                 C.GREEN if float(cfg.get('world_model_weight', 0.0) or 0.0) > 0 else C.WHITE)
            if float(cfg.get('distance_weight', 0.0) or 0.0) > 0:
                stat("Terminal distance aux",
                     f"cosine {cfg.get('distance_start_weight')}->{cfg.get('distance_weight')} "
                     f"n_think={cfg.get('distance_n_think')} max_batch={cfg.get('distance_model_max_batch')}",
                     C.GREEN)
            if float(cfg.get('policy_pressure_weight', 0.0) or 0.0) > 0:
                stat("Chess constitution",
                     f"cosine {cfg.get('policy_pressure_start_weight')}->{cfg.get('policy_pressure_weight')} "
                     f"win>{cfg.get('policy_pressure_win_threshold')} "
                     f"survive={cfg.get('policy_pressure_survival_weight')} "
                     f"anti-collapse={cfg.get('policy_pressure_loss_downweight')}",
                     C.GREEN)
            if float(cfg.get('legality_weight', 0.0) or 0.0) > 0:
                stat("Legality aux",
                     f"weight={cfg.get('legality_weight')} n_think={cfg.get('legality_n_think')} "
                     f"max_batch={cfg.get('legality_model_max_batch')}",
                     C.GREEN)
            stat("Final sharpening",
                 f"last {int(cfg.get('sharpen_steps', 0)):,} steps: depth {cfg.get('sharpen_n_think', cfg.get('n_think', N_THINK))}, "
                 f"aux={cfg.get('sharpen_aux_loss_weight')} kl={cfg.get('sharpen_consistency_weight')}",
                 C.GREEN)
        if hasattr(base_model(model), 'ghost_gate_logit'):
            gate = torch.sigmoid(base_model(model).ghost_gate_logit.detach()).item()
            stat("Ghost gate", f"{gate:.3f}", C.GREEN if gate < 0.25 else C.YELLOW)
        stat("Time limit", f"{cfg['max_minutes']} min")
        if cfg['type'] == 'gm':
            stat("GM cache", "enabled" if use_gm_cache else "disabled", C.GREEN if use_gm_cache else C.WHITE)
            if use_gm_cache:
                stat("Main cache samples", f"{int(gm_cache_samples):,}")
                stat("Main cache dir", gm_cache_dir or GM_CACHE_DIR)
                if enable_replay and cfg.get('replay', {}).get('gm', 0):
                    stat("Replay GM cache samples", f"{int(replay_gm_cache_samples):,}")
                    stat("Replay GM cache dir", replay_gm_cache_dir or REPLAY_GM_CACHE_DIR)
                if enable_replay and cfg.get('replay', {}).get('low_elo', 0):
                    stat("Low-Elo replay cache samples", f"{int(low_elo_cache_samples):,}")
                    stat("Low-Elo replay cache dir", replay_gm_cache_dir or REPLAY_GM_CACHE_DIR)
            stat("Replay mix", replay_mix_summary(cfg['batch_size'], cfg.get('replay', {}) if enable_replay else {}))

        # Train — puzzle or GM stream
        if cfg['type'] == 'puzzle':
            final_step, final_ema = train_puzzle_phase(model, optimizer, cfg)
            if is_main():
                heldout = evaluate_puzzle_heldout(base_model(model), cfg)
                final_ema = dict(final_ema)
                final_ema['heldout'] = heldout
                save_checkpoint(
                    model, optimizer, phase_num, final_step, final_ema,
                    lr=scheduled_lr(cfg, min(final_step, cfg['max_steps'] - 1)),
                    lr_schedule=lr_schedule_config(cfg),
                )
            ddp_barrier()
        else:
            streamer = make_phase_streamer(
                phase_num, cfg,
                use_gm_cache=use_gm_cache,
                gm_cache_samples=gm_cache_samples,
                replay_gm_cache_samples=replay_gm_cache_samples,
                low_elo_cache_samples=low_elo_cache_samples,
                gm_cache_dir=gm_cache_dir,
                replay_gm_cache_dir=replay_gm_cache_dir,
                enable_replay=enable_replay,
            )
            final_step, final_ema = train_phase(phase_num, model, optimizer, streamer, cfg)
            del streamer
            if use_gm_cache:
                cache_paths = gm_cache_paths(
                    phase_num, cfg,
                    max_samples=gm_cache_samples,
                    cache_dir=gm_cache_dir,
                )
                if is_main():
                    heldout = evaluate_gm_heldout(base_model(model), phase_num, cfg, cache_paths)
                    if heldout is not None:
                        final_ema = dict(final_ema)
                        final_ema['heldout'] = heldout
                        save_checkpoint(
                            model, optimizer, phase_num, final_step, final_ema,
                            lr=scheduled_lr(cfg, min(final_step, cfg['max_steps'] - 1)),
                            lr_schedule=lr_schedule_config(cfg),
                        )
                ddp_barrier()
            else:
                stat("GM held-out skipped", "enable --use-gm-cache for held-out split", C.YELLOW)

        # Phase summary
        if is_main():
            print(f"\n{C.CYAN}{'─'*64}{C.RESET}")
            print(f"  {C.BOLD}📋 PHASE {phase_num} COMPLETE{C.RESET}")
            print(f"{C.CYAN}{'─'*64}{C.RESET}")
            stat("Steps completed", str(final_step))
            if final_ema.get('acc'):
                stat("Final accuracy", f"{final_ema['acc']:.1f}%",
                     C.GREEN if final_ema['acc'] > 25 else C.YELLOW)
            if final_ema.get('pol'):
                stat("Final policy loss", f"{final_ema['pol']:.3f}")
            if final_ema.get('val'):
                stat("Final value loss", f"{final_ema['val']:.3f}")
            if final_ema.get('heldout'):
                heldout = final_ema['heldout']
                stat("Held-out full move acc", f"{heldout['full_move_acc']:.2f}%", C.GREEN)
                if 'loss' in heldout:
                    stat("Held-out policy loss", f"{heldout['loss']:.4f}")
                else:
                    stat("Held-out policy loss", f"{heldout['policy_loss']:.4f}")

            # Benchmark after each phase
            section("POST-PHASE BENCHMARK", "📊")
            safe_benchmark_vs_stockfish(model, [1500, 1800, 2000, 2200], n_games=20)
            print(f"{C.CYAN}{'─'*64}{C.RESET}")
        ddp_barrier()

        gc.collect()

    if stop_after_phase is not None and stop_after_phase < 5:
        if is_main():
            banner(f"STOPPED AFTER PHASE {end_phase}", C.GREEN)
            stat("Latest checkpoint", os.path.join(CKPT_DIR, "chess_god_v2_latest.pt"), C.GREEN)
            stat("Phase checkpoint", ckpt_path(end_phase), C.GREEN)
        ddp_barrier()
        return

    # Final
    if is_main():
        banner("🏁 CURRICULUM COMPLETE!", C.GREEN)
        stat("Model", os.path.join(CKPT_DIR, "chess_god_v2_latest.pt"))

        # Also save a clean "deploy" copy
        deploy_path = os.path.join(SCRIPT_DIR, "chess_god_v2.pt")
        torch.save({
            'model': model_state_dict(model),
            'config': checkpoint_config(),
        }, deploy_path)
        stat("Deploy checkpoint", deploy_path, C.GREEN)

        # Final benchmark
        section("FINAL BENCHMARK", "🏆")
        safe_benchmark_vs_stockfish(model, [1800, 2000, 2200, 2400], n_games=30)
    ddp_barrier()


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chess God V2 Curriculum Training")
    parser.add_argument("--phase", type=int, default=0, help="Start from phase N (0-5)")
    parser.add_argument("--stop-after-phase", type=int, default=None, help="Stop after completing phase N instead of continuing curriculum")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and retrain from scratch")
    parser.add_argument("--all-steps", action="store_true", help="Run phases to max_steps instead of stopping at max_minutes")
    parser.add_argument("--phase-max-steps", type=int, default=None, help="Override max_steps for the phase range being run")
    parser.add_argument("--force-phase", action="store_true", help="Load the requested phase checkpoint instead of auto-resuming latest")
    parser.add_argument("--init-from-phase", type=int, default=None, help="Initialize model from phase N checkpoint, then train requested phase from step 0")
    parser.add_argument("--ignore-phase-checkpoint", action="store_true", help="Do not resume the target phase checkpoint")
    parser.add_argument("--phase0-batch", type=int, default=None, help="Override Phase 0 puzzle batch size")
    parser.add_argument("--phase12-batch", type=int, default=None, help="Override Phase 1/2 batch size (try 768, then 1024 if VRAM allows)")
    parser.add_argument("--gm-batch", type=int, default=None, help="Override all GM phase batch sizes (phases 1-4)")
    parser.add_argument("--use-gm-cache", action="store_true", help="Cache GM phase samples to disk and train from the cache")
    parser.add_argument("--gm-cache-samples", type=int, default=1_000_000, help="Samples per GM phase cache when --use-gm-cache is enabled")
    parser.add_argument("--replay-gm-cache-samples", type=int, default=2_000_000, help="Samples per reusable prior-phase GM replay cache")
    parser.add_argument("--low-elo-cache-samples", type=int, default=2_000_000, help="Samples for low-Elo GM replay cache")
    parser.add_argument("--gm-cache-dir", default=None, help="Directory for GM cache files")
    parser.add_argument("--replay-gm-cache-dir", default=None, help="Directory for reusable replay GM caches; defaults to ./gm_replay_cache_v2")
    parser.add_argument("--build-cache-only", action="store_true", help="Build main/replay caches for the requested phase range, then exit without training")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers for puzzle/cache loading")
    parser.add_argument("--no-replay", action="store_true", help="Disable curriculum replay mixing in GM phases")
    parser.add_argument("--no-adaptive-thinking", action="store_true", help="Disable variable-depth/intermediate recurrent supervision")
    parser.add_argument("--think-depths", default=None, help="Comma-separated train depths, e.g. 2,4,6,8,12,16")
    parser.add_argument("--aux-loss-weight", type=float, default=None, help="Override adaptive intermediate-head loss weight")
    parser.add_argument("--consistency-weight", type=float, default=None, help="Override policy KL consistency weight")
    parser.add_argument("--value-consistency-weight", type=float, default=None, help="Override value consistency weight")
    parser.add_argument("--confidence-reg-weight", type=float, default=None, help="Override target-confidence monotonic regularizer weight")
    parser.add_argument("--depth-dropout", type=float, default=None, help="Override probability of sampling only early depths")
    parser.add_argument("--world-model-weight", type=float, default=None, help="Override final board-transition auxiliary loss weight")
    parser.add_argument("--world-model-start-weight", type=float, default=None, help="Override starting board-transition auxiliary loss weight")
    parser.add_argument("--world-model-max-batch", type=int, default=None, help="Max samples used for world-model auxiliary per step")
    parser.add_argument("--world-n-think", type=int, default=None, help="Override recurrent steps for board-transition auxiliary")
    parser.add_argument("--distance-weight", type=float, default=None, help="Override final terminal-distance auxiliary loss weight")
    parser.add_argument("--distance-start-weight", type=float, default=None, help="Override starting terminal-distance auxiliary loss weight")
    parser.add_argument("--distance-model-max-batch", type=int, default=None, help="Max samples used for terminal-distance auxiliary per step")
    parser.add_argument("--distance-n-think", type=int, default=None, help="Override recurrent steps for terminal-distance auxiliary")
    parser.add_argument("--policy-pressure-weight", type=float, default=None, help="Override final inbuilt conversion-pressure policy weight")
    parser.add_argument("--policy-pressure-start-weight", type=float, default=None, help="Override starting inbuilt conversion-pressure policy weight")
    parser.add_argument("--legality-weight", type=float, default=None, help="Override supervised legal-move-map auxiliary loss weight")
    parser.add_argument("--legality-model-max-batch", type=int, default=None, help="Max samples used for legality auxiliary per step")
    parser.add_argument("--legality-n-think", type=int, default=None, help="Override recurrent steps for legality auxiliary")
    parser.add_argument("--eval-phase0-only", action="store_true", help="Load Phase 0 checkpoint, run held-out puzzle validation, then exit")
    parser.add_argument("--eval-heldout-phase", type=int, default=None, help="Load phase N checkpoint, run held-out validation, then exit")
    parser.add_argument("--eval-phase0-before-start", action="store_true", help="Evaluate Phase 0 checkpoint before starting a later phase")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    args = parser.parse_args()

    try:
        apply_batch_overrides(phase0_batch_size=args.phase0_batch)
        apply_thinking_overrides(
            disable_adaptive=args.no_adaptive_thinking,
            think_depths=args.think_depths,
        )
        apply_thinking_loss_overrides(
            aux_loss_weight=args.aux_loss_weight,
            consistency_weight=args.consistency_weight,
            value_consistency_weight=args.value_consistency_weight,
            confidence_reg_weight=args.confidence_reg_weight,
            depth_dropout=args.depth_dropout,
            world_model_weight=args.world_model_weight,
            world_model_start_weight=args.world_model_start_weight,
            world_model_max_batch=args.world_model_max_batch,
            world_n_think=args.world_n_think,
            distance_weight=args.distance_weight,
            distance_start_weight=args.distance_start_weight,
            distance_model_max_batch=args.distance_model_max_batch,
            distance_n_think=args.distance_n_think,
            policy_pressure_weight=args.policy_pressure_weight,
            policy_pressure_start_weight=args.policy_pressure_start_weight,
            legality_weight=args.legality_weight,
            legality_model_max_batch=args.legality_model_max_batch,
            legality_n_think=args.legality_n_think,
        )
        apply_loader_overrides(num_workers=args.num_workers)

        if args.build_cache_only:
            build_cache_only(
                start_phase=args.phase,
                stop_after_phase=args.stop_after_phase,
                gm_cache_samples=args.gm_cache_samples,
                replay_gm_cache_samples=args.replay_gm_cache_samples,
                low_elo_cache_samples=args.low_elo_cache_samples,
                gm_cache_dir=args.gm_cache_dir,
                replay_gm_cache_dir=args.replay_gm_cache_dir,
                enable_replay=not args.no_replay,
            )
        elif args.eval_heldout_phase is not None:
            if is_main():
                phase = int(args.eval_heldout_phase)
                banner(f"🧪 PHASE {phase} HELD-OUT EVAL", C.BLUE)
                model = ChessGod(d1=D1, hidden=HIDDEN, n_encode=N_ENCODE,
                                 n_think=N_THINK, dropout=DROPOUT).to(DEVICE)
                if torch.cuda.device_count() > 1 and not dist.is_initialized():
                    model = nn.DataParallel(model)
                ckpt = load_checkpoint(phase)
                if ckpt is None:
                    print(f"    {C.RED}No Phase {phase} checkpoint found!{C.RESET}")
                    sys.exit(1)
                if not checkpoint_matches_model(ckpt):
                    print(f"    {C.RED}Checkpoint architecture does not match this curriculum config.{C.RESET}")
                    sys.exit(1)
                load_model_state(model, ckpt['model'])
                stat("Loaded", f"Phase {ckpt.get('phase', '?')}, step {ckpt.get('step', '?')}")
                if phase == 0:
                    evaluate_puzzle_heldout(model, PHASES[0])
                elif 1 <= phase <= 4:
                    cfg = dict(PHASES[phase])
                    cache_paths = gm_cache_paths(
                        phase, cfg,
                        max_samples=args.gm_cache_samples,
                        cache_dir=args.gm_cache_dir,
                    )
                    evaluate_gm_heldout(model, phase, cfg, cache_paths)
                else:
                    print(f"    {C.YELLOW}Phase {phase} has no held-out evaluator yet.{C.RESET}")
        elif args.eval_phase0_only:
            if is_main():
                banner("🧪 PHASE 0 HELD-OUT EVAL", C.BLUE)
                model = ChessGod(d1=D1, hidden=HIDDEN, n_encode=N_ENCODE,
                                 n_think=N_THINK, dropout=DROPOUT).to(DEVICE)
                if torch.cuda.device_count() > 1 and not dist.is_initialized():
                    model = nn.DataParallel(model)
                ckpt = load_checkpoint(0)
                if ckpt is None:
                    print(f"    {C.RED}No Phase 0 checkpoint found!{C.RESET}")
                    sys.exit(1)
                if not checkpoint_matches_model(ckpt):
                    print(f"    {C.RED}Checkpoint architecture does not match this curriculum config.{C.RESET}")
                    sys.exit(1)
                load_model_state(model, ckpt['model'])
                stat("Loaded", f"Phase {ckpt.get('phase', '?')}, step {ckpt.get('step', '?')}")
                evaluate_puzzle_heldout(model, PHASES[0])
        elif args.benchmark:
            if is_main():
                banner("♟️ BENCHMARK MODE", C.BLUE)
                model = ChessGod(d1=D1, hidden=HIDDEN, n_encode=N_ENCODE, n_think=N_THINK).to(DEVICE)
                ckpt = load_checkpoint()
                if ckpt:
                    if not checkpoint_matches_model(ckpt):
                        print(f"    {C.RED}Checkpoint architecture does not match this curriculum config.{C.RESET}")
                        sys.exit(1)
                    load_model_state(model, ckpt['model'])
                    stat("Loaded", f"Phase {ckpt.get('phase', '?')}")
                else:
                    print(f"    {C.RED}No checkpoint found!{C.RESET}")
                    sys.exit(1)
                model.eval()
                safe_benchmark_vs_stockfish(model, [1500, 1800, 2000, 2200, 2400], n_games=50)
        else:
            run_curriculum(
                start_phase=args.phase,
                fresh=args.fresh,
                all_steps=args.all_steps,
                force_phase=args.force_phase,
                use_gm_cache=args.use_gm_cache,
                gm_cache_samples=args.gm_cache_samples,
                replay_gm_cache_samples=args.replay_gm_cache_samples,
                low_elo_cache_samples=args.low_elo_cache_samples,
                gm_cache_dir=args.gm_cache_dir,
                replay_gm_cache_dir=args.replay_gm_cache_dir,
                gm_batch_size=args.gm_batch,
                phase12_batch_size=args.phase12_batch,
                phase0_batch_size=args.phase0_batch,
                num_workers=args.num_workers,
                enable_replay=not args.no_replay,
                eval_phase0_before_start=args.eval_phase0_before_start,
                disable_adaptive_thinking=args.no_adaptive_thinking,
                think_depths=args.think_depths,
                stop_after_phase=args.stop_after_phase,
                phase_max_steps=args.phase_max_steps,
                aux_loss_weight=args.aux_loss_weight,
                consistency_weight=args.consistency_weight,
                value_consistency_weight=args.value_consistency_weight,
                confidence_reg_weight=args.confidence_reg_weight,
                depth_dropout=args.depth_dropout,
                world_model_weight=args.world_model_weight,
                world_model_start_weight=args.world_model_start_weight,
                world_model_max_batch=args.world_model_max_batch,
                world_n_think=args.world_n_think,
                distance_weight=args.distance_weight,
                distance_start_weight=args.distance_start_weight,
                distance_model_max_batch=args.distance_model_max_batch,
                distance_n_think=args.distance_n_think,
                policy_pressure_weight=args.policy_pressure_weight,
                policy_pressure_start_weight=args.policy_pressure_start_weight,
                legality_weight=args.legality_weight,
                legality_model_max_batch=args.legality_model_max_batch,
                legality_n_think=args.legality_n_think,
                init_from_phase=args.init_from_phase,
                ignore_phase_checkpoint=args.ignore_phase_checkpoint,
            )
    finally:
        cleanup_ddp()
