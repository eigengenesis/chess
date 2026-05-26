#!/usr/bin/env python3
"""
VOID chess architecture.

Spatial board tensor -> geometric encoder -> recurrent latent thinking ->
policy, value, legality, distance, and action-conditioned world-model heads.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess


N_PLANES = 14
D1 = 192
HIDDEN = 512
N_ENCODE = 5
N_THINK = 16
DROPOUT = 0.05

# ============================================================================
# BOARD ENCODING (18 planes — upgraded from 14)
# ============================================================================
PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode board as [18, 8, 8] tensor — BITBOARD-OPTIMIZED.
    Uses python-chess bitboard API instead of per-square loops.
    ~5-10x faster than the naive version.
    """
    import numpy as np
    planes = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    is_white = board.turn == chess.WHITE

    # Planes 0-11: Pieces via bitboard iteration (only visits occupied squares)
    for piece_type, plane_idx in PIECE_TO_PLANE.items():
        for sq in board.pieces(piece_type, board.turn):
            s = sq if is_white else sq ^ 56
            planes[plane_idx, s >> 3, s & 7] = 1.0
        for sq in board.pieces(piece_type, not board.turn):
            s = sq if is_white else sq ^ 56
            planes[6 + plane_idx, s >> 3, s & 7] = 1.0

    # Plane 12: Turn indicator
    if is_white:
        planes[12] = 1.0

    # Plane 13: Castling rights
    if is_white:
        if board.has_kingside_castling_rights(chess.WHITE):  planes[13, 0, 7] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): planes[13, 0, 0] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):  planes[13, 7, 7] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): planes[13, 7, 0] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.BLACK):  planes[13, 0, 7] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): planes[13, 0, 0] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):  planes[13, 7, 7] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): planes[13, 7, 0] = 1.0

    # Plane 14: En Passant (only if N_PLANES > 14)
    if N_PLANES > 14:
        ep = board.ep_square
        if ep is not None:
            s = ep if is_white else ep ^ 56
            planes[14, s >> 3, s & 7] = 1.0

    # Plane 15: Legal FROM squares (only if N_PLANES > 15)
    if N_PLANES > 15:
        seen_from = set()
        for move in board.legal_moves:
            f = move.from_square
            if f not in seen_from:
                seen_from.add(f)
                s = f if is_white else f ^ 56
                planes[15, s >> 3, s & 7] = 1.0

    # Plane 16: Attack map (only if N_PLANES > 16)
    if N_PLANES > 16:
        for piece_type in chess.PIECE_TYPES:
            for sq in board.pieces(piece_type, board.turn):
                for target in board.attacks(sq):
                    s = target if is_white else target ^ 56
                    planes[16, s >> 3, s & 7] = 1.0

    # Plane 17: Halfmove clock (only if N_PLANES > 17)
    if N_PLANES > 17:
        planes[17] = min(board.halfmove_clock / 100.0, 1.0)

    return torch.from_numpy(planes)


def move_to_indices(move: chess.Move, board: chess.Board):
    """Convert move to (from_sq, to_sq). Flip for black."""
    f, t = move.from_square, move.to_square
    if board.turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f, t


# ============================================================================
# 4D CHESS BLOCK (9-Direction Shift + Contract)
# ============================================================================
class ChessBlock(nn.Module):
    """
    Shift + Contract + Expand block for chess.
    Looks in 9 directions (center + 8 neighbors) — naturally models
    how pieces see the board. Stacked blocks compose into long-range
    patterns (rook files, bishop diags, knight L-shapes).
    """
    def __init__(self, d1, hidden, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d1)
        # 9 directional contractions
        self.contractions = nn.ModuleList([
            nn.Linear(d1, hidden, bias=False) for _ in range(9)
        ])
        self.expand = nn.Linear(9 * hidden, d1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)  # Anti-memorization

        # 9 shift offsets: center, 4 cardinal, 4 diagonal
        self.shifts = [(0, 0),
                       (0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # Stability initialization
        for m in self.contractions:
            nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.trunc_normal_(self.expand.weight, std=0.01)

    def forward(self, x):
        # x: [B, d1, 8, 8]
        residual = x
        
        # TURBO OPTIMIZATION: Normalize once before shifting
        # Since shifts are spatial and Norm is over channels, 
        # roll(Norm(x)) effectively equals Norm(roll(x)) but is 9x faster.
        x_normed = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        contracted = []
        for i, (dy, dx) in enumerate(self.shifts):
            if dy == 0 and dx == 0:
                # Center: use the normed permuted view
                view = x_normed.permute(0, 2, 3, 1)
            else:
                # Shift: roll the normed tensor
                shifted = torch.roll(x_normed, shifts=(dy, dx), dims=(2, 3))
                view = shifted.permute(0, 2, 3, 1)
            
            contracted.append(self.contractions[i](view))

        fused = torch.cat(contracted, dim=-1)  # [B, 8, 8, 9*hidden]
        out = self.expand(self.act(fused))      # [B, 8, 8, d1]
        out = self.drop(out)                    
        # Residual scaling
        return residual + 0.1 * out.permute(0, 3, 1, 2)


# ============================================================================
# PURE GEOMETRIC PROJECTIONS (No Conv2d — only einsum)
# ============================================================================
class PointwiseProject(nn.Module):
    """1×1 projection via einsum. Pure replacement for Conv2d(k=1)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        # x: [B, C_in, H, W] → [B, C_out, H, W]
        return torch.einsum('bchw,dc->bdhw', x, self.weight)


class SpatialMix(nn.Module):
    """3×3 spatial mixing via 9 shifts + einsum.
    Pure geometric equivalent of Conv2d(k=3, padding=1).
    Each direction has its own learned contraction matrix.
    """
    def __init__(self, d):
        super().__init__()
        self.shifts = [(0,0), (0,1), (0,-1), (1,0), (-1,0),
                       (1,1), (1,-1), (-1,1), (-1,-1)]
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(d, d)) for _ in range(9)
        ])
        for w in self.weights:
            nn.init.trunc_normal_(w, std=0.02 / 3)

    def forward(self, x):
        # x: [B, C, H, W]
        out = torch.zeros_like(x)
        for i, (dy, dx) in enumerate(self.shifts):
            shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3)) if (dy or dx) else x
            out = out + torch.einsum('bchw,dc->bdhw', shifted, self.weights[i])
        return out


# ============================================================================
# CHESS GOD MODEL
# ============================================================================
class ChessGod(nn.Module):
    """
    Board [14,8,8] -> Encoder -> Recurrent Ghost -> Policy + Value

    The Ghost block has SHARED weights and runs N times.
    More thinking steps at inference = deeper calculation.
    This IS the "System 2" — active reasoning during inference.
    100% Pure Tensor Contractions — ZERO Conv2d.
    """
    def __init__(self, d1=D1, hidden=HIDDEN, n_encode=N_ENCODE,
                 n_think=N_THINK, dropout=0.0):
        super().__init__()
        self.n_think = n_think

        # Encoder (Pure geometric projections — NO Conv2d)
        self.proj_pointwise = PointwiseProject(N_PLANES, d1)
        self.proj_spatial = SpatialMix(d1)
        self.encode_blocks = nn.ModuleList([
            ChessBlock(d1, hidden, dropout=dropout) for _ in range(n_encode)
        ])

        # Recurrent Ghost (Single shared-weight block)
        self.ghost = ChessBlock(d1, hidden, dropout=dropout)
        # Small thought gate: sigmoid(-2.1972) ~= 0.10. This lets the
        # recurrent loop refine the spatial state instead of overwriting it.
        self.ghost_gate_logit = nn.Parameter(torch.tensor(-2.1972))

        # Policy: from_sq [B,64] + to_sq [B,64,64]
        self.policy_norm = nn.LayerNorm(d1)
        self.from_head = nn.Linear(d1, 1, bias=False)
        self.to_head = nn.Linear(d1, 64, bias=False)
        self.legality_norm = nn.LayerNorm(d1)
        self.legality_head = nn.Linear(d1, 64, bias=True)
        nn.init.trunc_normal_(self.legality_head.weight, std=0.01)
        nn.init.constant_(self.legality_head.bias, 0.0)

        # Value: win probability [-1, 1]
        self.value_norm = nn.LayerNorm(d1)
        self.value_head = nn.Sequential(
            nn.Linear(d1, 64, bias=False), nn.GELU(),
            nn.Linear(64, 1, bias=False), nn.Tanh(),
        )
        # Terminal distance: normalized "how soon does this game resolve?"
        # Used as an endgame auxiliary and inbuilt conversion-pressure signal.
        self.distance_norm = nn.LayerNorm(d1)
        self.distance_head = nn.Sequential(
            nn.Linear(d1, 64, bias=False), nn.GELU(),
            nn.Linear(64, 1, bias=True),
        )
        # Learned conversion pressure inside the policy head. This is not a
        # move-time reranker: the policy logits themselves get a tiny,
        # trainable urgency bias when the model believes it is winning and the
        # terminal distance is short.
        self.policy_pressure_norm = nn.LayerNorm(d1)
        self.pressure_from_head = nn.Linear(d1, 1, bias=False)
        self.pressure_to_head = nn.Linear(d1, 64, bias=False)
        self.policy_pressure_gain = nn.Parameter(torch.tensor(-4.5951))
        nn.init.trunc_normal_(self.pressure_from_head.weight, std=0.01)
        nn.init.trunc_normal_(self.pressure_to_head.weight, std=0.01)

        # Action-conditioned transition head. This is an auxiliary world-model
        # path: board + move -> full next board tensor. It shares the encoder
        # and ghost loop, so old checkpoints remain useful spatial backbones.
        self.move_from_embed = nn.Embedding(64, d1)
        self.move_to_embed = nn.Embedding(64, d1)
        self.transition_norm = nn.LayerNorm(d1)
        self.transition_head = nn.Linear(d1, N_PLANES, bias=True)
        nn.init.trunc_normal_(self.move_from_embed.weight, std=0.01)
        nn.init.trunc_normal_(self.move_to_embed.weight, std=0.01)
        nn.init.trunc_normal_(self.transition_head.weight, std=0.01)
        nn.init.constant_(self.transition_head.bias, 0.0)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict:
            current = self.state_dict()
            compatible = {}
            for key, value in state_dict.items():
                if key not in current or tuple(current[key].shape) == tuple(value.shape):
                    compatible[key] = value
            return super().load_state_dict(compatible, strict=False, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def encode_board(self, board):
        """Encode the board once before recurrent thinking."""
        x = F.gelu(self.proj_pointwise(board))
        x = F.gelu(self.proj_spatial(x))
        for block in self.encode_blocks:
            x = block(x)
        return x

    def ghost_step(self, x, x_init):
        """One gated recurrent thought update."""
        candidate = self.ghost(x + x_init)
        gate = torch.sigmoid(self.ghost_gate_logit)
        return x + gate * (candidate - x)

    def heads_from_state(self, x):
        """Policy/value heads from a recurrent state."""
        raw_sq = x.flatten(2).permute(0, 2, 1)
        sq = self.policy_norm(raw_sq)

        from_logits = self.from_head(sq).squeeze(-1)  # [B, 64]
        to_logits = self.to_head(sq)                  # [B, 64, 64]

        global_feat = self.value_norm(sq.mean(dim=1))
        value = self.value_head(global_feat)

        distance_feat = self.distance_norm(raw_sq.mean(dim=1))
        distance = torch.sigmoid(self.distance_head(distance_feat)).detach().squeeze(-1)
        value_signal = value.detach().squeeze(-1)
        winning = torch.sigmoid((value_signal - 0.25) * 4.0)
        losing = torch.sigmoid((-value_signal - 0.25) * 4.0)
        conversion = winning * (1.0 - distance)
        survival = losing * distance
        pressure = torch.sigmoid(self.policy_pressure_gain) * (conversion + 0.5 * survival)
        pressure_sq = self.policy_pressure_norm(sq)
        from_logits = from_logits + pressure[:, None] * self.pressure_from_head(pressure_sq).squeeze(-1)
        to_logits = to_logits + pressure[:, None, None] * self.pressure_to_head(pressure_sq)
        return from_logits, to_logits, value

    def condition_on_move(self, x, from_sq, to_sq):
        """Inject a candidate action into the board-shaped latent state."""
        bsz, channels, height, width = x.shape
        sq = x.flatten(2).permute(0, 2, 1).contiguous()
        batch_idx = torch.arange(bsz, device=x.device)
        from_embed = self.move_from_embed(from_sq).to(dtype=sq.dtype)
        to_embed = self.move_to_embed(to_sq).to(dtype=sq.dtype)
        sq[batch_idx, from_sq] = sq[batch_idx, from_sq] + from_embed
        sq[batch_idx, to_sq] = sq[batch_idx, to_sq] + to_embed
        return sq.permute(0, 2, 1).reshape(bsz, channels, height, width)

    def transition_logits(self, board, from_sq, to_sq, n_think=None):
        """Predict the full next board tensor from board + move."""
        if n_think is None:
            n_think = min(4, self.n_think)
        n_think = int(n_think)

        x = self.encode_board(board)
        x = self.condition_on_move(x, from_sq, to_sq)
        x_init = x
        for _ in range(n_think):
            x = self.ghost_step(x, x_init)

        sq = x.flatten(2).permute(0, 2, 1)
        sq = self.transition_norm(sq)
        logits = self.transition_head(sq)
        return logits.permute(0, 2, 1).reshape(board.size(0), N_PLANES, 8, 8)

    def legality_logits(self, board, n_think=None):
        """Predict legal move map [B, 64, 64] as an optional auxiliary."""
        if n_think is None:
            n_think = self.n_think
        n_think = int(n_think)

        x = self.encode_board(board)
        x_init = x
        for _ in range(n_think):
            x = self.ghost_step(x, x_init)

        sq = x.flatten(2).permute(0, 2, 1)
        sq = self.legality_norm(sq)
        return self.legality_head(sq)

    def distance_logits(self, board, n_think=None):
        """Predict normalized plies-to-terminal from the current board."""
        if n_think is None:
            n_think = self.n_think
        n_think = int(n_think)

        x = self.encode_board(board)
        x_init = x
        for _ in range(n_think):
            x = self.ghost_step(x, x_init)

        sq = x.flatten(2).permute(0, 2, 1)
        global_feat = self.distance_norm(sq.mean(dim=1))
        return self.distance_head(global_feat)

    def forward(self, board, n_think=None, return_steps=False, supervision_steps=None,
                transition_from=None, transition_to=None, distance_only=False,
                legality_only=False):
        if transition_from is not None and transition_to is not None:
            return self.transition_logits(board, transition_from, transition_to, n_think=n_think)
        if legality_only:
            return self.legality_logits(board, n_think=n_think)
        if distance_only:
            return self.distance_logits(board, n_think=n_think)

        if n_think is None:
            n_think = self.n_think
        n_think = int(n_think)

        x = self.encode_board(board)

        # Dense Geometric Routing: anchor every ghost step to the raw board state
        # This creates a gradient highway bypassing 16-step diffusion decay
        x_init = x
        wanted_steps = set()
        if return_steps:
            wanted_steps = {int(s) for s in (supervision_steps or []) if 1 <= int(s) <= n_think}
            wanted_steps.add(n_think)
        step_outputs = {}

        for step in range(1, n_think + 1):
            x = self.ghost_step(x, x_init)
            if return_steps and step in wanted_steps:
                step_outputs[step] = self.heads_from_state(x)

        final = step_outputs.get(n_think) if return_steps else None
        if final is None:
            final = self.heads_from_state(x)

        if return_steps:
            return final[0], final[1], final[2], step_outputs
        return final


__all__ = [
    "N_PLANES",
    "D1",
    "HIDDEN",
    "N_ENCODE",
    "N_THINK",
    "DROPOUT",
    "board_to_tensor",
    "move_to_indices",
    "PointwiseProject",
    "SpatialMix",
    "ChessBlock",
    "ChessGod",
]
