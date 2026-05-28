from __future__ import annotations

import io
import math
import os
import random
import secrets
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import chess
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_god_mode import (  # noqa: E402
    ChessGod,
    DEVICE,
    N_THINK_PLAY,
    board_to_tensor,
    move_to_indices,
)
from visualize_prescience import build_prescience_snapshot  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "chess_god.pt"
DEFAULT_THINK_STEPS = int(os.getenv("CHESS_GOD_THINK_STEPS", str(N_THINK_PLAY)))
DEFAULT_ROLLOUT_WIDTH = int(os.getenv("CHESS_GOD_ROLLOUT_WIDTH", "6"))
DEFAULT_ROLLOUT_PLIES = int(os.getenv("CHESS_GOD_ROLLOUT_PLIES", "3"))
ACTIVE_AMBIGUITY_WEIGHT = float(os.getenv("CHESS_GOD_AMBIGUITY_WEIGHT", "0.08"))
ACTIVE_EPISTEMIC_WEIGHT = float(os.getenv("CHESS_GOD_EPISTEMIC_WEIGHT", "0.08"))
ACTIVE_PRECISION_FLOOR = float(os.getenv("CHESS_GOD_PRECISION_FLOOR", "0.35"))

app = FastAPI(title="Chess God private inference API", version="1.0.0")

origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)


class MoveRequest(BaseModel):
    fen: str = Field(..., min_length=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_k: int = Field(1, ge=1, le=10)
    think_steps: Optional[int] = Field(None, ge=1, le=64)
    include_prescience: bool = False
    rollout_mode: str = Field("off")


class PrescienceRequest(BaseModel):
    fen: str = Field(..., min_length=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    think_steps: Optional[int] = Field(None, ge=1, le=64)
    max_candidates: int = Field(8, ge=1, le=16)
    max_trace: int = Field(8, ge=2, le=16)


class Candidate(BaseModel):
    move: str
    san: str
    score: float
    policy_score: Optional[float] = None
    probability: Optional[float] = None
    rollout_score: Optional[float] = None
    imagined_value: Optional[float] = None
    line_depth: Optional[int] = None
    line_confidence: Optional[float] = None
    energy: Optional[float] = None
    free_energy: Optional[float] = None
    ambiguity: Optional[float] = None
    epistemic_value: Optional[float] = None
    precision: Optional[float] = None
    material_delta: Optional[float] = None
    selector: Optional[str] = None


class MoveResponse(BaseModel):
    move: Optional[str]
    san: Optional[str]
    fen_before: str
    fen_after: str
    value: float
    status: str
    game_over: bool
    result: str
    selector: str = "policy"
    candidates: list[Candidate]
    prescience: Optional[dict[str, Any]] = None


@app.middleware("http")
async def require_shared_secret(request: Request, call_next):
    expected = os.getenv("API_SHARED_SECRET", "")
    protected = request.url.path.startswith("/api/") and request.url.path != "/api/health"
    if expected and protected and request.method != "OPTIONS":
        supplied = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(supplied, expected):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


def _read_checkpoint(path: Path) -> bytes:
    encrypted = os.getenv("CHESS_GOD_ENCRYPTED", "").lower() in {"1", "true", "yes"} or path.suffix == ".enc"
    raw = path.read_bytes()
    if not encrypted:
        return raw

    key = os.getenv("CHESS_GOD_KEY")
    if not key:
        raise RuntimeError("CHESS_GOD_KEY is required for encrypted checkpoints.")

    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError("Install cryptography to load encrypted checkpoints.") from exc

    return Fernet(key.encode("utf-8")).decrypt(raw)


def _map_legacy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("ghost_tactics.") for key in state):
        return state

    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("ghost_tactics."):
            mapped[key.replace("ghost_tactics.", "ghost.")] = value
        elif key.startswith("ghost_strategy."):
            continue
        else:
            mapped[key] = value
    return mapped


@lru_cache(maxsize=1)
def load_model() -> ChessGod:
    checkpoint_path = Path(os.getenv("CHESS_GOD_CHECKPOINT", str(DEFAULT_CHECKPOINT))).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if os.getenv("TORCH_NUM_THREADS"):
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

    checkpoint = torch.load(io.BytesIO(_read_checkpoint(checkpoint_path)), map_location=DEVICE, weights_only=False)
    cfg = checkpoint["config"]
    model = ChessGod(
        d1=cfg["d1"],
        hidden=cfg["hidden"],
        n_encode=cfg["n_encode"],
        n_think=DEFAULT_THINK_STEPS,
    ).to(DEVICE)
    model.load_state_dict(_map_legacy_state(checkpoint["model"]), strict=False)
    model.eval()
    return model


def _status(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate, {winner} wins"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Draw by insufficient material"
    if board.can_claim_fifty_moves():
        return "Draw can be claimed by fifty-move rule"
    if board.can_claim_threefold_repetition():
        return "Draw can be claimed by repetition"
    turn = "White" if board.turn == chess.WHITE else "Black"
    return f"{turn} to move" + (" in check" if board.is_check() else "")


def _score_moves(model: ChessGod, board: chess.Board, temperature: float, think_steps: int) -> tuple[float, list[dict]]:
    board_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    softmax_temp = max(temperature, 0.05) if temperature else 1.0

    with torch.no_grad():
        from_logits, to_logits, value = model(board_tensor, n_think=think_steps)

    from_probs = torch.softmax(from_logits[0] / softmax_temp, dim=0)
    to_probs_all = torch.softmax(to_logits[0] / softmax_temp, dim=-1)
    scored = []
    for move in board.legal_moves:
        from_square, to_square = move_to_indices(move, board)
        score = float((from_probs[from_square] * to_probs_all[from_square, to_square]).item())
        scored.append({"move": move, "score": score, "policy_score": score})

    total = sum(item["score"] for item in scored)
    fallback_probability = 1.0 / max(1, len(scored))
    for item in scored:
        probability = item["score"] / total if total > 0 else fallback_probability
        item["probability"] = probability
        item["policy_logprob"] = math.log(max(probability, 1e-9))
    scored.sort(key=lambda item: item["score"], reverse=True)
    return float(value[0].item()), scored


def _candidate_payload(item: dict, board: chess.Board) -> Candidate:
    move = item["move"]
    return Candidate(
        move=move.uci(),
        san=board.san(move),
        score=float(item.get("score", 0.0)),
        policy_score=float(item.get("policy_score", item.get("score", 0.0))),
        probability=float(item.get("probability", 0.0)),
        rollout_score=None if item.get("rollout_score") is None else float(item["rollout_score"]),
        imagined_value=None if item.get("imagined_value") is None else float(item["imagined_value"]),
        line_depth=None if item.get("line_depth") is None else int(item["line_depth"]),
        line_confidence=None if item.get("line_confidence") is None else float(item["line_confidence"]),
        energy=None if item.get("energy") is None else float(item["energy"]),
        free_energy=None if item.get("free_energy") is None else float(item["free_energy"]),
        ambiguity=None if item.get("ambiguity") is None else float(item["ambiguity"]),
        epistemic_value=None if item.get("epistemic_value") is None else float(item["epistemic_value"]),
        precision=None if item.get("precision") is None else float(item["precision"]),
        material_delta=None if item.get("material_delta") is None else float(item["material_delta"]),
        selector=item.get("selector"),
    )


def _prescience_candidate_payloads(scored: list[dict], board: chess.Board, limit: int = 8) -> list[dict[str, Any]]:
    payloads = []
    for item in scored[:limit]:
        rollout_score = item.get("rollout_score")
        imagined_value = item.get("imagined_value")
        line_confidence = item.get("line_confidence")
        energy = item.get("energy")
        material_delta = item.get("material_delta")
        free_energy = item.get("free_energy")
        ambiguity = item.get("ambiguity")
        epistemic_value = item.get("epistemic_value")
        precision = item.get("precision")
        payloads.append({
            "move": item["move"].uci(),
            "san": board.san(item["move"]),
            "from": chess.square_name(item["move"].from_square),
            "to": chess.square_name(item["move"].to_square),
            "score": round(float(item.get("score", 0.0)), 8),
            "probability": round(float(item.get("probability", 0.0)), 4),
            "policy_score": round(float(item.get("policy_score", item.get("score", 0.0))), 8),
            "rollout_score": None if rollout_score is None else round(float(rollout_score), 8),
            "imagined_value": None if imagined_value is None else round(float(imagined_value), 4),
            "line_depth": item.get("line_depth"),
            "line_confidence": None if line_confidence is None else round(float(line_confidence), 4),
            "energy": None if energy is None else round(float(energy), 4),
            "free_energy": None if free_energy is None else round(float(free_energy), 4),
            "ambiguity": None if ambiguity is None else round(float(ambiguity), 4),
            "epistemic_value": None if epistemic_value is None else round(float(epistemic_value), 4),
            "precision": None if precision is None else round(float(precision), 4),
            "material_delta": None if material_delta is None else round(float(material_delta), 2),
            "selector": item.get("selector", "policy"),
        })
    return payloads


def _rollout_enabled(mode: str) -> bool:
    return str(mode or "off").lower() in {"on", "rollout", "world", "latent", "line", "3ply", "1", "one", "single", "shallow"}


def _rollout_depth(mode: str) -> int:
    lowered = str(mode or "").lower()
    if lowered in {"1", "one", "single", "shallow"}:
        return 1
    return max(1, min(3, DEFAULT_ROLLOUT_PLIES))


def _latent_policy_choice(from_logits: torch.Tensor, to_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from_probs = torch.softmax(from_logits.float(), dim=-1)
    from_sq = from_probs.argmax(dim=-1)
    row_idx = torch.arange(from_logits.size(0), device=from_logits.device)
    selected_to_logits = to_logits.float()[row_idx, from_sq]
    to_probs = torch.softmax(selected_to_logits, dim=-1)
    to_sq = to_probs.argmax(dim=-1)
    confidence = torch.sqrt(torch.clamp(from_probs[row_idx, from_sq] * to_probs[row_idx, to_sq], min=0.0, max=1.0))
    return from_sq.long(), to_sq.long(), confidence


def _policy_uncertainty(from_logits: torch.Tensor, to_logits: torch.Tensor, from_sq: torch.Tensor) -> torch.Tensor:
    row_idx = torch.arange(from_logits.size(0), device=from_logits.device)
    from_probs = torch.softmax(from_logits.float(), dim=-1).clamp(1e-8, 1.0)
    selected_to_probs = torch.softmax(to_logits.float()[row_idx, from_sq.long()], dim=-1).clamp(1e-8, 1.0)
    from_entropy = -(from_probs * from_probs.log()).sum(dim=-1)
    to_entropy = -(selected_to_probs * selected_to_probs.log()).sum(dim=-1)
    return torch.clamp((from_entropy + to_entropy) / (2.0 * math.log(64.0)), min=0.0, max=1.0)


def _board_ambiguity(board_tensor: torch.Tensor) -> torch.Tensor:
    probability = board_tensor.float().clamp(1e-5, 1.0 - 1e-5)
    entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())
    normalized = entropy / math.log(2.0)
    return torch.clamp(normalized.flatten(1).mean(dim=1), min=0.0, max=1.0)


def _distance_energy(value: float, terminal_distance: float) -> float:
    winning = max(0.0, min(1.0, (value - 0.25) / 0.75))
    losing = max(0.0, min(1.0, (-value - 0.25) / 0.75))
    convert_soon = winning * terminal_distance
    survive_longer = 0.5 * losing * (1.0 - terminal_distance)
    return convert_soon + survive_longer


def _material_for(board: chess.Board, color: chess.Color) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    total = 0
    for piece in board.piece_map().values():
        value = values[piece.piece_type]
        total += value if piece.color == color else -value
    return total


def _score_world_rollout(
    model: ChessGod,
    board: chess.Board,
    scored: list[dict],
    think_steps: int,
    rollout_width: int = DEFAULT_ROLLOUT_WIDTH,
    transition_steps: int = 2,
    line_depth: int = 3,
) -> list[dict]:
    if not scored:
        return scored

    for item in scored:
        item.setdefault("policy_score", item.get("score", 0.0))

    line_depth = max(1, min(3, int(line_depth)))
    candidates = scored[:max(1, min(rollout_width, len(scored)))]
    for item in candidates:
        probe = board.copy(stack=False)
        probe.push(item["move"])
        item["material_delta"] = float(_material_for(probe, board.turn) - _material_for(board, board.turn))
        item["gives_check"] = probe.is_check()
        if probe.is_checkmate():
            item["rollout_score"] = 2.0
            item["imagined_value"] = 1.0
            item["line_depth"] = 1
            item["line_confidence"] = 1.0
            item["energy"] = 0.0
            item["free_energy"] = 0.0
            item["ambiguity"] = 0.0
            item["epistemic_value"] = 1.0
            item["precision"] = 1.0
            item["material_delta"] = 99.0
            item["selector"] = "world_rollout"
            item["score"] = item["rollout_score"]
            return sorted(scored, key=lambda entry: entry.get("rollout_score", entry["score"]), reverse=True)

    board_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    batch = board_tensor.repeat(len(candidates), 1, 1, 1)
    from_sq, to_sq = zip(*(move_to_indices(item["move"], board) for item in candidates))
    from_tensor = torch.tensor(from_sq, device=DEVICE, dtype=torch.long)
    to_tensor = torch.tensor(to_sq, device=DEVICE, dtype=torch.long)

    with torch.inference_mode():
        imagined_logits = model(
            batch,
            transition_from=from_tensor,
            transition_to=to_tensor,
            n_think=transition_steps,
        )
        imagined_boards = torch.sigmoid(imagined_logits).to(dtype=batch.dtype)
        first_board_ambiguity = _board_ambiguity(imagined_boards)
        imagined_from, imagined_to, imagined_value = model(imagined_boards, n_think=think_steps)
        opponent_from, opponent_to, opponent_conf = _latent_policy_choice(imagined_from, imagined_to)
        opponent_uncertainty = _policy_uncertainty(imagined_from, imagined_to, opponent_from)
        after_our_move_value = -imagined_value.squeeze(-1).float()
        final_boards = imagined_boards
        final_board_ambiguity = first_board_ambiguity
        current_side_value = after_our_move_value
        line_confidence = opponent_conf
        rollout_ambiguity = 0.55 * first_board_ambiguity + 0.45 * opponent_uncertainty
        policy_epistemic = torch.zeros_like(first_board_ambiguity)

        if line_depth >= 3:
            after_reply_logits = model(
                imagined_boards,
                transition_from=opponent_from,
                transition_to=opponent_to,
                n_think=transition_steps,
            )
            after_reply_boards = torch.sigmoid(after_reply_logits).to(dtype=batch.dtype)
            after_reply_ambiguity = _board_ambiguity(after_reply_boards)
            follow_from_logits, follow_to_logits, after_reply_value = model(after_reply_boards, n_think=think_steps)
            follow_from, follow_to, follow_conf = _latent_policy_choice(follow_from_logits, follow_to_logits)
            follow_uncertainty = _policy_uncertainty(follow_from_logits, follow_to_logits, follow_from)

            after_follow_logits = model(
                after_reply_boards,
                transition_from=follow_from,
                transition_to=follow_to,
                n_think=transition_steps,
            )
            after_follow_boards = torch.sigmoid(after_follow_logits).to(dtype=batch.dtype)
            after_follow_ambiguity = _board_ambiguity(after_follow_boards)
            next_from_logits, next_to_logits, after_follow_value = model(after_follow_boards, n_think=think_steps)
            next_from, _, next_conf = _latent_policy_choice(next_from_logits, next_to_logits)
            next_uncertainty = _policy_uncertainty(next_from_logits, next_to_logits, next_from)
            final_boards = after_follow_boards
            final_board_ambiguity = after_follow_ambiguity

            after_reply_current_value = after_reply_value.squeeze(-1).float()
            after_follow_current_value = -after_follow_value.squeeze(-1).float()
            current_side_value = (
                0.20 * after_our_move_value
                + 0.30 * after_reply_current_value
                + 0.50 * after_follow_current_value
            )
            line_confidence = torch.clamp(opponent_conf * follow_conf * next_conf, min=0.0, max=1.0).pow(1.0 / 3.0)
            policy_uncertainty = 0.30 * opponent_uncertainty + 0.35 * follow_uncertainty + 0.35 * next_uncertainty
            rollout_ambiguity = (
                0.25 * first_board_ambiguity
                + 0.25 * after_reply_ambiguity
                + 0.30 * after_follow_ambiguity
                + 0.20 * policy_uncertainty
            )
            policy_epistemic = torch.clamp(opponent_uncertainty - next_uncertainty, min=0.0, max=1.0)

        terminal_distance = torch.sigmoid(model(final_boards, n_think=think_steps, distance_only=True).squeeze(-1).float())
        epistemic_value = torch.clamp(
            0.65 * torch.clamp(first_board_ambiguity - final_board_ambiguity, min=0.0, max=1.0)
            + 0.35 * policy_epistemic,
            min=0.0,
            max=1.0,
        )
        precision = torch.clamp(
            ACTIVE_PRECISION_FLOOR + (1.0 - ACTIVE_PRECISION_FLOOR) * line_confidence * (1.0 - rollout_ambiguity),
            min=ACTIVE_PRECISION_FLOOR,
            max=1.0,
        )

    for index, item in enumerate(candidates):
        policy_probability = float(item.get("probability", 0.0))
        value = float(current_side_value[index].item())
        confidence_term = float(line_confidence[index].item())
        distance_term = float(terminal_distance[index].item())
        ambiguity_term = float(rollout_ambiguity[index].item())
        epistemic_term = float(epistemic_value[index].item())
        precision_term = float(precision[index].item())
        policy_energy = -math.log(max(policy_probability, 1e-9)) / abs(math.log(1e-9))
        value_energy = (1.0 - value) * 0.5
        uncertainty_energy = 1.0 - confidence_term
        distance_pressure = _distance_energy(value, distance_term)
        material_energy = -max(-1.0, min(1.0, float(item.get("material_delta", 0.0)) / 9.0))
        check_energy = -0.04 if item.get("gives_check") else 0.0
        pragmatic_energy = (
            0.30 * value_energy
            + 0.05 * distance_pressure
            + 0.09 * material_energy
            + check_energy
        )
        ambiguity_energy = 0.12 * uncertainty_energy + ACTIVE_AMBIGUITY_WEIGHT * ambiguity_term
        precision_penalty = 0.05 * (1.0 - precision_term)
        energy = (
            0.44 * policy_energy
            + precision_term * pragmatic_energy
            + ambiguity_energy
            + precision_penalty
            - ACTIVE_EPISTEMIC_WEIGHT * epistemic_term
        )
        rollout_score = max(1e-6, 1.0 - energy)
        item["imagined_value"] = value
        item["rollout_score"] = rollout_score
        item["line_depth"] = line_depth
        item["line_confidence"] = confidence_term
        item["energy"] = energy
        item["free_energy"] = energy
        item["ambiguity"] = ambiguity_term
        item["epistemic_value"] = epistemic_term
        item["precision"] = precision_term
        item["selector"] = "world_rollout"
        item["score"] = rollout_score

    base = candidates[0]
    best = max(candidates, key=lambda entry: entry.get("rollout_score", entry.get("score", 0.0)))
    if best is not base:
        improvement = float(base.get("energy", 1.0)) - float(best.get("energy", 1.0))
        policy_ratio = float(best.get("probability", 0.0)) / max(float(base.get("probability", 0.0)), 1e-9)
        value_gain = float(best.get("imagined_value", 0.0)) - float(base.get("imagined_value", 0.0))
        if improvement < 0.045 and value_gain < 0.18 and policy_ratio < 0.72:
            base["score"] = max(float(best["score"]) + 1e-6, float(base.get("score", 0.0)))
            base["rollout_score"] = base["score"]
            base["selector"] = "world_rollout"
            base["anchor_gate"] = True

    for item in scored[len(candidates):]:
        item["rollout_score"] = item.get("score", 0.0)
        item["imagined_value"] = None
        item["line_depth"] = 0
        item["line_confidence"] = None
        item["energy"] = None
        item["free_energy"] = None
        item["ambiguity"] = None
        item["epistemic_value"] = None
        item["precision"] = None
        item["material_delta"] = None
        item["selector"] = "policy"

    return sorted(scored, key=lambda entry: entry.get("rollout_score", entry["score"]), reverse=True)


def _choose_move(scored: list[dict], top_k: int, temperature: float) -> dict:
    if not scored:
        raise HTTPException(status_code=422, detail="No legal moves available.")
    if top_k <= 1 or temperature <= 0:
        return scored[0]

    pool = scored[:top_k]
    total = sum(max(0.0, item["score"]) for item in pool)
    if total <= 0:
        return pool[0]
    return random.choices(pool, weights=[max(0.0, item["score"]) for item in pool], k=1)[0]


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "checkpoint": Path(os.getenv("CHESS_GOD_CHECKPOINT", str(DEFAULT_CHECKPOINT))).name,
    }


@app.post("/api/move", response_model=MoveResponse)
def move(request: MoveRequest):
    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid FEN.") from exc

    fen_before = board.fen()
    if board.is_game_over(claim_draw=True):
        return MoveResponse(
            move=None,
            san=None,
            fen_before=fen_before,
            fen_after=fen_before,
            value=0.0,
            status=_status(board),
            game_over=True,
            result=board.result(claim_draw=True),
            selector="none",
            candidates=[],
        )

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    think_steps = request.think_steps or DEFAULT_THINK_STEPS
    prescience: Optional[dict[str, Any]] = None
    selector = "policy"
    if request.include_prescience:
        prescience, scored = build_prescience_snapshot(
            model,
            board,
            think_steps=think_steps,
            temperature=request.temperature,
            max_candidates=8,
            max_trace=8,
        )
        value = float(prescience["value"])
    else:
        value, scored = _score_moves(model, board, request.temperature, think_steps)

    if _rollout_enabled(request.rollout_mode):
        try:
            scored = _score_world_rollout(model, board, scored, think_steps, line_depth=_rollout_depth(request.rollout_mode))
            selector = "world_rollout"
            if scored and scored[0].get("imagined_value") is not None:
                value = float(scored[0]["imagined_value"])
            if prescience is not None:
                prescience["selector"] = selector
                prescience["candidates"] = _prescience_candidate_payloads(scored, board, limit=8)
                prescience["confidence"] = float(scored[0].get("probability", 0.0)) if scored else 0.0
        except Exception as exc:
            print(f"World rollout fallback: {type(exc).__name__}: {exc}")
            selector = "policy"

    chosen = _choose_move(scored, request.top_k, request.temperature)
    chosen_move: chess.Move = chosen["move"]
    san = board.san(chosen_move)

    candidates = [_candidate_payload(item, board) for item in scored[:5]]

    board.push(chosen_move)
    return MoveResponse(
        move=chosen_move.uci(),
        san=san,
        fen_before=fen_before,
        fen_after=board.fen(),
        value=value,
        status=_status(board),
        game_over=board.is_game_over(claim_draw=True),
        result=board.result(claim_draw=True),
        selector=selector,
        candidates=candidates,
        prescience=prescience,
    )


def _inference_snapshot(request: PrescienceRequest):
    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid FEN.") from exc

    if board.is_game_over(claim_draw=True):
        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == chess.WHITE else "black",
            "think_steps": request.think_steps or DEFAULT_THINK_STEPS,
            "value": 0.0,
            "confidence": 0.0,
            "candidate_count": 0,
            "candidates": [],
            "source_heatmap": {},
            "target_heatmap": {},
            "activation_heatmap": {},
            "trace": [],
        }

    try:
        model = load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    snapshot, _ = build_prescience_snapshot(
        model,
        board,
        think_steps=request.think_steps or DEFAULT_THINK_STEPS,
        temperature=request.temperature,
        max_candidates=request.max_candidates,
        max_trace=request.max_trace,
    )
    return snapshot


@app.post("/api/inference")
def inference(request: PrescienceRequest):
    return _inference_snapshot(request)


@app.post("/api/prescience")
def prescience(request: PrescienceRequest):
    return _inference_snapshot(request)
