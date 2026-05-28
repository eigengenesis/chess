import { Chess } from "https://cdn.jsdelivr.net/npm/chess.js@1.0.0/+esm";

const boardEl = document.querySelector("#board");
const gameStatusEl = document.querySelector("#gameStatus");
const connectionStatusEl = document.querySelector("#connectionStatus");
const wakeApiButton = document.querySelector("#wakeApi");
const wakeHintEl = document.querySelector("#wakeHint");
const newGameButton = document.querySelector("#newGame");
const undoButton = document.querySelector("#undoMove");
const flipButton = document.querySelector("#flipBoard");
const rolloutButton = document.querySelector("#rolloutMode");
const colorButtons = [...document.querySelectorAll("button[data-color]")];
const prescienceStateEl = document.querySelector("#prescienceState");
const valueScoreEl = document.querySelector("#valueScore");
const confidenceScoreEl = document.querySelector("#confidenceScore");
const sourceMapEl = document.querySelector("#sourceMap");
const targetMapEl = document.querySelector("#targetMap");
const activationMapEl = document.querySelector("#activationMap");
const candidateMovesEl = document.querySelector("#candidateMoves");
const thinkTraceEl = document.querySelector("#thinkTrace");
const resultBurstEl = document.querySelector("#resultBurst");
const gameOverModalEl = document.querySelector("#gameOverModal");
const gameOverTitleEl = document.querySelector("#gameOverTitle");
const gameOverMessageEl = document.querySelector("#gameOverMessage");
const restartGameButton = document.querySelector("#restartGame");
const thoughtIndicatorEl = document.querySelector("#thoughtIndicator");
const thoughtTextEl = document.querySelector("#thoughtText");

const API_BASE = (window.CHESS_GOD_CONFIG?.apiBaseUrl || "http://127.0.0.1:8000/api").replace(/\/$/, "");
const TRACE_FRAME_MS = 120;
const TRACE_SETTLE_MS = 220;
const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const whiteRanks = [8, 7, 6, 5, 4, 3, 2, 1];
const blackRanks = [1, 2, 3, 4, 5, 6, 7, 8];
const pieceImages = {
  wp: "https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg",
  wn: "https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg",
  wb: "https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg",
  wr: "https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg",
  wq: "https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg",
  wk: "https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg",
  bp: "https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg",
  bn: "https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg",
  bb: "https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg",
  br: "https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg",
  bq: "https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg",
  bk: "https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg",
};
const fallbackGlyphs = {
  wp: "♙",
  wn: "♘",
  wb: "♗",
  wr: "♖",
  wq: "♕",
  wk: "♔",
  bp: "♟",
  bn: "♞",
  bb: "♝",
  br: "♜",
  bq: "♛",
  bk: "♚",
};
const pieceNames = {
  p: "pawn",
  n: "knight",
  b: "bishop",
  r: "rook",
  q: "queen",
  k: "king",
};

let game = new Chess();
let selectedSquare = null;
let legalTargets = new Set();
let orientation = "white";
let playerColor = "w";
let rolloutEnabled = false;
let locked = false;
let apiReachable = false;
let apiWaking = false;
let prescienceData = null;
let prescienceMode = "Idle";
let prescienceFrame = null;
let activeTraceStep = null;
let replayTimer = null;
let replayResolve = null;
let prescienceRequestId = 0;
let moveRequestId = 0;
let lastResultFen = null;
let resultNoticeTimer = null;
let dragState = null;
let dragGhost = null;

function boardFiles() {
  return orientation === "white" ? files : [...files].reverse();
}

function boardRanks() {
  return orientation === "white" ? whiteRanks : blackRanks;
}

function statusText() {
  if (game.isCheckmate()) return `Checkmate, ${game.turn() === "w" ? "Black" : "White"} wins`;
  if (game.isStalemate()) return "Stalemate";
  if (game.isDraw()) return "Draw";
  const turn = game.turn() === "w" ? "White" : "Black";
  return game.inCheck() ? `🚨 ${turn} in check` : `${turn} to move`;
}

function colorName(color) {
  return color === "w" ? "White" : "Black";
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function signedScore(value) {
  const score = Number(value);
  if (!Number.isFinite(score)) return "--";
  return `${score >= 0 ? "+" : ""}${score.toFixed(2)}`;
}

function pieceImage(piece) {
  return piece ? pieceImages[`${piece.color}${piece.type}`] : "";
}

function fallbackPiece(piece) {
  return piece ? fallbackGlyphs[`${piece.color}${piece.type}`] : "";
}

function kingSquare(color) {
  for (let rank = 1; rank <= 8; rank += 1) {
    for (const file of files) {
      const square = `${file}${rank}`;
      const piece = game.get(square);
      if (piece?.type === "k" && piece.color === color) return square;
    }
  }
  return null;
}

function renderBoard() {
  const lastMove = game.history({ verbose: true }).at(-1);
  const checkedKing = game.inCheck() ? kingSquare(game.turn()) : null;
  const isMate = game.isCheckmate();
  const isOver = game.isGameOver();
  boardEl.innerHTML = "";

  for (const rank of boardRanks()) {
    for (const file of boardFiles()) {
      const square = `${file}${rank}`;
      const piece = game.get(square);
      const isLight = (files.indexOf(file) + rank) % 2 === 1;
      const button = document.createElement("button");

      button.className = `square ${isLight ? "light" : "dark"}`;
      button.dataset.square = square;
      button.disabled = locked || isOver;
      button.setAttribute("aria-label", square);

      if (selectedSquare === square) button.classList.add("selected");
      if (legalTargets.has(square)) button.classList.add(piece ? "capture" : "target");
      if (lastMove && (lastMove.from === square || lastMove.to === square)) button.classList.add("last-move");
      if (checkedKing === square) button.classList.add(isMate ? "king-checkmate" : "king-check");

      if (piece) {
        const img = document.createElement("img");
        img.className = `piece ${piece.color === "b" ? "black-piece" : "white-piece"}`;
        img.src = pieceImage(piece);
        img.alt = `${piece.color === "w" ? "White" : "Black"} ${pieceNames[piece.type]}`;
        img.draggable = false;
        img.addEventListener(
          "error",
          () => {
            const span = document.createElement("span");
            span.className = "piece piece-fallback";
            span.textContent = fallbackPiece(piece);
            img.replaceWith(span);
          },
          { once: true },
        );
        button.appendChild(img);
      }

      button.addEventListener("click", () => onSquareClick(square));
      button.addEventListener("pointerdown", (e) => onDragStart(e, square));
      boardEl.appendChild(button);
    }
  }
}

function renderStatus() {
  gameStatusEl.textContent = statusText();
  if (locked) {
    connectionStatusEl.textContent = "Thinking";
  } else if (apiWaking) {
    connectionStatusEl.textContent = "Starting";
  } else {
    connectionStatusEl.textContent = apiReachable ? "✓ Ready" : "Offline";
  }
  connectionStatusEl.classList.toggle("ready", apiReachable && !apiWaking && !locked);
  connectionStatusEl.classList.toggle("waking", apiWaking);
  wakeApiButton.textContent = apiReachable ? "Model ready" : apiWaking ? "Starting..." : "Start model";
  wakeApiButton.classList.toggle("ready", apiReachable && !apiWaking);
  wakeApiButton.disabled = apiWaking;
  wakeHintEl.hidden = apiReachable && !apiWaking;
  rolloutButton.classList.toggle("active", rolloutEnabled);
  rolloutButton.setAttribute("aria-pressed", rolloutEnabled ? "true" : "false");
  rolloutButton.title = rolloutEnabled ? "Imagine mode is on" : "Single-pass policy inference is on";
}

function isModelThinking() {
  return locked || prescienceMode === "Moving" || prescienceMode === "Mapping" || Boolean(prescienceFrame);
}

function renderThoughtIndicator() {
  const thinking = isModelThinking();
  thoughtIndicatorEl.classList.toggle("active", thinking);
  thoughtIndicatorEl.setAttribute("aria-label", thinking ? "Model thinking" : "Model idle");
  thoughtTextEl.textContent = "thinking";
}

function clearResultBurst() {
  if (resultNoticeTimer) window.clearTimeout(resultNoticeTimer);
  resultNoticeTimer = null;
  resultBurstEl.innerHTML = "";
  resultBurstEl.className = "result-burst";
}

function triggerResultBurst(kind) {
  clearResultBurst();
  resultBurstEl.classList.add("show", kind);

  const toast = document.createElement("div");
  toast.className = "result-toast";
  toast.textContent = kind === "win" ? "YOU WIN" : "YOU LOST";
  resultBurstEl.appendChild(toast);

  if (kind === "win") {
    for (let index = 0; index < 84; index += 1) {
      const piece = document.createElement("span");
      piece.className = "confetti";
      piece.style.setProperty("--x", `${Math.random() * 100}vw`);
      piece.style.setProperty("--delay", `${Math.random() * 0.28}s`);
      piece.style.setProperty("--dur", `${0.85 + Math.random() * 0.75}s`);
      piece.style.setProperty("--rot", `${Math.round(Math.random() * 540)}deg`);
      piece.style.setProperty("--shade", index % 3 === 0 ? "#ffffff" : index % 3 === 1 ? "#bdbdbd" : "#707070");
      resultBurstEl.appendChild(piece);
    }
  }

  resultNoticeTimer = window.setTimeout(() => {
    clearResultBurst();
  }, kind === "win" ? 2200 : 1500);
}

function gameOverDetails() {
  if (!game.isGameOver()) return null;

  if (game.isCheckmate()) {
    const winner = game.turn() === "w" ? "b" : "w";
    return {
      kind: winner === playerColor ? "win" : "lose",
      message: `Checkmate. ${colorName(winner)} wins.`,
    };
  }

  if (game.isStalemate()) {
    return {
      kind: "draw",
      message: "Stalemate. The game is a draw.",
    };
  }

  if (typeof game.isInsufficientMaterial === "function" && game.isInsufficientMaterial()) {
    return {
      kind: "draw",
      message: "Draw by insufficient material.",
    };
  }

  if (typeof game.isThreefoldRepetition === "function" && game.isThreefoldRepetition()) {
    return {
      kind: "draw",
      message: "Draw by threefold repetition.",
    };
  }

  if (game.isDraw()) {
    return {
      kind: "draw",
      message: "Draw. No winner this time.",
    };
  }

  return {
    kind: "draw",
    message: "The game has ended.",
  };
}

function renderGameOverModal(details) {
  if (!details) {
    gameOverModalEl.hidden = true;
    gameOverModalEl.className = "game-over-modal";
    return;
  }

  gameOverModalEl.hidden = false;
  gameOverModalEl.className = `game-over-modal ${details.kind}`;
  gameOverTitleEl.textContent = "Game Over";
  gameOverMessageEl.textContent = details.message;
}

function renderGameEffects() {
  const details = gameOverDetails();
  if (!details) {
    renderGameOverModal(null);
    return;
  }

  const fen = game.fen();
  if (lastResultFen === fen) return;
  lastResultFen = fen;

  if (details.kind !== "draw") {
    triggerResultBurst(details.kind);
    setTimeout(() => renderGameOverModal(details), 1500);
  } else {
    renderGameOverModal(details);
  }
  restartGameButton.focus({ preventScroll: true });
}

function renderMindBoard(container, heatmap = {}, mode = "policy") {
  const boardKey = orientation;
  if (container.children.length !== 64 || container.dataset.orientation !== boardKey) {
    container.innerHTML = "";
    container.dataset.orientation = boardKey;
    for (const rank of boardRanks()) {
      for (const file of boardFiles()) {
        const square = `${file}${rank}`;
        const isLight = (files.indexOf(file) + rank) % 2 === 1;
        const cell = document.createElement("span");
        cell.className = `mind-cell ${isLight ? "light" : "dark"}`;
        cell.dataset.square = square;
        container.appendChild(cell);
      }
    }
  }

  const cells = [...container.children];
  for (const rank of boardRanks()) {
    for (const file of boardFiles()) {
      const square = `${file}${rank}`;
      const rawHeat = Number(heatmap[square] || 0);
      const shapedHeat = mode === "activation" ? Math.pow(rawHeat, 1.85) : rawHeat;
      const cell = cells.shift();
      cell.style.setProperty("--heat", Math.min(0.92, shapedHeat * 0.92).toFixed(3));
      cell.setAttribute("aria-label", `${square} ${rawHeat.toFixed(3)}`);
    }
  }
}

function emptyPanel(text) {
  const box = document.createElement("div");
  box.className = "empty-panel";
  box.textContent = text;
  return box;
}

function renderCandidates(candidates = []) {
  candidateMovesEl.innerHTML = "";
  if (!candidates.length) {
    candidateMovesEl.appendChild(emptyPanel("No model move yet."));
    return;
  }

  const top = candidates[0];
  const topLegal = document.createElement("div");
  topLegal.className = "top-legal-stat";
  topLegal.innerHTML = `
    <span>Top legal</span>
    <strong>${percent(top?.probability)}</strong>
  `;
  candidateMovesEl.appendChild(topLegal);

  const maxProbability = Math.max(...candidates.map((item) => item.probability || 0), 0.0001);
  candidates.slice(0, 3).forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "candidate";
    row.innerHTML = `
      <div class="candidate-top">
        <strong>${index + 1}. ${item.san}</strong>
        <span>${item.from}->${item.to} · ${percent(item.probability)}</span>
      </div>
      <div class="bar" aria-hidden="true">
        <div class="bar-fill"></div>
      </div>
    `;
    row.querySelector(".bar-fill").style.setProperty("--bar", ((item.probability || 0) / maxProbability).toFixed(3));
    candidateMovesEl.appendChild(row);
  });
}

function renderTrace(trace = []) {
  thinkTraceEl.innerHTML = "";
  if (!trace.length) {
    thinkTraceEl.appendChild(emptyPanel("Trace appears during inference."));
    return;
  }

  for (const item of trace) {
    const confidence = Number(item.confidence || 0);
    const margin = Number(item.margin || 0);
    const uncertainty = Number(item.uncertainty || 0);
    const move = item.top_san || item.top_move || "...";
    const row = document.createElement("div");
    row.className = `trace-row ${item.step === activeTraceStep ? "active" : ""}`;
    row.innerHTML = `
      <div class="trace-top trace-main">
        <strong>${String(item.step).padStart(2, "0")}</strong>
        <span class="trace-move">${move}</span>
        <span class="trace-confidence">${percent(confidence)}</span>
      </div>
      <div class="trace-metrics">
        <span>value <b>${signedScore(item.value)}</b></span>
        <span>margin <b>${percent(margin)}</b></span>
        <span>doubt <b>${percent(uncertainty)}</b></span>
      </div>
      <div class="trace-focus">
        <span>latent focus</span>
        <strong>${item.focus || "--"}</strong>
      </div>
      <div class="bar" aria-hidden="true">
        <div class="bar-fill"></div>
      </div>
    `;
    row.querySelector(".bar-fill").style.setProperty("--bar", confidence.toFixed(3));
    thinkTraceEl.appendChild(row);
  }
}

function renderPrescience() {
  renderThoughtIndicator();
  prescienceStateEl.textContent = prescienceMode;
  prescienceStateEl.classList.toggle("has-step", prescienceMode.startsWith("Step"));
  prescienceStateEl.classList.toggle("thinking", prescienceMode === "Thinking" || prescienceFrame);

  if (!prescienceData) {
    valueScoreEl.textContent = "--";
    confidenceScoreEl.textContent = "--";
    renderMindBoard(sourceMapEl);
    renderMindBoard(targetMapEl);
    renderMindBoard(activationMapEl);
    renderCandidates([]);
    renderTrace([]);
    return;
  }

  const frame = prescienceFrame || prescienceData;
  valueScoreEl.textContent = Number(frame.value ?? prescienceData.value ?? 0).toFixed(2);
  confidenceScoreEl.textContent = `${Math.round((frame.confidence ?? prescienceData.confidence ?? 0) * 100)}%`;
  renderMindBoard(sourceMapEl, frame.source_heatmap ?? prescienceData.source_heatmap);
  renderMindBoard(targetMapEl, frame.target_heatmap ?? prescienceData.target_heatmap);
  const worldHeatmap = frame.world_heatmap
    ?? prescienceData.world_heatmap
    ?? frame.activation_heatmap
    ?? prescienceData.activation_heatmap;
  renderMindBoard(activationMapEl, worldHeatmap, prescienceData.world_heatmap ? "world" : "activation");
  renderCandidates(prescienceData.candidates);
  renderTrace(prescienceData.trace);
}

function stopPrescienceReplay() {
  if (replayTimer) window.clearInterval(replayTimer);
  if (replayResolve) replayResolve();
  replayTimer = null;
  replayResolve = null;
  prescienceFrame = null;
  activeTraceStep = null;
}

function startPrescienceReplay(data) {
  stopPrescienceReplay();
  const frames = data.trace || [];
  if (!frames.length) {
    prescienceMode = "Ready";
    renderPrescience();
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    replayResolve = resolve;
    let index = 0;
    const showFrame = () => {
      const frame = frames[index];
      prescienceFrame = frame;
      activeTraceStep = frame.step;
      prescienceMode = `Step ${String(frame.step).padStart(2, "0")}/${data.think_steps || 16}`;
      renderPrescience();
      index += 1;

      if (index >= frames.length) {
        window.clearInterval(replayTimer);
        replayTimer = window.setTimeout(() => {
          replayTimer = null;
          replayResolve = null;
          prescienceFrame = null;
          activeTraceStep = null;
          prescienceMode = "Ready";
          renderPrescience();
          resolve();
        }, TRACE_SETTLE_MS);
      }
    };

    showFrame();
    replayTimer = window.setInterval(showFrame, TRACE_FRAME_MS);
  });
}

async function requestPrescience(fen) {
  const requestId = ++prescienceRequestId;
  prescienceMode = "Mapping";
  renderPrescience();

  try {
    const response = await fetch(`${API_BASE}/inference`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fen,
        temperature: 0.2,
        max_candidates: 8,
        max_trace: 8,
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    if (requestId !== prescienceRequestId) return;
    apiReachable = true;
    prescienceData = data;
    startPrescienceReplay(data);
  } catch (error) {
    if (requestId !== prescienceRequestId) return;
    prescienceMode = "Offline";
    apiReachable = false;
    console.error(error);
    render();
  }
}

function render() {
  renderBoard();
  renderStatus();
  renderPrescience();
  renderGameEffects();
}

function clearSelection() {
  selectedSquare = null;
  legalTargets.clear();
}

function selectSquare(square) {
  selectedSquare = square;
  legalTargets = new Set(game.moves({ square, verbose: true }).map((move) => move.to));
  render();
}

function promotionFor(from, to) {
  const piece = game.get(from);
  if (!piece || piece.type !== "p") return undefined;
  return to.endsWith(piece.color === "w" ? "8" : "1") ? "q" : undefined;
}

function squareFromPoint(x, y) {
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  const sq = el.closest(".square");
  return sq ? sq.dataset.square : null;
}

function onDragStart(e, square) {
  if (locked || game.isGameOver() || game.turn() !== playerColor) return;
  const piece = game.get(square);
  if (!piece || piece.color !== playerColor) return;

  e.preventDefault();
  dragState = { from: square, started: false };

  // Create ghost piece
  dragGhost = document.createElement("img");
  dragGhost.src = pieceImage(piece);
  dragGhost.style.cssText = `
    position: fixed; pointer-events: none; z-index: 9999;
    width: 60px; height: 60px; opacity: 0.85;
    transform: translate(-50%, -50%); display: none;
  `;
  document.body.appendChild(dragGhost);

  const px = e.clientX;
  const py = e.clientY;
  dragGhost.style.left = `${px}px`;
  dragGhost.style.top = `${py}px`;

  selectSquare(square);
}

function onDragMove(e) {
  if (!dragState || !dragGhost) return;
  const px = e.clientX ?? e.touches?.[0]?.clientX;
  const py = e.clientY ?? e.touches?.[0]?.clientY;
  if (px == null) return;
  dragState.started = true;
  dragGhost.style.display = "block";
  dragGhost.style.left = `${px}px`;
  dragGhost.style.top = `${py}px`;
}

function onDragEnd(e) {
  if (!dragState) return;
  const wasDrag = dragState.started;
  const from = dragState.from;

  // Remove ghost
  if (dragGhost) {
    dragGhost.remove();
    dragGhost = null;
  }

  if (!wasDrag) {
    dragState = null;
    return; // Let click handler deal with it
  }

  const px = e.clientX ?? e.changedTouches?.[0]?.clientX;
  const py = e.clientY ?? e.changedTouches?.[0]?.clientY;
  const to = squareFromPoint(px, py);
  dragState = null;

  if (!to || to === from) {
    clearSelection();
    render();
    return;
  }

  let move = null;
  try {
    move = game.move({ from, to, promotion: promotionFor(from, to) });
  } catch { move = null; }

  if (move) {
    clearSelection();
    render();
    window.setTimeout(requestModelMove, 120);
  } else {
    clearSelection();
    render();
  }
}

document.addEventListener("pointermove", onDragMove);
document.addEventListener("pointerup", onDragEnd);

function onSquareClick(square) {
  if (locked || game.isGameOver() || game.turn() !== playerColor) return;

  const piece = game.get(square);
  if (!selectedSquare) {
    if (piece?.color === playerColor) selectSquare(square);
    return;
  }

  if (selectedSquare === square) {
    clearSelection();
    render();
    return;
  }

  if (piece?.color === playerColor) {
    selectSquare(square);
    return;
  }

  let move = null;
  try {
    move = game.move({
      from: selectedSquare,
      to: square,
      promotion: promotionFor(selectedSquare, square),
    });
  } catch {
    move = null;
  }

  if (move) {
    clearSelection();
    render();
    window.setTimeout(requestModelMove, 120);
    return;
  }

  if (piece?.color === playerColor) selectSquare(square);
  else {
    clearSelection();
    render();
  }
}

function parseUci(uci) {
  return {
    from: uci.slice(0, 2),
    to: uci.slice(2, 4),
    promotion: uci.length > 4 ? uci[4] : undefined,
  };
}

async function requestModelMove() {
  if (game.isGameOver() || game.turn() === playerColor) {
    render();
    return;
  }

  const requestId = ++moveRequestId;
  const fenBefore = game.fen();
  locked = true;
  stopPrescienceReplay();
  render();

  try {
    prescienceMode = "Moving";
    renderPrescience();

    const response = await fetch(`${API_BASE}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fen: fenBefore,
        temperature: 0.2,
        top_k: 1,
        include_prescience: true,
        rollout_mode: rolloutEnabled ? "line" : "off",
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    if (requestId !== moveRequestId || game.fen() !== fenBefore || data.fen_before !== fenBefore) return;

    apiReachable = true;
    if (data.prescience) {
      prescienceData = data.prescience;
      await startPrescienceReplay(data.prescience);
      if (requestId !== moveRequestId || game.fen() !== fenBefore) return;
    }
    if (data.move) {
      const move = game.move(parseUci(data.move));
      if (!move) throw new Error(`Server returned illegal move: ${data.move}`);
      if (data.fen_after && game.fen() !== data.fen_after) game.load(data.fen_after);
    }
    prescienceMode = "Ready";
  } catch (error) {
    if (requestId !== moveRequestId) return;
    apiReachable = false;
    prescienceMode = "Offline";
    console.error(error);
  } finally {
    if (requestId === moveRequestId) {
      locked = false;
      render();
    }
  }
}

function newGame() {
  prescienceRequestId += 1;
  moveRequestId += 1;
  locked = false;
  game = new Chess();
  lastResultFen = null;
  clearResultBurst();
  clearSelection();
  stopPrescienceReplay();
  prescienceData = null;
  prescienceMode = "Idle";
  render();
  if (playerColor === "b") window.setTimeout(requestModelMove, 120);
}

async function pingApi() {
  apiWaking = true;
  renderStatus();
  try {
    const response = await fetch(`${API_BASE}/health`);
    apiReachable = response.ok;
  } catch {
    apiReachable = false;
  } finally {
    apiWaking = false;
  }
  renderStatus();
}

wakeApiButton.addEventListener("click", pingApi);
newGameButton.addEventListener("click", newGame);
restartGameButton.addEventListener("click", newGame);

undoButton.addEventListener("click", () => {
  if (locked) return;
  prescienceRequestId += 1;
  moveRequestId += 1;
  lastResultFen = null;
  clearResultBurst();
  stopPrescienceReplay();
  const undone = game.undo();
  if (!undone) {
    render();
    return;
  }
  if (game.history().length) game.undo();
  clearSelection();
  prescienceData = null;
  prescienceMode = "Idle";
  render();
  if (!game.isGameOver()) requestPrescience(game.fen());
});

flipButton.addEventListener("click", () => {
  orientation = orientation === "white" ? "black" : "white";
  render();
});

rolloutButton.addEventListener("click", () => {
  if (locked) return;
  rolloutEnabled = !rolloutEnabled;
  render();
});

colorButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (locked) return;
    prescienceRequestId += 1;
    stopPrescienceReplay();
    playerColor = button.dataset.color;
    orientation = playerColor === "w" ? "white" : "black";
    colorButtons.forEach((item) => item.classList.toggle("active", item === button));
    newGame();
  });
});

render();
pingApi();
