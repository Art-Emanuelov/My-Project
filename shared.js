// shared.js — вся математика: константы, карта, сеть, симуляция
const TILE = 20;
const COLS = 30, ROWS = 20;
const W = COLS * TILE, H = ROWS * TILE;
const FPS = 30;

const PLAYER_SPEED = 3;
const BULLET_SPEED = 500 / FPS;
const BULLET_DAMAGE = 20;
const SHOOT_COOLDOWN = 3;
const RELOAD_TIME = 45;
const MAG_SIZE = 30;
const RESERVE_MAX = 90;
const HP_MAX = 100;
const MAX_FRAMES = 2310;

const DIRS = [
  {dx: 1, dy: 0},  {dx: 1, dy: 1},  {dx: 0, dy: 1},  {dx: -1, dy: 1},
  {dx: -1, dy: 0}, {dx: -1, dy: -1},{dx: 0, dy: -1}, {dx: 1, dy: -1},
];
const DIR_ANGLES = [0, Math.PI/4, Math.PI/2, 3*Math.PI/4, Math.PI,
                    -3*Math.PI/4, -Math.PI/2, -Math.PI/4];

const RAY_COUNT = 16;
const RAY_ANGLES = [];
for (let i = 0; i < RAY_COUNT; i++) RAY_ANGLES.push(i * Math.PI * 2 / RAY_COUNT);
const RAY_MAX = 250;

const N_RAYS = 16;
const N_RAYS_3 = N_RAYS * 3;
const N_DIRECT = 14;
const N_INPUTS = N_RAYS_3 + N_DIRECT;
const N_HIDDEN = 24;
const N_OUT_MOVE = 9, N_OUT_ROT = 9, N_OUT_ACT = 2;
const N_OUTPUTS = N_OUT_MOVE + N_OUT_ROT + N_OUT_ACT;
const GENOME_SIZE = N_INPUTS*N_HIDDEN + N_HIDDEN + N_HIDDEN*N_OUTPUTS + N_OUTPUTS;
const WEIGHT_CLAMP = 8;

function randomGenome() {
  const g = new Float32Array(GENOME_SIZE);
  for (let i = 0; i < GENOME_SIZE; i++) g[i] = (Math.random()*2-1)*0.5;
  return g;
}

function genomeToWeights(g) {
  let k = 0;
  const W1 = [];
  for (let i = 0; i < N_INPUTS; i++) {
    W1[i] = new Float32Array(N_HIDDEN);
    for (let j = 0; j < N_HIDDEN; j++) W1[i][j] = g[k++];
  }
  const b1 = new Float32Array(N_HIDDEN);
  for (let j = 0; j < N_HIDDEN; j++) b1[j] = g[k++];
  const W2 = [];
  for (let i = 0; i < N_HIDDEN; i++) {
    W2[i] = new Float32Array(N_OUTPUTS);
    for (let j = 0; j < N_OUTPUTS; j++) W2[i][j] = g[k++];
  }
  const b2 = new Float32Array(N_OUTPUTS);
  for (let j = 0; j < N_OUTPUTS; j++) b2[j] = g[k++];
  return { W1, b1, W2, b2 };
}

function freshBaseMap() {
  const m = [];
  for (let r = 0; r < ROWS; r++) {
    m[r] = [];
    for (let c = 0; c < COLS; c++) {
      m[r][c] = (r === 0 || r === ROWS-1 || c === 0 || c === COLS-1) ? 1 : 0;
    }
  }
  return m;
}

function isConnected(m) {
  const visited = new Uint8Array(ROWS * COLS);
  const queue = [[10, 2]];
  visited[10 * COLS + 2] = 1;
  const dirs = [[1,0],[-1,0],[0,1],[0,-1]];
  while (queue.length) {
    const [r, c] = queue.shift();
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
      if (visited[nr * COLS + nc] || m[nr][nc] !== 0) continue;
      visited[nr * COLS + nc] = 1;
      queue.push([nr, nc]);
    }
  }
  return visited[10 * COLS + (COLS - 3)] === 1;
}

function generateRandomMap() {
  for (let attempt = 0; attempt < 30; attempt++) {
    const m = freshBaseMap();
    const numBlocks = 3 + Math.floor(Math.random() * 4);
    for (let i = 0; i < numBlocks; i++) {
      const halfC = Math.floor(COLS / 2) - 2;
      const r = 2 + Math.floor(Math.random() * (ROWS - 6));
      const c = 2 + Math.floor(Math.random() * (halfC - 3));
      const h = 2 + Math.floor(Math.random() * 3);
      const w = 1 + Math.floor(Math.random() * 3);
      for (let rr = r; rr < r + h && rr < ROWS - 1; rr++)
        for (let cc = c; cc < c + w && cc < COLS - 1; cc++)
          m[rr][cc] = 1;
    }
    for (let r = 0; r < ROWS; r++)
      for (let c = 0; c < Math.floor(COLS / 2); c++)
        m[r][COLS - 1 - c] = m[r][c];

    m[10][2] = 0; m[10][COLS-3] = 0;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if (10+dr > 0 && 10+dr < ROWS-1 && 2+dc > 0 && 2+dc < COLS-1)
          m[10+dr][2+dc] = 0;
        if (10+dr > 0 && 10+dr < ROWS-1 && COLS-3+dc > 0 && COLS-3+dc < COLS-1)
          m[10+dr][COLS-3+dc] = 0;
      }
    }
    if (!isConnected(m)) continue;

    const numBoxes = 2 + Math.floor(Math.random() * 4);
    let placed = 0;
    for (let tries = 0; tries < 50 && placed < numBoxes; tries++) {
      const r = 2 + Math.floor(Math.random() * (ROWS - 4));
      const c = 2 + Math.floor(Math.random() * (COLS - 4));
      if (m[r][c] !== 0) continue;
      if (Math.abs(r - 10) <= 2 && (c <= 4 || c >= COLS - 5)) continue;
      m[r][c] = 2;
      if (!isConnected(m)) { m[r][c] = 0; continue; }
      placed++;
    }
    return m;
  }
  return freshBaseMap();
}

function isSolidMap(m, r, c) {
  if (r < 0 || r >= ROWS || c < 0 || c >= COLS) return true;
  return m[r][c] !== 0;
}

function circleHitsWallMap(mapRef, cx, cy, r) {
  const r0 = Math.floor((cy - r) / TILE);
  const r1 = Math.floor((cy + r) / TILE);
  const c0 = Math.floor((cx - r) / TILE);
  const c1 = Math.floor((cx + r) / TILE);
  for (let rr = r0; rr <= r1; rr++) {
    for (let cc = c0; cc <= c1; cc++) {
      if (isSolidMap(mapRef, rr, cc)) {
        const x0 = cc * TILE, y0 = rr * TILE, x1 = x0 + TILE, y1 = y0 + TILE;
        const nx = Math.max(x0, Math.min(cx, x1));
        const ny = Math.max(y0, Math.min(cy, y1));
        if ((cx - nx) ** 2 + (cy - ny) ** 2 < r * r) return true;
      }
    }
  }
  return false;
}

function hasLineOfSightMap(mapRef, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const dist = Math.hypot(dx, dy);
  const steps = Math.ceil(dist / 6);
  for (let i = 1; i < steps; i++) {
    const t = i / steps;
    if (isSolidMap(mapRef, Math.floor((y1+dy*t)/TILE), Math.floor((x1+dx*t)/TILE)))
      return false;
  }
  return true;
}

function castRayFrom(mapRef, x0, y0, angleRad, enemyPos) {
  const dx = Math.cos(angleRad), dy = Math.sin(angleRad);
  const step = 4;
  let wallDist = RAY_MAX, enemyDist = 0, boxDist = 0;
  let enemyFound = false, boxFound = false;
  for (let d = step; d <= RAY_MAX; d += step) {
    const x = x0 + dx * d, y = y0 + dy * d;
    const c = Math.floor(x / TILE), r = Math.floor(y / TILE);
    if (r < 0 || r >= ROWS || c < 0 || c >= COLS) { wallDist = d; break; }
    if (mapRef[r][c] === 1) { wallDist = d; break; }
    if (mapRef[r][c] === 2 && !boxFound) { boxDist = d; boxFound = true; }
    if (!enemyFound && enemyPos) {
      const edx = enemyPos.x - x, edy = enemyPos.y - y;
      if (edx*edx + edy*edy < 100) { enemyDist = d; enemyFound = true; }
    }
  }
  return { wallDist, enemyDist, boxDist, enemyFound, boxFound };
}

function computeInputsFrom(agent, enemy, mapRef, noiseList, inputArr) {
  for (let i = 0; i < N_RAYS; i++) {
    const r = castRayFrom(mapRef, agent.x, agent.y, RAY_ANGLES[i], enemy);
    inputArr[i*3 + 0] = 1 / (r.wallDist + 1);
    inputArr[i*3 + 1] = r.enemyFound ? 1 / (r.enemyDist + 1) : 0;
    inputArr[i*3 + 2] = r.boxFound ? 1 / (r.boxDist + 1) : 0;
  }
  const dxE = enemy.x - agent.x, dyE = enemy.y - agent.y;
  const distE = Math.hypot(dxE, dyE);
  const angleE = Math.atan2(dyE, dxE);
  const seesEnemy = hasLineOfSightMap(mapRef, agent.x, agent.y, enemy.x, enemy.y) ? 1 : 0;
  const facing = DIR_ANGLES[agent.dir];
  let nStrength = 0, nSin = 0, nCos = 0;
  if (noiseList.length) {
    let maxN = noiseList[0];
    for (const n of noiseList) if (n.life > maxN.life) maxN = n;
    nStrength = maxN.life / maxN.maxLife;
    const ndx = maxN.x - agent.x, ndy = maxN.y - agent.y;
    const nd = Math.hypot(ndx, ndy) || 1;
    nSin = ndy / nd;
    nCos = ndx / nd;
  }
  const o = N_RAYS_3;
  inputArr[o + 0] = agent.hp / HP_MAX;
  inputArr[o + 1] = agent.ammo / MAG_SIZE;
  inputArr[o + 2] = agent.reserve / RESERVE_MAX;
  inputArr[o + 3] = agent.reloadTimer > 0 ? 1 : 0;
  inputArr[o + 4] = Math.sin(facing);
  inputArr[o + 5] = Math.cos(facing);
  inputArr[o + 6] = seesEnemy ? Math.sin(angleE) : 0;
  inputArr[o + 7] = seesEnemy ? Math.cos(angleE) : 0;
  inputArr[o + 8] = seesEnemy ? 1 / (distE + 1) : 0;
  inputArr[o + 9] = seesEnemy;
  inputArr[o + 10] = nStrength;
  inputArr[o + 11] = nSin;
  inputArr[o + 12] = nCos;
  inputArr[o + 13] = 1;
}

function softmaxRange(arr, start, len) {
  let mx = -Infinity;
  for (let i = start; i < start + len; i++) if (arr[i] > mx) mx = arr[i];
  let sum = 0;
  for (let i = start; i < start + len; i++) {
    arr[i] = Math.exp(arr[i] - mx); sum += arr[i];
  }
  for (let i = start; i < start + len; i++) arr[i] /= sum;
}
function argmaxRange(arr, start, len) {
  let best = start, bv = -Infinity;
  for (let i = start; i < start + len; i++) if (arr[i] > bv) { bv = arr[i]; best = i; }
  return best - start;
}

function forwardWith(weights, inputArr) {
  const { W1, b1, W2, b2 } = weights;
  const hidden = new Float32Array(N_HIDDEN);
  for (let j = 0; j < N_HIDDEN; j++) {
    let s = b1[j];
    for (let i = 0; i < N_INPUTS; i++) s += inputArr[i] * W1[i][j];
    hidden[j] = s > 0 ? s : 0;
  }
  const out = new Float32Array(N_OUTPUTS);
  for (let k = 0; k < N_OUTPUTS; k++) {
    let s = b2[k];
    for (let j = 0; j < N_HIDDEN; j++) s += hidden[j] * W2[j][k];
    out[k] = s;
  }
  softmaxRange(out, 0, N_OUT_MOVE);
  softmaxRange(out, N_OUT_MOVE, N_OUT_ROT);
  softmaxRange(out, N_OUT_MOVE + N_OUT_ROT, N_OUT_ACT);
  return { hidden, out };
}

function applyAgentActions(agent, weights, otherAgent, mapRef, bullets, noise) {
  const localInputs = agent._inputs;
  computeInputsFrom(agent, otherAgent, mapRef, noise, localInputs);
  const { hidden, out } = forwardWith(weights, localInputs);

  if (agent._hidden) agent._hidden.set(hidden);
  if (agent._output) agent._output.set(out);

  const moveIdx = argmaxRange(out, 0, N_OUT_MOVE);
  const rotIdx = argmaxRange(out, N_OUT_MOVE, N_OUT_ROT);
  const actBase = N_OUT_MOVE + N_OUT_ROT;
  const actionIdx = argmaxRange(out, actBase, 2);
  const wantFire = actionIdx === 0;

  if (moveIdx > 0) {
    const d = DIRS[moveIdx - 1];
    const l = Math.hypot(d.dx, d.dy) || 1;
    const nx = agent.x + d.dx * PLAYER_SPEED / l;
    const ny = agent.y + d.dy * PLAYER_SPEED / l;
    if (!circleHitsWallMap(mapRef, nx, agent.y, 8)) agent.x = nx;
    if (!circleHitsWallMap(mapRef, agent.x, ny, 8)) agent.y = ny;
  }
  if (rotIdx > 0) agent.dir = rotIdx - 1;

  if (agent.reloadTimer > 0) {
    agent.reloadTimer--;
    if (agent.reloadTimer === 0) {
      const take = Math.min(MAG_SIZE - agent.ammo, agent.reserve);
      agent.ammo += take; agent.reserve -= take;
    }
  } else if (!wantFire && agent.ammo < MAG_SIZE && agent.reserve > 0) {
    agent.reloadTimer = RELOAD_TIME;
  }
  if (agent.cooldown > 0) agent.cooldown--;
  if (wantFire && agent.cooldown === 0 && agent.reloadTimer === 0 && agent.ammo > 0) {
    const d = DIRS[agent.dir];
    const l = Math.hypot(d.dx, d.dy) || 1;
    bullets.push({
      x: agent.x + d.dx * 12, y: agent.y + d.dy * 12,
      vx: d.dx / l * BULLET_SPEED, vy: d.dy / l * BULLET_SPEED,
      owner: agent.team,
    });
    agent.ammo--; agent.cooldown = SHOOT_COOLDOWN;
    noise.push({ x: agent.x, y: agent.y, radius: 15*TILE, life: 30, maxLife: 30, kind: 'gunshot' });
  }
}

function simulateNNvsNN(gA, gB, mapRef) {
  const wA = genomeToWeights(gA);
  const wB = genomeToWeights(gB);

  const A = {
    x: 2.5 * TILE, y: 10.5 * TILE, dir: 0,
    hp: HP_MAX, ammo: MAG_SIZE, reserve: RESERVE_MAX,
    reloadTimer: 0, cooldown: 0, team: 'A',
    _inputs: new Float32Array(N_INPUTS), _hidden: null, _output: null,
  };
  const B = {
    x: (COLS - 2.5) * TILE, y: 10.5 * TILE, dir: 4,
    hp: HP_MAX, ammo: MAG_SIZE, reserve: RESERVE_MAX,
    reloadTimer: 0, cooldown: 0, team: 'B',
    _inputs: new Float32Array(N_INPUTS), _hidden: null, _output: null,
  };
  const bullets = [];
  const noise = [];

  let frames = 0;
  let dmgA = 0, dmgB = 0;
  let minDist = 10000;
  let framesSinceSeenA = 0, framesSinceSeenB = 0;
  let stillFramesA = 0, stillFramesB = 0;
  let lastPosA = { x: A.x, y: A.y }, lastPosB = { x: B.x, y: B.y };
  const visitedA = new Set(), visitedB = new Set();
  visitedA.add(Math.floor(A.y / TILE) * COLS + Math.floor(A.x / TILE));
  visitedB.add(Math.floor(B.y / TILE) * COLS + Math.floor(B.x / TILE));

  while (frames < MAX_FRAMES) {
    frames++;

    const ddx = A.x - B.x, ddy = A.y - B.y;
    const d = Math.hypot(ddx, ddy);
    if (d < minDist) minDist = d;

    const aSeesB = hasLineOfSightMap(mapRef, A.x, A.y, B.x, B.y);
    const bSeesA = hasLineOfSightMap(mapRef, B.x, B.y, A.x, A.y);
    if (aSeesB) framesSinceSeenA = 0; else framesSinceSeenA++;
    if (bSeesA) framesSinceSeenB = 0; else framesSinceSeenB++;

    applyAgentActions(A, wA, B, mapRef, bullets, noise);
    applyAgentActions(B, wB, A, mapRef, bullets, noise);

    if (frames % 30 === 0) {
      const movedA = Math.hypot(A.x - lastPosA.x, A.y - lastPosA.y);
      const movedB = Math.hypot(B.x - lastPosB.x, B.y - lastPosB.y);
      if (movedA < 20) stillFramesA += 30; else stillFramesA = 0;
      if (movedB < 20) stillFramesB += 30; else stillFramesB = 0;
      lastPosA = { x: A.x, y: A.y };
      lastPosB = { x: B.x, y: B.y };
    }

    visitedA.add(Math.floor(A.y / TILE) * COLS + Math.floor(A.x / TILE));
    visitedB.add(Math.floor(B.y / TILE) * COLS + Math.floor(B.x / TILE));

    for (let i = bullets.length - 1; i >= 0; i--) {
      const b = bullets[i];
      const steps = Math.ceil(Math.hypot(b.vx, b.vy) / 4);
      let remove = false;
      for (let s = 0; s < steps; s++) {
        b.x += b.vx / steps; b.y += b.vy / steps;
        const c = Math.floor(b.x / TILE), r = Math.floor(b.y / TILE);
        if (r < 0 || r >= ROWS || c < 0 || c >= COLS) { remove = true; break; }
        if (mapRef[r][c] === 1) {
          noise.push({ x: b.x, y: b.y, radius: 3*TILE, life: 30, maxLife: 30, kind: 'impact' });
          remove = true; break;
        }
        if (mapRef[r][c] === 2) {
          mapRef[r][c] = 0;
          noise.push({ x: b.x, y: b.y, radius: 20*TILE, life: 30, maxLife: 30, kind: 'break' });
          remove = true; break;
        }
        if (b.owner === 'A') {
          if (Math.hypot(b.x - B.x, b.y - B.y) < 8) {
            B.hp -= BULLET_DAMAGE; dmgA += BULLET_DAMAGE;
            noise.push({ x: b.x, y: b.y, radius: 6*TILE, life: 30, maxLife: 30, kind: 'hit' });
            remove = true; break;
          }
        } else {
          if (Math.hypot(b.x - A.x, b.y - A.y) < 8) {
            A.hp -= BULLET_DAMAGE; dmgB += BULLET_DAMAGE;
            noise.push({ x: b.x, y: b.y, radius: 6*TILE, life: 30, maxLife: 30, kind: 'hit' });
            remove = true; break;
          }
        }
      }
      if (remove) bullets.splice(i, 1);
    }

    for (let i = noise.length - 1; i >= 0; i--) {
      noise[i].life--;
      if (noise[i].life <= 0) noise.splice(i, 1);
    }

    if (A.hp <= 0 || B.hp <= 0) break;
  }

  const winA = A.hp > 0 && B.hp <= 0 ? 1 : 0;
  const winB = B.hp > 0 && A.hp <= 0 ? 1 : 0;
  const hpA = Math.max(0, A.hp), hpB = Math.max(0, B.hp);
  const proximity = Math.max(0, 500 - minDist);

  const exploredA = visitedA.size / (ROWS * COLS);
  const exploredB = visitedB.size / (ROWS * COLS);
  const exploreBonusA = exploredA * 300;
  const exploreBonusB = exploredB * 300;

  const stillPenaltyA = Math.min(500, stillFramesA > 60 ? (stillFramesA - 60) * 0.5 : 0);
  const stillPenaltyB = Math.min(500, stillFramesB > 60 ? (stillFramesB - 60) * 0.5 : 0);
  const unseenPenaltyA = Math.min(400, framesSinceSeenA > 300 ? (framesSinceSeenA - 300) * 0.3 : 0);
  const unseenPenaltyB = Math.min(400, framesSinceSeenB > 300 ? (framesSinceSeenB - 300) * 0.3 : 0);

  const idlePenaltyA = dmgA === 0 ? 200 : 0;
  const idlePenaltyB = dmgB === 0 ? 200 : 0;

  const fitA = dmgA * 20 + winA * 2000 + hpA * 1 - dmgB * 1
             + proximity + exploreBonusA
             - stillPenaltyA - unseenPenaltyA - idlePenaltyA;
  const fitB = dmgB * 20 + winB * 2000 + hpB * 1 - dmgA * 1
             + proximity + exploreBonusB
             - stillPenaltyB - unseenPenaltyB - idlePenaltyB;

  return { fitA, fitB, winA, winB, dmgA, dmgB, hpA, hpB, frames, exploredA, exploredB };
}

module.exports = {
  GENOME_SIZE, WEIGHT_CLAMP,
  randomGenome, genomeToWeights,
  generateRandomMap, simulateNNvsNN,
};
