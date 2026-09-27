// train.js — оркестратор с батчингом матчей
const fs = require('fs');
const path = require('path');
const os = require('os');
const { Worker } = require('worker_threads');
const { GENOME_SIZE, WEIGHT_CLAMP, randomGenome } = require('./shared');

const TOTAL_GENS = parseInt(process.argv[2] || '10000', 10);
const CKPT_EVERY = parseInt(process.argv[3] || '1000', 10);
const OUT_DIR = path.join(__dirname, 'out');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// ==== CFG — не трогаем ====
const CFG = {
  popSize: 24, matchesPerGenome: 2, elite: 4,
  mutRate: 0.05, mutPower: 0.25, tournament: 3, hofSize: 5,
  batchSize: 10, hofMatchRate: 0.2,
};

// ==== Пул воркеров с батчингом ====
const NUM_WORKERS = Math.max(1, os.cpus().length);
// Размер пачки. Меньше = лучше балансировка, больше = меньше IPC.
// 4 — хороший компромисс для popSize=24, K=2 (48 матчей за поколение).
const CHUNK_SIZE = 4;
console.log(`[pool] workers: ${NUM_WORKERS}, chunk: ${CHUNK_SIZE}`);

class Pool {
  constructor(size) {
    this.workers = [];
    this.queue = [];
    this.nextId = 0;
    for (let i = 0; i < size; i++) {
      const w = new Worker(path.join(__dirname, 'worker.js'));
      w.busy = false;
      w.on('message', (msg) => this._onDone(w, msg));
      this.workers.push(w);
    }
  }
  runBatch(tasks) {
    return new Promise((resolve) => {
      this.queue.push({ tasks, resolve });
      this._pump();
    });
  }
  _pump() {
    for (const w of this.workers) {
      if (w.busy) continue;
      const task = this.queue.shift();
      if (!task) return;
      w.busy = true;
      w.currentResolve = task.resolve;
      const id = this.nextId++;
      w.postMessage({ type: 'batch', batchId: id, tasks: task.tasks });
    }
  }
  _onDone(w, msg) {
    w.busy = false;
    const resolve = w.currentResolve;
    w.currentResolve = null;
    resolve(msg.results);
    this._pump();
  }
}

const pool = new Pool(NUM_WORKERS);

// ==== Эволюция ====
let popA = [], popB = [];
let hofA = [], hofB = [];
let generation = 0;
let bestGenomeA = null, bestGenomeB = null;
let bestFitA = -Infinity, bestFitB = -Infinity;
let cumWinsA = 0, cumWinsB = 0;

function initPopulations() {
  popA = []; popB = [];
  for (let i = 0; i < CFG.popSize; i++) popA.push(randomGenome());
  for (let i = 0; i < CFG.popSize; i++) popB.push(randomGenome());
  hofA = []; hofB = [];
  generation = 0;
  bestGenomeA = popA[0].slice();
  bestGenomeB = popB[0].slice();
  bestFitA = -Infinity; bestFitB = -Infinity;
  cumWinsA = 0; cumWinsB = 0;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function recordMatch(genome, fit, dmg, win, hp, explored, fitMap, statsMap, K, countWin) {
  if (!fitMap.has(genome)) {
    fitMap.set(genome, 0);
    statsMap.set(genome, { dmg: 0, wins: 0, hp: 0, explored: 0 });
  }
  fitMap.set(genome, fitMap.get(genome) + fit / K);
  const s = statsMap.get(genome);
  s.dmg += dmg / K;
  if (countWin) s.wins += win / K;
  s.hp += hp / K;
  s.explored += explored / K;
}

function crossover(a, b) {
  const c = new Float32Array(GENOME_SIZE);
  for (let i = 0; i < GENOME_SIZE; i++) c[i] = Math.random() < 0.5 ? a[i] : b[i];
  return c;
}
function mutate(g) {
  for (let i = 0; i < GENOME_SIZE; i++) {
    if (Math.random() < CFG.mutRate) {
      g[i] += (Math.random() * 2 - 1) * CFG.mutPower;
      if (g[i] > WEIGHT_CLAMP) g[i] = WEIGHT_CLAMP;
      else if (g[i] < -WEIGHT_CLAMP) g[i] = -WEIGHT_CLAMP;
    }
  }
}
function tournamentPick(scored) {
  let best = null;
  for (let i = 0; i < CFG.tournament; i++) {
    const p = scored[(Math.random() * scored.length) | 0];
    if (!best || p.fitness > best.fitness) best = p;
  }
  return best.genome;
}
function evolvePopulation(scored) {
  const next = [];
  for (let i = 0; i < Math.min(CFG.elite, scored.length); i++) {
    next.push(scored[i].genome.slice());
  }
  while (next.length < CFG.popSize) {
    const c = crossover(tournamentPick(scored), tournamentPick(scored));
    mutate(c);
    next.push(c);
  }
  return next;
}

async function doOneGeneration() {
  const fitMapA = new Map();
  const fitMapB = new Map();
  const statsA = new Map();
  const statsB = new Map();

  const K = CFG.matchesPerGenome;
  let winsA_gen = 0, winsB_gen = 0;

  const tasks = [];
  for (let round = 0; round < K; round++) {
    const orderA = shuffle([...popA]);
    const orderB = shuffle([...popB]);
    const N = Math.min(orderA.length, orderB.length);
    for (let i = 0; i < N; i++) {
      const gA = orderA[i];
      const gB = orderB[i];

      const isHofRound = (hofA.length > 0 && hofB.length > 0)
                      && Math.random() < CFG.hofMatchRate;

      if (!isHofRound) {
        tasks.push({ kind: 'normal', gA, gB });
      } else {
        const oppB = hofB[(Math.random() * hofB.length) | 0];
        const oppA = hofA[(Math.random() * hofA.length) | 0];
        tasks.push({ kind: 'hofA', gA, gB: oppB });
        tasks.push({ kind: 'hofB', gA: oppA, gB });
      }
    }
  }

  // Нарезаем на пачки и раздаём воркерам
  const chunks = [];
  for (let i = 0; i < tasks.length; i += CHUNK_SIZE) {
    chunks.push(tasks.slice(i, i + CHUNK_SIZE));
  }

  const chunkResults = await Promise.all(
    chunks.map(chunk => pool.runBatch(chunk.map(t => ({ gA: t.gA, gB: t.gB }))))
  );

  // Собираем обратно — порядок сохранён
  for (let ci = 0; ci < chunks.length; ci++) {
    const chunk = chunks[ci];
    const results = chunkResults[ci];
    for (let j = 0; j < chunk.length; j++) {
      const t = chunk[j];
      const r = results[j];
      if (t.kind === 'normal') {
        winsA_gen += r.winA;
        winsB_gen += r.winB;
        recordMatch(t.gA, r.fitA, r.dmgA, r.winA, r.hpA, r.exploredA, fitMapA, statsA, K, true);
        recordMatch(t.gB, r.fitB, r.dmgB, r.winB, r.hpB, r.exploredB, fitMapB, statsB, K, true);
      } else if (t.kind === 'hofA') {
        recordMatch(t.gA, r.fitA, r.dmgA, r.winA, r.hpA, r.exploredA, fitMapA, statsA, K, false);
      } else if (t.kind === 'hofB') {
        recordMatch(t.gB, r.fitB, r.dmgB, r.winB, r.hpB, r.exploredB, fitMapB, statsB, K, false);
      }
    }
  }

  const scoredA = popA.map(g => {
    const s = statsA.get(g) || { dmg: 0, wins: 0, hp: 0, explored: 0 };
    return { genome: g, fitness: fitMapA.get(g) ?? -1000, dmg: s.dmg, wins: s.wins, hp: s.hp, explored: s.explored };
  });
  const scoredB = popB.map(g => {
    const s = statsB.get(g) || { dmg: 0, wins: 0, hp: 0, explored: 0 };
    return { genome: g, fitness: fitMapB.get(g) ?? -1000, dmg: s.dmg, wins: s.wins, hp: s.hp, explored: s.explored };
  });

  scoredA.sort((a, b) => b.fitness - a.fitness);
  scoredB.sort((a, b) => b.fitness - a.fitness);

  if (scoredA[0].fitness > bestFitA) {
    bestFitA = scoredA[0].fitness;
    bestGenomeA = scoredA[0].genome.slice();
  }
  if (scoredB[0].fitness > bestFitB) {
    bestFitB = scoredB[0].fitness;
    bestGenomeB = scoredB[0].genome.slice();
  }

  if (CFG.hofSize > 0) {
    hofA.push(scoredA[0].genome.slice());
    hofB.push(scoredB[0].genome.slice());
    if (hofA.length > CFG.hofSize) hofA.shift();
    if (hofB.length > CFG.hofSize) hofB.shift();
  }

  cumWinsA += winsA_gen;
  cumWinsB += winsB_gen;

  popA = evolvePopulation(scoredA);
  popB = evolvePopulation(scoredB);

  generation++;
  return { winsA_gen, winsB_gen, bestA: scoredA[0].fitness, bestB: scoredB[0].fitness };
}

// ==== Сохранение ====
function saveGenome(genome, name, gen) {
  const data = {
    version: 1, type: 'single',
    savedAt: new Date().toISOString(),
    generation: gen,
    genome: Array.from(genome),
  };
  fs.writeFileSync(path.join(OUT_DIR, `${name}-gen${gen}.json`), JSON.stringify(data));
}

function saveHybrids(gen) {
  if (!bestGenomeA || !bestGenomeB) return;
  const hybrids = [];
  for (let k = 0; k < 20; k++) {
    const h = new Float32Array(GENOME_SIZE);
    for (let i = 0; i < GENOME_SIZE; i++) {
      h[i] = Math.random() < 0.5 ? bestGenomeA[i] : bestGenomeB[i];
      if (Math.random() < 0.03) h[i] += (Math.random() * 2 - 1) * 0.15;
    }
    hybrids.push(h);
  }
  const data = {
    version: 1, type: 'hybrids',
    savedAt: new Date().toISOString(),
    generation: gen, count: hybrids.length,
    genomes: hybrids.map(h => Array.from(h)),
  };
  fs.writeFileSync(path.join(OUT_DIR, `hybrids-AxB-gen${gen}.json`), JSON.stringify(data));
}

// ==== Main ====
async function main() {
  console.log(`[start] ${TOTAL_GENS} gens, ckpt every ${CKPT_EVERY}`);
  initPopulations();

  const t0 = Date.now();
  let lastReport = t0;

  for (let g = 1; g <= TOTAL_GENS; g++) {
    await doOneGeneration();

    if (g % CKPT_EVERY === 0) {
      saveGenome(bestGenomeA, 'genome-A', g);
      saveGenome(bestGenomeB, 'genome-B', g);
      saveHybrids(g);
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      const gps = (g / parseFloat(dt)).toFixed(2);
      console.log(`[ckpt] gen=${g} bestA=${bestFitA.toFixed(0)} bestB=${bestFitB.toFixed(0)} time=${dt}s (${gps} gen/s)`);
      lastReport = Date.now();
    } else if (Date.now() - lastReport > 30000) {
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      const gps = (g / parseFloat(dt)).toFixed(2);
      console.log(`[info] gen=${g}/${TOTAL_GENS} bestA=${bestFitA.toFixed(0)} bestB=${bestFitB.toFixed(0)} (${gps} gen/s)`);
      lastReport = Date.now();
    }
  }

  saveGenome(bestGenomeA, 'genome-A', TOTAL_GENS);
  saveGenome(bestGenomeB, 'genome-B', TOTAL_GENS);
  saveHybrids(TOTAL_GENS);
  console.log(`[done] ${TOTAL_GENS} gens in ${((Date.now() - t0)/1000).toFixed(1)}s`);
}

main().catch(err => { console.error(err); process.exit(1); });
