// worker.js — принимает пачку матчей, возвращает пачку результатов
const { parentPort } = require('worker_threads');
const { generateRandomMap, simulateNNvsNN } = require('./shared');

parentPort.on('message', (msg) => {
  if (msg.type === 'batch') {
    const results = new Array(msg.tasks.length);
    for (let i = 0; i < msg.tasks.length; i++) {
      const t = msg.tasks[i];
      const mapRef = generateRandomMap();
      results[i] = simulateNNvsNN(t.gA, t.gB, mapRef);
    }
    parentPort.postMessage({ batchId: msg.batchId, results });
  }
});
