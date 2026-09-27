// worker.js — один воркер, гоняет матчи
const { parentPort } = require('worker_threads');
const { generateRandomMap, simulateNNvsNN } = require('./shared');

parentPort.on('message', (msg) => {
  if (msg.type === 'match') {
    const mapRef = generateRandomMap();
    const result = simulateNNvsNN(msg.gA, msg.gB, mapRef);
    parentPort.postMessage({ id: msg.id, result });
  }
});
