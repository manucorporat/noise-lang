// Run real Noise programs through the freshly-built wasm engine and dump the
// plot payloads (IntrospectionOut JSON) that the playground would receive.
import { readFileSync, writeFileSync } from 'node:fs';
import init, { run } from '/Users/manumtzalmeida/repos/noise-lang/packages/core/wasm/noise.js';

const wasmBytes = readFileSync('/Users/manumtzalmeida/repos/noise-lang/packages/core/wasm/noise_bg.wasm');
await init(wasmBytes);

const PROGRAM = `
# Barrier option: 52-week GBM, knock-out barrier, payoff distribution.
use rand; use math; use vec;
s0 = 100; k = 100; barrier = 75;
mu = 0.0008; sigma = 0.025;
zs ~[52] normal(0, 1);
logrets = mu - sigma^2/2 + sigma * zs;
path = s0 * exp(cumsum(logrets));
knocked = any(path < barrier);
st = path[51];
vanilla = if st > k { st - k } else { 0 };
payoff = if knocked { 0 } else { vanilla };
plot::fan(path);
plot::hist(st);
plot::scatter(st, payoff);
Print("E(payoff) =", E(payoff))
`;

const res = JSON.parse(run(PROGRAM));
if (!res.ok) { console.error('Noise error:', res.error); process.exit(1); }
console.log('output:', res.output.trim());
console.log('log kinds:', res.log.map(l => l.kind === 'text' ? 'text' : `plot:${l.plot.type}`).join(', '));
writeFileSync(new URL('./noise-log.json', import.meta.url), JSON.stringify(res.log, null, 1));
console.log('wrote noise-log.json');
