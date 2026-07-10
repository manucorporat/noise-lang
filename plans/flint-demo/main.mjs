// Browser entry: real Noise plot payloads (embedded at build time) -> Flint -> Vega-Lite -> render.
import embed from 'vega-embed';
import { payloadToVegaLite } from './mapper.mjs';
import log from './noise-log.json';

const root = document.getElementById('charts');

for (const item of log) {
  if (item.kind === 'text') {
    const pre = document.createElement('pre');
    pre.className = 'stdout';
    pre.textContent = item.text ?? '';
    root.appendChild(pre);
    continue;
  }
  const out = payloadToVegaLite(item.plot);
  if (!out) continue;
  const card = document.createElement('div');
  card.className = 'card';
  const h = document.createElement('h3');
  h.textContent = out.title;
  const mount = document.createElement('div');
  card.append(h, mount);
  root.appendChild(card);
  embed(mount, out.spec, { actions: false }).catch(e => {
    mount.textContent = 'render error: ' + e.message;
  });
}
