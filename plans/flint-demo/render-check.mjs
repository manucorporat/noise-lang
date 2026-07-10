// Headless verification: same pipeline as the browser demo, rendered to SVG in Node.
import { readFileSync, writeFileSync } from 'node:fs';
import * as vegaLite from 'vega-lite';
import * as vega from 'vega';
import { payloadToVegaLite } from './mapper.mjs';

const log = JSON.parse(readFileSync(new URL('./noise-log.json', import.meta.url)));

let i = 0;
for (const item of log) {
  if (item.kind !== 'plot') continue;
  const out = payloadToVegaLite(item.plot);
  if (!out) continue;
  const vgSpec = vegaLite.compile(out.spec).spec;
  const view = new vega.View(vega.parse(vgSpec), { renderer: 'none' });
  const svg = await view.toSVG();
  const name = `chart-${i++}-${item.plot.type}.svg`;
  writeFileSync(new URL('./' + name, import.meta.url), svg);
  console.log('OK', name, out.title, `${(svg.length / 1024).toFixed(0)}kB`);
}
