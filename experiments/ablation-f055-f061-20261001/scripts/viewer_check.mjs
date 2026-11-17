// Inspeção visual dos mapas finalistas no ContextMap2 Viewer, por CDP (Chromium headless).
//
// Uso: node viewer_check.mjs <url-do-viewer> <dir-de-saída> <map_id> [<map_id> ...]
//
// Para cada mapa: abre o viewer, escolhe o mapa no seletor, liga a coloração por label, captura o
// viewport; abre a lista de entidades, seleciona a primeira entidade cujo registro cita "door" e
// captura o inspetor; abre a aba de evidência da seleção e captura; liga a camada de relações e
// captura. Os passos procuram controles pelo texto visível, então quebram (com erro) se a UI mudar,
// em vez de capturar a tela errada em silêncio.
import { spawn } from 'node:child_process'
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs'
const metrics = JSON.parse(readFileSync(new URL('../results/metrics.json', import.meta.url)))

const [url, outDir, ...maps] = process.argv.slice(2)
mkdirSync(outDir, { recursive: true })
const port = 9333
const browser = spawn('chromium', ['--headless=new', '--disable-gpu', '--no-sandbox', '--hide-scrollbars',
  `--remote-debugging-port=${port}`, '--window-size=1600,1000', 'about:blank'], { stdio: 'ignore' })
const sleep = ms => new Promise(r => setTimeout(r, ms))

async function target() {
  for (let i = 0; i < 50; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
      const page = list.find(t => t.type === 'page')
      if (page) return page.webSocketDebuggerUrl
    } catch { /* ainda subindo */ }
    await sleep(200)
  }
  throw new Error('chromium did not expose a page target')
}

const ws = new WebSocket(await target())
await new Promise(r => ws.addEventListener('open', r, { once: true }))
let id = 0
const waiting = new Map()
ws.addEventListener('message', event => {
  const message = JSON.parse(event.data)
  if (message.id && waiting.has(message.id)) { waiting.get(message.id)(message); waiting.delete(message.id) }
})
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const n = ++id
  waiting.set(n, m => m.error ? reject(new Error(`${method}: ${m.error.message}`)) : resolve(m.result))
  ws.send(JSON.stringify({ id: n, method, params }))
})
const evaluate = async expression => {
  const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description ?? 'evaluate failed')
  return result.result.value
}
const shot = async name => {
  const { data } = await send('Page.captureScreenshot', { format: 'png' })
  writeFileSync(`${outDir}/${name}.png`, Buffer.from(data, 'base64'))
}
// Clique real (mousePressed/mouseReleased) no centro do elemento que `finder` (JS) devolve: os
// controles do Nuxt UI/Reka abrem com eventos de ponteiro, não com element.click().
const clickNode = async (finder, description) => {
  const rect = await evaluate(`(() => {
    const node = (${finder})()
    if (!node) return null
    node.scrollIntoView({ block: 'center' })
    const r = node.getBoundingClientRect()
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 }
  })()`)
  if (!rect) throw new Error(`no visible control: ${description}`)
  for (const type of ['mouseMoved', 'mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: rect.x, y: rect.y, button: 'left', clickCount: 1 })
  }
  await sleep(1200)
}
// Clica no menor elemento visível cujo texto contém (ou é igual a) `text`.
const clickText = (text, { exact = false, prefix = false } = {}) => clickNode(`() => {
  const want = ${JSON.stringify(text.toLowerCase())}
  return [...document.querySelectorAll('button, [role=option], [role=tab], a, li, label, span, div')]
    .filter(n => { const t = (n.innerText || '').trim().toLowerCase(); return ${exact} ? t === want : ${prefix} ? t.startsWith(want) : t.includes(want) })
    .filter(n => n.offsetParent !== null || n.getClientRects().length)
    .sort((a, b) => (a.innerText || '').length - (b.innerText || '').length)[0]
}`, `text ${JSON.stringify(text)}`)
const mapShown = () => evaluate(`(document.querySelector('[aria-label="Mapa disponível"]')?.innerText || '').trim()`)

await send('Page.enable'); await send('Runtime.enable')
const report = {}
for (const map of maps) {
  const short = map.replace('corridor02-f055-f061-', '').replace('-vox10', '')
  report[map] = { steps: [] }
  const step = async (name, action) => {
    try { await action(); report[map].steps.push({ name, ok: true }) }
    catch (error) { report[map].steps.push({ name, ok: false, error: String(error.message) }) }
  }
  await send('Page.navigate', { url })
  await sleep(9000)
  // O seletor de mapa mostra o rótulo do mapa atual; abrir e escolher pelo map_id.
  await step('select map', async () => {
    if (!(await mapShown()).startsWith(map)) {
      await clickNode(`() => document.querySelector('[aria-label="Mapa disponível"]')`, 'map selector')
      await clickNode(`() => [...document.querySelectorAll('[role=option]')].find(o => o.innerText.trim().startsWith(${JSON.stringify(map + ' ·')}))`, `option ${map}`)
    }
    await sleep(9000)
    const shown = await mapShown()
    report[map].selectorText = shown
    if (!shown.startsWith(map)) throw new Error(`selector shows ${shown}`)
  })
  const combobox = text => `() => [...document.querySelectorAll('[role=combobox], button')].find(b => (b.innerText || '').trim() === ${JSON.stringify(text)} && b.closest('.inspector-section, details, aside, section'))`
  await step('focus geometry', async () => { await clickText('Focar geometria', { exact: true }); await sleep(3000) })
  await step('geometry', () => shot(`${short}-1-geometry`))
  await step('semantic labels mode', async () => {
    await clickNode(combobox('Geometria'), 'mode selector')
    await clickNode(`() => [...document.querySelectorAll('[role=option]')].find(o => o.innerText.trim() === 'Labels semânticas')`, 'option Labels semânticas')
    await sleep(4000)
    await shot(`${short}-2-labels`)
  })
  await step('relations layer', async () => {
    await clickNode(`() => [...document.querySelectorAll('button')].find(b => (b.innerText || '').trim().startsWith('Relações') && b.closest('aside, nav, .layers, div'))`, 'relations layer')
    await sleep(4000)
    await shot(`${short}-3-relations`)
  })
  await step('select door entity', async () => {
    await evaluate(`(() => { const d = [...document.querySelectorAll('details')].find(d => d.querySelector('summary')?.innerText.toLowerCase().includes('registros persistidos')); if (d) d.open = true })()`)
    const entity = await evaluate(`(async () => {
      const maps = await (await fetch('/api/v1/maps')).json()
      const entry = maps.find(m => m.summary.sourceLabel === ${JSON.stringify(map)})
      const ins = await (await fetch('/api/v1/maps/' + entry.summary.contextMapId + '/inspection')).json()
      const doors = ins.entities.filter(e => e.semanticHypotheses.some(h => (h.label || '').includes('door')))
        .sort((a, b) => (a.semanticHypotheses.length - b.semanticHypotheses.length) || (b.geometryReferenceCount - a.geometryReferenceCount))
      const pick = doors[0] || ins.entities[0]
      return pick ? { id: pick.entityId, labels: pick.semanticHypotheses.map(h => h.label), refs: pick.geometryReferenceCount } : null
    })()`)
    if (!entity) throw new Error('map has no entity to select')
    report[map].selectedEntity = entity
    const wanted = JSON.stringify(entity.id)
    for (let i = 0; i < 60; i++) {
      if (await evaluate(`[...document.querySelectorAll('button, a')].some(b => (b.textContent || '').trim() === ${wanted})`)) break
      await sleep(500)
    }
    await clickNode(`() => [...document.querySelectorAll('button, a')].filter(b => (b.textContent || '').trim() === ${wanted})[0]`, `entity ${entity.id}`)
    await sleep(3000)
    await clickText('Focar entidade ou relação em 3D', { exact: true })
    await sleep(4000)
    await shot(`${short}-4-entity`)
    const text = await evaluate('document.body.innerText')
    const at = text.indexOf(entity.id + '\n', text.indexOf('RELAÇÕES') + 1)
    report[map].entityDetail = text.slice(Math.max(0, at - 200), at + 1500)
  })
  await step('evidence of the right door', async () => {
    const match = metrics[map]?.frames?.['000055']?.matches?.['door-right']
    if (!match) throw new Error('no region of this arm matches door-right in frame 55')
    await clickNode(`() => [...document.querySelectorAll('button')].find(b => (b.innerText || '').trim().startsWith('Evidência') && b.getAttribute('role') !== 'tab')`, 'evidence layer')
    await clickNode(`() => [...document.querySelectorAll('[role=tab], button')].find(b => (b.innerText || '').trim().toLowerCase() === 'evidência' && b.getAttribute('role') === 'tab') || [...document.querySelectorAll('button')].filter(b => (b.innerText || '').trim().toLowerCase() === 'evidência')[1]`, 'evidence tab')
    const row = `camera_1_image_raw-000055 · ${match.region_id} ·`
    for (let i = 0; i < 40; i++) {
      if (await evaluate(`[...document.querySelectorAll('button')].some(b => (b.innerText || '').trim().startsWith(${JSON.stringify(row)}))`)) break
      // A lista mostra 50 observações por vez: pede mais até a linha aparecer.
      await evaluate(`(() => { const more = [...document.querySelectorAll('button')].find(b => (b.innerText || '').trim().toLowerCase() === 'mostrar mais'); if (more) more.click() })()`)
      await sleep(700)
    }
    await clickNode(`() => [...document.querySelectorAll('button')].find(b => (b.innerText || '').trim().startsWith(${JSON.stringify(row)}))`, `evidence row ${row}`)
    await sleep(6000)
    await shot(`${short}-5-evidence`)
    const text = await evaluate('document.body.innerText')
    const at = text.indexOf('EVIDÊNCIA VISUAL')
    report[map].evidencePanel = text.slice(at, at + 2500)
    await clickText('focar suporte da região em 3D', { exact: true })
    await sleep(4000)
    await shot(`${short}-6-evidence-support`)
  })
  report[map].visibleText = (await evaluate('document.body.innerText')).slice(0, 4000)
}
writeFileSync(`${outDir}/viewer-check.json`, JSON.stringify(report, null, 2))
ws.close(); browser.kill()
console.log(JSON.stringify(Object.fromEntries(Object.entries(report).map(([k, v]) => [k, v.steps])), null, 1))
