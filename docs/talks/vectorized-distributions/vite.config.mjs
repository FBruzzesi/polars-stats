// Run through npx, Slidev lives in the npx cache, and its dev server serves only @slidev/cli and @slidev/client
// from there: KaTeX's fonts, one folder over, are refused and the formulas fall back to system fonts.
// Allow the node_modules Slidev itself runs from. Plain object, no `vite` import: this folder has no node_modules.
import { realpathSync } from 'node:fs'
import { sep } from 'node:path'

const cli = realpathSync(process.argv[1])
const marker = `${sep}node_modules${sep}`
const at = cli.indexOf(marker)

// The event names the cover eyebrow and the footer; pick one with EVENT=<key>, Paris when unset.
const events = {
  'compute-paris': 'compute! Paris 2026',
  'pydata-milan': 'PyData Milan · 19 November 2026',
}
const event = process.env.EVENT ?? 'compute-paris'
if (!Object.hasOwn(events, event)) {
  throw new Error(`EVENT=${event} is not one of: ${Object.keys(events).join(', ')}`)
}

export default {
  define: { __EVENT__: JSON.stringify(events[event]) },
  server: { fs: { allow: at === -1 ? [] : [cli.slice(0, at + marker.length - 1)] } },
}
