// Run through npx, Slidev lives in the npx cache, and its dev server serves only @slidev/cli and @slidev/client
// from there: KaTeX's fonts, one folder over, are refused and the formulas fall back to system fonts.
// Allow the node_modules Slidev itself runs from. Plain object, no `vite` import: this folder has no node_modules.
import { realpathSync } from 'node:fs'
import { sep } from 'node:path'

const cli = realpathSync(process.argv[1])
const marker = `${sep}node_modules${sep}`
const at = cli.indexOf(marker)

export default {
  server: { fs: { allow: at === -1 ? [] : [cli.slice(0, at + marker.length - 1)] } },
}
