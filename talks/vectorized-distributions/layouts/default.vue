<script setup lang="ts">
import { computed } from 'vue'
import { useNav } from '@slidev/client'

declare const __EVENT__: string

const props = defineProps<{ eyebrow?: string; event?: boolean }>()
const eventName = __EVENT__
const label = computed(() => (props.event ? eventName : props.eyebrow))

// A section divider's eyebrow reads "03 / Reproducibility"; content slides use "03 · …".
const DIVIDER = /^(\d{2}) \/ (.+)$/
const { slides } = useNav()
const sections = computed(() =>
  slides.value.flatMap((slide) => {
    const match = String(slide.meta?.slide?.frontmatter?.eyebrow ?? '').match(DIVIDER)
    return match ? [{ number: match[1], name: match[2] }] : []
  }),
)
const current = computed(() => props.eyebrow?.match(DIVIDER)?.[1])
</script>

<template>
  <div class="slidev-layout default">
    <p v-if="label" class="eyebrow">{{ label }}</p>
    <slot />
    <ol v-if="current" class="agenda">
      <li
        v-for="section in sections"
        :key="section.number"
        :class="{ done: section.number < current, now: section.number === current }"
      >
        <span>{{ section.number }}</span>{{ section.name }}
      </li>
    </ol>
    <footer class="deck-footer">
      <span>polars-stats · {{ eventName }}</span>
      <span class="page">{{ String($page).padStart(2, '0') }}</span>
    </footer>
    <div class="deck-progress"><span :style="{ width: `${(100 * $page) / $nav.total}%` }" /></div>
  </div>
</template>
