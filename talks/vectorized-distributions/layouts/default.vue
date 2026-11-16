<script setup lang="ts">
import { computed } from 'vue'

declare const __EVENT__: string

const props = defineProps<{ eyebrow?: string; event?: boolean }>()
const eventName = __EVENT__
const label = computed(() => (props.event ? eventName : props.eyebrow))
</script>

<template>
  <div class="slidev-layout default">
    <p v-if="label" class="eyebrow">{{ label }}</p>
    <slot />
    <footer class="deck-footer">
      <span>polars-stats · {{ eventName }}</span>
      <span class="page">{{ String($page).padStart(2, '0') }}</span>
    </footer>
    <div class="deck-progress"><span :style="{ width: `${(100 * $page) / $nav.total}%` }" /></div>
  </div>
</template>
