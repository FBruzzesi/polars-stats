<script setup lang="ts">
defineProps<{ eyebrow?: string }>()
</script>

<template>
  <div class="slidev-layout default">
    <p v-if="eyebrow" class="eyebrow">{{ eyebrow }}</p>
    <slot />
    <footer class="deck-footer">
      <span>polars-stats · compute! Paris 2026</span>
      <span class="page">{{ String($page).padStart(2, '0') }}</span>
    </footer>
    <div class="deck-progress"><span :style="{ width: `${(100 * $page) / $nav.total}%` }" /></div>
  </div>
</template>
