<script setup lang="ts">
const CONTRACTS = ['reproducibility', 'broadcast', 'parameter validation', 'accuracy']

// `summary` shows every filled pin alike, with none marked as the latest.
withDefaults(defineProps<{ pins?: number, mini?: boolean, summary?: boolean }>(), {
  pins: 0,
  mini: false,
  summary: false,
})
</script>

<template>
  <div class="stack" :class="{ mini }">
    <div class="stack-main">
      <div v-if="!mini" class="giant"><b>Polars</b><span>the engine: lazy, optimised, streaming, every core</span></div>
      <div class="strip">
        <b>polars-stats</b>
        <div class="pins">
          <span v-for="(c, i) in CONTRACTS" :key="c" class="pin" :class="{ on: i < pins, latest: !summary && i === pins - 1 }">
            {{ i < pins ? c : '?' }}
          </span>
        </div>
      </div>
      <div v-if="!mini" class="giant"><b>statrs</b><span>the maths: one distribution, one value at a time</span></div>
    </div>
    <div v-if="!mini" class="side"><b>scipy</b><span>the vocabulary, and the ruler</span></div>
  </div>
</template>
