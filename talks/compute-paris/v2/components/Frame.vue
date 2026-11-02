<script setup lang="ts">
// The six tutorial readings; every derived value was recorded from a polars-stats run.
const ROWS: Record<string, string>[] = [
  { sensor: 'temp-a', reading: '9.8', mu: '10.0', sigma: '0.5', z: '-0.4', upper_tail: '0.655422', log_tail: '-0.422476', simulated: '9.763219' },
  { sensor: 'temp-a', reading: '10.4', mu: '10.0', sigma: '0.5', z: '0.8', upper_tail: '0.211855', log_tail: '-1.551851', simulated: '10.825292' },
  { sensor: 'temp-a', reading: '40.0', mu: '10.0', sigma: '0.5', z: '60.0', upper_tail: '0.0', log_tail: '-1805.013561', simulated: '11.172469' },
  { sensor: 'temp-b', reading: '99.4', mu: '100.0', sigma: '2.0', z: '-0.3', upper_tail: '0.617911', log_tail: '-0.48141', simulated: '98.532788' },
  { sensor: 'temp-b', reading: '101.2', mu: '100.0', sigma: '2.0', z: '0.6', upper_tail: '0.274253', log_tail: '-1.293704', simulated: '100.147242' },
  { sensor: 'temp-b', reading: '250.0', mu: '100.0', sigma: '2.0', z: '75.0', upper_tail: '0.0', log_tail: '-2817.736604', simulated: '97.717269' },
]
const FAULTS = new Set([2, 5])

withDefaults(defineProps<{ cols: string[], fresh?: string, flag?: string }>(), {
  fresh: undefined,
  flag: undefined,
})
</script>

<template>
  <table class="frame">
    <thead>
      <tr><th v-for="c in cols" :key="c" :class="{ fresh: c === fresh }">{{ c }}</th></tr>
    </thead>
    <tbody>
      <tr v-for="(row, i) in ROWS" :key="i">
        <td v-for="c in cols" :key="c" :class="{ flag: c === flag && FAULTS.has(i), fresh: c === fresh }">{{ row[c] }}</td>
      </tr>
    </tbody>
  </table>
</template>
