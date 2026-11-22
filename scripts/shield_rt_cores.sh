#!/usr/bin/env bash
set -euo pipefail

# Shield the isolated RT core (default: Core 3 on 4-core Raspberry Pi CM4)
# Usage: sudo ./shield_rt_cores.sh [rt_core_index]
NUM_CPUS=$(nproc)
RT_CORE="${1:-$((NUM_CPUS - 1))}"

if [ "$RT_CORE" -lt 0 ] || [ "$RT_CORE" -ge "$NUM_CPUS" ]; then
    echo "Error: RT_CORE=$RT_CORE out of range for $NUM_CPUS CPUs (0..$((NUM_CPUS - 1)))" >&2
    exit 1
fi

# Compute housekeeping bitmask with bit RT_CORE cleared (e.g. 4-core CM4, core 3 -> 0x7)
ALL_MASK=$(( (1 << NUM_CPUS) - 1 ))
HOUSEKEEPING_DEC=$(( ALL_MASK & ~(1 << RT_CORE) ))
HOUSEKEEPING_MASK=$(printf "%x" "$HOUSEKEEPING_DEC")

# Direct all non-pinned hardware IRQs away from the RT core
for affinity in /proc/irq/*/smp_affinity; do
    if [ -f "$affinity" ]; then
        echo "$HOUSEKEEPING_MASK" > "$affinity" 2>/dev/null || true
    fi
done

# Route kernel unbound workqueues away from the RT core
echo "$HOUSEKEEPING_MASK" > /sys/devices/virtual/workqueue/cpumask 2>/dev/null || true

echo "Core $RT_CORE shielded from hardware IRQs and unbound workqueues (housekeeping mask: 0x$HOUSEKEEPING_MASK)."
