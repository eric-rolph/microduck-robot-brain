#!/usr/bin/env bash
set -euo pipefail

# Housekeeping cores bitmask: Cores 0-3, 5-7 = 0xEF (binary: 11101111)
# Bit 4 is set to 0 to exclude Core 4 from receiving hardware interrupts.
HOUSEKEEPING_MASK="ef"

# Direct all non-pinned hardware IRQs away from Core 4
for affinity in /proc/irq/*/smp_affinity; do
    if [ -f "$affinity" ]; then
        echo "$HOUSEKEEPING_MASK" > "$affinity" 2>/dev/null || true
    fi
done

# Route kernel unbound workqueues away from Core 4
echo "$HOUSEKEEPING_MASK" > /sys/devices/virtual/workqueue/cpumask 2>/dev/null || true

echo "Core 4 shielded from hardware IRQs and unbound workqueues."
