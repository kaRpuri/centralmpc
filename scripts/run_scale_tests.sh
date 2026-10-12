#!/usr/bin/env bash
# Measure QP solve time and solve success against the number of agents.
# Writes assets/scaling.png and assets/scaling.json.
set -euo pipefail

DURATION="${DURATION:-10}"
CONFIGS="${CONFIGS:-scale_4_agents scale_8_agents scale_16_agents scale_32_agents scale_64_agents}"

centralmpc scale --configs ${CONFIGS} --duration "${DURATION}" --output assets/scaling.png
