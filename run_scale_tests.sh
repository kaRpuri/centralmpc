#!/bin/bash

# Centralized MPC Scalability Test Script
# Run scalability tests for centralized MPC with different agent counts

echo "================================================="
echo "Centralized MPC Scalability Test"
echo "================================================="

# Test all agent counts up to 64
echo "Testing all agent counts (4, 8, 16, 32, 64)..."
python3 run_scenario_scale_central.py --agents 4 8 16 32 64 --runs 1 2 3

echo ""
echo "================================================="
echo "Quick Test (4 agents only)"
echo "================================================="
# Quick test with just 4 agents
# python3 run_scenario_scale_central.py --agents 4 --runs 1

echo ""
echo "================================================="
echo "Test up to specific agent count"
echo "================================================="
# Test up to 16 agents
# python3 run_scenario_scale_central.py --max-agents 16 --runs 1

echo ""
echo "================================================="
echo "Single run test"
echo "================================================="
# Single run test for quick validation
# python3 run_scenario_scale_central.py --agents 4 8 --runs 1

echo ""
echo "Results will be saved in experiments/ directory:"
echo "  - scale_4_agents_central/"
echo "  - scale_8_agents_central/"
echo "  - scale_16_agents_central/"
echo "  - scale_32_agents_central/"
echo "  - scale_64_agents_central/"
echo ""
echo "Each directory contains run_1/, run_2/, run_3/ subdirectories"
echo "with trajectories.txt, goals.txt, and config.json files"