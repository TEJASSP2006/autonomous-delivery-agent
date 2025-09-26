# Usage Guide - Autonomous Delivery Agent

## Quick Start

### Installation
```bash
pip install numpy matplotlib pandas seaborn


Basic Usage
bash# Run BFS on small map
python delivery_agent.py maps/small_map.json bfs

# Run A* with visualization
python delivery_agent.py maps/medium_map.json astar --visualize

# Run dynamic replanning demo
python delivery_agent.py maps/dynamic_map.json dynamic --verbose --visualize
Complete Experimental Evaluation
bashpython experiment_runner.py
Dynamic Replanning Demonstration
bashpython demo_replanning.py
Command Line Options

--visualize: Show graphical visualization of the solution
--verbose: Display detailed algorithm execution information

Output Files Generated

experimental_results.csv: Raw experimental data
results_summary.csv: Summary statistics
*.png: Visualization plots
demo_report.txt: Demonstration summary

Map File Format
Maps are stored in JSON format with:

width, height: Grid dimensions
start, goal: Start and goal positions [x, y]
obstacles: List of obstacle positions
terrain_costs: 2D array of movement costs
moving_obstacles: List of moving obstacle paths

Troubleshooting

Import errors: Install required packages with pip
File not found: Ensure you're in the correct directory
Visualization issues: Use --visualize flag only in GUI environments



