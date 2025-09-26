# Requirements for Autonomous Delivery Agent Project

## System Requirements

- Python 3.7 or higher
- Operating System: Windows, macOS, or Linux

## Python Dependencies

Install the following packages using pip:

```bash
pip install numpy matplotlib pandas seaborn
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### Core Dependencies:
- **numpy**: For numerical computations and array operations
- **matplotlib**: For visualization and plotting
- **pandas**: For data manipulation and analysis
- **seaborn**: For enhanced statistical visualizations

### Standard Library Dependencies (included with Python):
- heapq (priority queue for A*)
- random (for random restarts and map generation)
- time (for performance measurement)
- json (for map file handling)
- argparse (for command-line interface)
- copy (for deep copying objects)
- typing (for type hints)
- dataclasses (for structured data)
- enum (for enumerations)
- os (for file operations)

## Installation Instructions

1. **Clone/Download the project files**
   ```bash
   git clone <repository-url>
   cd autonomous-delivery-agent
   ```

2. **Install Python dependencies**
   ```bash
   pip install numpy matplotlib pandas seaborn
   ```

3. **Verify installation**
   ```bash
   python delivery_agent.py --help
   ```

## Running the Project

### Basic Usage:
```bash
# Run BFS on small map
python delivery_agent.py maps/small_map.json bfs

# Run A* on medium map with visualization
python delivery_agent.py maps/medium_map.json astar --visualize

# Run dynamic replanning on dynamic map with verbose output
python delivery_agent.py maps/dynamic_map.json dynamic --verbose --visualize
```

### Running Experiments:
```bash
# Run all experiments and generate report
python experiment_runner.py
```

### Available Algorithms:
- `bfs`: Breadth-First Search
- `ucs`: Uniform Cost Search
- `astar`: A* Search with Manhattan distance heuristic
- `hill_climbing`: Hill Climbing with random restarts
- `dynamic`: Dynamic A* with replanning

### Available Test Maps:
- `maps/small_map.json`: 8x8 grid with basic obstacles
- `maps/medium_map.json`: 15x15 grid with varied terrain costs
- `maps/large_map.json`: 25x25 grid for scalability testing
- `maps/dynamic_map.json`: 12x12 grid with moving obstacles

## File Structure

```
autonomous-delivery-agent/
├── delivery_agent.py          # Main implementation
├── experiment_runner.py       # Experimental evaluation
├── requirements.md           # This file
├── README.md                # Project documentation
├── maps/                    # Test map files
│   ├── small_map.json
│   ├── medium_map.json
│   ├── large_map.json
│   └── dynamic_map.json
└── output files (generated):
    ├── experimental_results.csv
    ├── results_summary.csv
    ├── algorithm_performance_comparison.png
    └── time_vs_nodes_comparison.png
```

## Troubleshooting

### Common Issues:

1. **Import Error for matplotlib/pandas/seaborn**
   - Solution: `pip install matplotlib pandas seaborn`

2. **Map file not found**
   - Solution: Run the program once to generate test maps, or check file paths

3. **Permission denied when creating directories**
   - Solution: Run with appropriate permissions or create `maps/` directory manually

4. **Visualization not showing**
   - Solution: Ensure you have a display/GUI environment. Use `--visualize` flag only in GUI environments.

### Performance Notes:

- Large maps may take several minutes to complete with exhaustive search algorithms
- Hill climbing may not always find optimal solutions due to local optima
- Dynamic replanning requires moving obstacles to demonstrate replanning capability

## Testing

The project includes built-in testing through the experiment runner:
```bash
python experiment_runner.py
```

This will:
1. Create test maps if they don't exist
2. Run all algorithms on all maps
3. Generate performance comparisons
4. Create visualizations
5. Provide analysis and recommendations

## Hardware Recommendations

- **Minimum**: 4GB RAM, modern CPU
- **Recommended**: 8GB+ RAM for large map experiments
- **Storage**: ~100MB for project files and outputs

## Additional Notes

- All algorithms are implemented from scratch without external pathfinding libraries
- The project supports both 4-connected movement (up/down/left/right)
- Heuristics are admissible (Manhattan distance for A*)
- Dynamic replanning demonstrates real-time obstacle avoidance
- Results are automatically saved in CSV format for analysis
