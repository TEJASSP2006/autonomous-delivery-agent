# Autonomous Delivery Agent

**Author:** Tejas Santosh Paithankar
**REG.NO:** 24BCY10104
**Course:** CSA2001 - Fundamentals of AI and ML  
**Institution:** VIT BHOPAL UNIVERSITY  
**Date:** September 2025  
**GitHub Repository:** https://github.com/TEJASSP2006/autonomous-delivery-agent

An intelligent delivery agent that navigates 2D grid environments using various pathfinding algorithms to efficiently deliver packages while handling static obstacles, varying terrain costs, and dynamic moving obstacles.

## 🎯 Project Overview

This project implements a rational autonomous delivery agent capable of:
- Navigating complex 2D grid environments
- Handling static obstacles and varying terrain costs
- Adapting to dynamic moving obstacles through replanning
- Maximizing delivery efficiency under time and fuel constraints
- Comparing multiple pathfinding algorithms experimentally

## 🚀 Features

### Implemented Algorithms
- **Breadth-First Search (BFS)**: Uninformed search guaranteeing shortest path in unweighted grids
- **Uniform Cost Search (UCS)**: Optimal search for weighted grids
- **A* Search**: Informed search with Manhattan distance heuristic
- **Hill Climbing with Random Restarts**: Local search for comparison
- **Dynamic A* with Replanning**: Real-time adaptation to changing environments

### Environment Modeling
- **Static Obstacles**: Impassable cells in the grid
- **Terrain Costs**: Variable movement costs (1-5 units per cell)
- **Moving Obstacles**: Dynamic obstacles with predefined or unpredictable movement patterns
- **4-Connected Movement**: Agent moves up/down/left/right (no diagonal movement)

### Performance Metrics
- Path cost optimization
- Nodes expanded (computational efficiency)
- Execution time measurement
- Success rate analysis
- Scalability assessment

## 📁 Project Structure

```
autonomous-delivery-agent/
├── delivery_agent.py          # Core implementation with all algorithms
├── experiment_runner.py       # Automated experimental evaluation
├── requirements.md           # Installation and setup instructions
├── README.md                # This documentation file
├── maps/                    # Test environments
│   ├── small_map.json       # 8x8 grid - basic testing
│   ├── medium_map.json      # 15x15 grid - varied terrain
│   ├── large_map.json       # 25x25 grid - scalability test
│   └── dynamic_map.json     # 12x12 grid - moving obstacles
└── results/ (generated)
    ├── experimental_results.csv
    ├── results_summary.csv
    └── performance_plots.png
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Quick Start
```bash
# 1. Clone/download the project
git clone <repository-url>
cd autonomous-delivery-agent

# 2. Install dependencies
pip install numpy matplotlib pandas seaborn

# 3. Generate test maps and verify setup
python delivery_agent.py maps/small_map.json astar --visualize
```

For detailed installation instructions, see [requirements.md](requirements.md).

## 🎮 Usage Examples

### Command Line Interface

```bash
# Basic pathfinding
python delivery_agent.py maps/small_map.json bfs

# A* with visualization
python delivery_agent.py maps/medium_map.json astar --visualize

# Dynamic replanning with detailed output
python delivery_agent.py maps/dynamic_map.json dynamic --verbose --visualize

# Hill climbing local search
python delivery_agent.py maps/large_map.json hill_climbing --verbose
```

### Experimental Evaluation

```bash
# Run comprehensive experiments on all algorithms and maps
python experiment_runner.py
```

This generates:
- Performance comparison tables
- Statistical analysis
- Visualization plots  
- Algorithm recommendations

### Algorithm Options
- `bfs` - Breadth-First Search
- `ucs` - Uniform Cost Search  
- `astar` - A* Search
- `hill_climbing` - Hill Climbing with Random Restarts
- `dynamic` - Dynamic A* with Replanning

## 📊 Experimental Results

### Test Environments

| Map | Size | Obstacles | Terrain | Moving Obstacles | Purpose |
|-----|------|-----------|---------|------------------|---------|
| Small | 8×8 | Static | Uniform | None | Basic algorithm testing |
| Medium | 15×15 | Static | Varied (1-3 cost) | None | Terrain cost evaluation |
| Large | 25×25 | Complex | Varied (1-5 cost) | None | Scalability analysis |
| Dynamic | 12×12 | Mixed | Uniform | 2 moving | Replanning demonstration |

### Performance Comparison

Based on experimental evaluation:

**Path Optimality** (lower cost = better):
1. A* Search - Optimal paths with minimal computation
2. Uniform Cost Search - Optimal but more expensive
3. BFS - Optimal for uniform costs only
4. Dynamic A* - Near-optimal with replanning overhead
5. Hill Climbing - Suboptimal due to local search nature

**Computational Efficiency** (speed):
1. BFS - Fastest for small uniform grids
2. A* - Best balance of speed and optimality
3. Hill Climbing - Fast but unreliable
4. UCS - Slower due to exhaustive search
5. Dynamic A* - Slowest due to replanning

**Recommended Use Cases**:
- **Optimal paths needed**: Use A* Search
- **Real-time/dynamic environments**: Use Dynamic A* 
- **Simple uniform grids**: Use BFS
- **Memory constraints**: Use Hill Climbing (with caveats)

## 🧠 Algorithm Analysis

### A* Search Excellence
- **Admissible Heuristic**: Manhattan distance never overestimates
- **Optimal Solution**: Guaranteed shortest path
- **Efficient**: Minimal node expansion compared to uninformed search
- **Scalable**: Performance degrades gracefully with map size

### Dynamic Replanning Capability
- **Obstacle Detection**: Real-time identification of blocked paths
- **Replanning Trigger**: Automatic recalculation when needed
- **Logging**: Detailed replanning event tracking
- **Adaptability**: Handles unpredictable moving obstacles

### Hill Climbing Limitations
- **Local Optima**: May get stuck in suboptimal solutions
- **Random Restarts**: Helps escape local optima
- **Non-optimal**: No guarantee of finding best path
- **Use Case**: Quick approximate solutions only

## 📈 Key Insights

### When Each Algorithm Performs Better:

**BFS** excels when:
- All movements have equal cost
- Memory is not a constraint
- Simplicity is preferred

**UCS** is best when:
- Terrain costs vary significantly
- Optimal path cost is critical
- Time is not a major constraint

**A*** dominates when:
- Balance of optimality and speed needed
- Heuristic information available
- Most practical applications

**Hill Climbing** works when:
- Quick approximate solution acceptable
- Memory is extremely limited
- Perfect solution not required

**Dynamic A*** necessary when:
- Environment changes during execution
- Moving obstacles present
- Real-time adaptation required

### Scalability Findings:
- A* scales best to large environments
- BFS memory usage grows exponentially
- Hill Climbing performance is unpredictable
- Dynamic replanning overhead is manageable

## 🔧 Technical Implementation

### Core Components

**Environment Class**:
- Grid representation with cell types and costs
- Moving obstacle simulation
- Validity checking and neighbor generation

**PathfindingAlgorithms Class**:
- All search algorithms in unified interface
- Performance metric tracking
- Heuristic functions (Manhattan, Euclidean)

**DeliveryAgent Class**:
- High-level agent behavior
- Algorithm selection and execution
- Results formatting and analysis

### Key Design Decisions

1. **4-Connected Movement**: Simpler and more realistic for delivery vehicles
2. **Manhattan Distance Heuristic**: Admissible for 4-connected grids
3. **JSON Map Format**: Human-readable and easily editable
4. **Modular Design**: Easy to extend with new algorithms
5. **Comprehensive Metrics**: Detailed performance tracking

## 🎥 Demo & Visualization

The project includes visualization capabilities:
- **Grid Visualization**: Color-coded terrain and obstacles
- **Path Visualization**: Clear path marking from start to goal
- **Dynamic Replanning**: Step-by-step replanning demonstration
- **Performance Plots**: Algorithm comparison charts

Use `--visualize` flag to see the agent in action!

## 📝 Report Generation

The experimental framework automatically generates data for academic reports:
- Statistical performance tables
- Comparative analysis charts  
- Algorithm ranking by various metrics
- Recommendations based on use case

Perfect for inclusion in your 6-page project report!

## 🤝 Contributing

This is an academic project, but suggestions for improvements are welcome:
- Additional heuristic functions
- New pathfinding algorithms
- Enhanced visualization
- Performance optimizations

## 📄 License

Educational use only - CSA2001 course project.

## 🙋‍♀️ Support

For issues with:
- **Setup**: Check [requirements.md](requirements.md)
- **Usage**: Review examples in this README
- **Performance**: Run experiment_runner.py for analysis

---

**Note**: This implementation is designed for educational purposes to demonstrate AI pathfinding concepts. All algorithms are implemented from scratch without external pathfinding libraries to showcase understanding of the underlying principles.
