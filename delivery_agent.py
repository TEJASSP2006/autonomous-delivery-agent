#!/usr/bin/env python3
"""
CSA2001 - Fundamentals of AI and ML
Project 1: Autonomous Delivery Agent

This module implements an autonomous delivery agent that navigates a 2D grid city
to deliver packages using various pathfinding algorithms.

Author: TEJAS SANTOSH PAITHANKAR
REG. NO.: 24BCY10104
Institution: VIT BHOPAL UNIVERSITY
Date: September 2025
"""

import heapq
import random
import time
import json
import argparse
import copy
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class CellType(Enum):
    """Enumeration for different cell types in the grid."""
    EMPTY = 0
    OBSTACLE = 1
    START = 2
    GOAL = 3
    MOVING_OBSTACLE = 4


@dataclass
class GridCell:
    """Represents a cell in the grid environment."""
    cell_type: CellType
    movement_cost: int
    x: int
    y: int


@dataclass
class MovingObstacle:
    """Represents a moving obstacle with a predefined path."""
    obstacle_id: int
    path: List[Tuple[int, int]]
    current_step: int = 0
    
    def get_position_at_time(self, time_step: int) -> Tuple[int, int]:
        """Get obstacle position at a specific time step."""
        if not self.path:
            return (-1, -1)
        return self.path[time_step % len(self.path)]


class Environment:
    """2D Grid environment for the delivery agent."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[GridCell(CellType.EMPTY, 1, x, y) for x in range(width)] 
                    for y in range(height)]
        self.start_pos = (0, 0)
        self.goal_pos = (width-1, height-1)
        self.moving_obstacles: List[MovingObstacle] = []
        self.current_time = 0
    
    def load_from_file(self, filename: str):
        """Load environment from a grid file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.width = data['width']
        self.height = data['height']
        self.start_pos = tuple(data['start'])
        self.goal_pos = tuple(data['goal'])
        
        # Initialize grid
        self.grid = [[GridCell(CellType.EMPTY, 1, x, y) for x in range(self.width)] 
                    for y in range(self.height)]
        
        # Set terrain costs
        for y in range(self.height):
            for x in range(self.width):
                if 'terrain_costs' in data:
                    self.grid[y][x].movement_cost = data['terrain_costs'][y][x]
        
        # Set obstacles
        for obs in data.get('obstacles', []):
            x, y = obs
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y][x].cell_type = CellType.OBSTACLE
        
        # Set start and goal
        sx, sy = self.start_pos
        gx, gy = self.goal_pos
        self.grid[sy][sx].cell_type = CellType.START
        self.grid[gy][gx].cell_type = CellType.GOAL
        
        # Load moving obstacles
        self.moving_obstacles = []
        for i, moving_obs in enumerate(data.get('moving_obstacles', [])):
            obstacle = MovingObstacle(i, moving_obs['path'])
            self.moving_obstacles.append(obstacle)
    
    def is_valid_position(self, x: int, y: int, time_step: int = 0) -> bool:
        """Check if position is valid and not occupied."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        if self.grid[y][x].cell_type == CellType.OBSTACLE:
            return False
        
        # Check moving obstacles
        for obstacle in self.moving_obstacles:
            obs_pos = obstacle.get_position_at_time(time_step)
            if obs_pos == (x, y):
                return False
        
        return True
    
    def get_movement_cost(self, x: int, y: int) -> int:
        """Get movement cost for a specific cell."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return float('inf')
        return self.grid[y][x].movement_cost
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            neighbors.append((nx, ny))
        
        return neighbors


class PathfindingAlgorithms:
    """Collection of pathfinding algorithms for the delivery agent."""
    
    def __init__(self, environment: Environment):
        self.env = environment
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0
    
    def reset_metrics(self):
        """Reset algorithm performance metrics."""
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance heuristic."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def bfs(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Breadth-First Search implementation."""
        start_time = time.time()
        self.reset_metrics()
        
        queue = [(start, [start])]
        visited = set([start])
        
        while queue:
            current_pos, path = queue.pop(0)
            self.nodes_expanded += 1
            
            if current_pos == goal:
                self.execution_time = time.time() - start_time
                self.path_cost = len(path) - 1
                return path
            
            for neighbor in self.env.get_neighbors(*current_pos):
                if neighbor not in visited and self.env.is_valid_position(*neighbor):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        self.execution_time = time.time() - start_time
        return []
    
    def uniform_cost_search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Uniform Cost Search implementation."""
        start_time = time.time()
        self.reset_metrics()
        
        heap = [(0, start, [start])]
        visited = set()
        
        while heap:
            cost, current_pos, path = heapq.heappop(heap)
            
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            self.nodes_expanded += 1
            
            if current_pos == goal:
                self.execution_time = time.time() - start_time
                self.path_cost = cost
                return path
            
            for neighbor in self.env.get_neighbors(*current_pos):
                if neighbor not in visited and self.env.is_valid_position(*neighbor):
                    neighbor_cost = cost + self.env.get_movement_cost(*neighbor)
                    heapq.heappush(heap, (neighbor_cost, neighbor, path + [neighbor]))
        
        self.execution_time = time.time() - start_time
        return []
    
    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int], 
              heuristic_func=None) -> List[Tuple[int, int]]:
        """A* Search implementation with admissible heuristic."""
        start_time = time.time()
        self.reset_metrics()
        
        if heuristic_func is None:
            heuristic_func = self.manhattan_distance
        
        heap = [(heuristic_func(start, goal), 0, start, [start])]
        visited = set()
        g_costs = {start: 0}
        
        while heap:
            f_cost, g_cost, current_pos, path = heapq.heappop(heap)
            
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            self.nodes_expanded += 1
            
            if current_pos == goal:
                self.execution_time = time.time() - start_time
                self.path_cost = g_cost
                return path
            
            for neighbor in self.env.get_neighbors(*current_pos):
                if self.env.is_valid_position(*neighbor):
                    tentative_g = g_cost + self.env.get_movement_cost(*neighbor)
                    
                    if neighbor not in visited and (neighbor not in g_costs or tentative_g < g_costs[neighbor]):
                        g_costs[neighbor] = tentative_g
                        h_cost = heuristic_func(neighbor, goal)
                        f_cost = tentative_g + h_cost
                        heapq.heappush(heap, (f_cost, tentative_g, neighbor, path + [neighbor]))
        
        self.execution_time = time.time() - start_time
        return []
    
    def hill_climbing_with_restarts(self, start: Tuple[int, int], goal: Tuple[int, int], 
                                   max_restarts: int = 5) -> List[Tuple[int, int]]:
        """Hill Climbing with random restarts for local search."""
        start_time = time.time()
        self.reset_metrics()
        
        best_path = []
        best_cost = float('inf')
        
        for restart in range(max_restarts):
            current_pos = start
            path = [start]
            
            while current_pos != goal:
                neighbors = self.env.get_neighbors(*current_pos)
                valid_neighbors = [(n, self.manhattan_distance(n, goal)) 
                                 for n in neighbors if self.env.is_valid_position(*n)]
                
                if not valid_neighbors:
                    break
                
                # Sort by heuristic distance to goal
                valid_neighbors.sort(key=lambda x: x[1])
                current_distance = self.manhattan_distance(current_pos, goal)
                
                # Find improving neighbors
                improving_neighbors = [n for n, dist in valid_neighbors if dist < current_distance]
                
                if improving_neighbors:
                    current_pos = improving_neighbors[0]
                    path.append(current_pos)
                else:
                    # Random restart
                    valid_positions = []
                    for y in range(self.env.height):
                        for x in range(self.env.width):
                            if self.env.is_valid_position(x, y):
                                valid_positions.append((x, y))
                    if valid_positions:
                        current_pos = random.choice(valid_positions)
                        path = [current_pos]
                
                self.nodes_expanded += 1
                
                if len(path) > self.env.width * self.env.height * 2:  # Prevent infinite loops
                    break
            
            if current_pos == goal:
                path_cost = sum(self.env.get_movement_cost(*pos) for pos in path[1:])
                if path_cost < best_cost:
                    best_cost = path_cost
                    best_path = path
        
        self.execution_time = time.time() - start_time
        self.path_cost = best_cost if best_path else 0
        return best_path
    
    def dynamic_replan_a_star(self, start: Tuple[int, int], goal: Tuple[int, int], 
                             initial_time: int = 0) -> Tuple[List[Tuple[int, int]], List[str]]:
        """A* with dynamic replanning when obstacles are detected."""
        start_time = time.time()
        self.reset_metrics()
        
        log = []
        full_path = []
        current_pos = start
        current_time = initial_time
        
        while current_pos != goal:
            # Plan path from current position
            partial_path = self.a_star_with_time(current_pos, goal, current_time)
            
            if not partial_path:
                log.append(f"Time {current_time}: No path found from {current_pos}")
                break
            
            log.append(f"Time {current_time}: Planned path from {current_pos}: {partial_path[:5]}...")
            
            # Execute path until obstacle or goal
            for i, next_pos in enumerate(partial_path[1:], 1):
                next_time = current_time + i
                
                # Check if path is still valid
                if not self.env.is_valid_position(*next_pos, next_time):
                    log.append(f"Time {next_time}: Obstacle detected at {next_pos}, replanning...")
                    current_time = next_time - 1
                    break
                
                full_path.append(next_pos)
                current_pos = next_pos
                current_time = next_time
                
                if current_pos == goal:
                    break
            else:
                # Path completed successfully
                break
        
        self.execution_time = time.time() - start_time
        self.path_cost = len(full_path)
        return full_path, log
    
    def a_star_with_time(self, start: Tuple[int, int], goal: Tuple[int, int], 
                        start_time: int) -> List[Tuple[int, int]]:
        """A* implementation considering time for moving obstacles."""
        heap = [(self.manhattan_distance(start, goal), 0, start, [start], start_time)]
        visited = set()
        
        while heap:
            f_cost, g_cost, current_pos, path, current_time = heapq.heappop(heap)
            
            state = (current_pos, current_time)
            if state in visited:
                continue
            
            visited.add(state)
            
            if current_pos == goal:
                return path
            
            for neighbor in self.env.get_neighbors(*current_pos):
                next_time = current_time + 1
                if self.env.is_valid_position(*neighbor, next_time):
                    tentative_g = g_cost + self.env.get_movement_cost(*neighbor)
                    h_cost = self.manhattan_distance(neighbor, goal)
                    f_cost = tentative_g + h_cost
                    
                    next_state = (neighbor, next_time)
                    if next_state not in visited:
                        heapq.heappush(heap, (f_cost, tentative_g, neighbor, 
                                           path + [neighbor], next_time))
        
        return []


class DeliveryAgent:
    """Main delivery agent class."""
    
    def __init__(self, environment: Environment):
        self.env = environment
        self.pathfinder = PathfindingAlgorithms(environment)
        self.current_pos = environment.start_pos
        self.fuel = 1000  # Maximum fuel
        self.packages_delivered = 0
    
    def solve_with_algorithm(self, algorithm: str) -> Dict:
        """Solve the delivery problem using specified algorithm."""
        start_pos = self.env.start_pos
        goal_pos = self.env.goal_pos
        
        if algorithm == "bfs":
            path = self.pathfinder.bfs(start_pos, goal_pos)
        elif algorithm == "ucs":
            path = self.pathfinder.uniform_cost_search(start_pos, goal_pos)
        elif algorithm == "astar":
            path = self.pathfinder.a_star(start_pos, goal_pos)
        elif algorithm == "hill_climbing":
            path = self.pathfinder.hill_climbing_with_restarts(start_pos, goal_pos)
        elif algorithm == "dynamic":
            path, log = self.pathfinder.dynamic_replan_a_star(start_pos, goal_pos)
            return {
                'algorithm': algorithm,
                'path': path,
                'path_cost': self.pathfinder.path_cost,
                'nodes_expanded': self.pathfinder.nodes_expanded,
                'execution_time': self.pathfinder.execution_time,
                'success': len(path) > 0 and path[-1] == goal_pos,
                'replan_log': log
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return {
            'algorithm': algorithm,
            'path': path,
            'path_cost': self.pathfinder.path_cost,
            'nodes_expanded': self.pathfinder.nodes_expanded,
            'execution_time': self.pathfinder.execution_time,
            'success': len(path) > 0 and path[-1] == goal_pos
        }


def create_test_maps():
    """Create test maps for evaluation."""
    maps = {
        'small': {
            'width': 8,
            'height': 8,
            'start': [0, 0],
            'goal': [7, 7],
            'obstacles': [[2, 2], [3, 2], [4, 2], [2, 3], [2, 4]],
            'terrain_costs': [[1 for _ in range(8)] for _ in range(8)],
            'moving_obstacles': []
        },
        'medium': {
            'width': 15,
            'height': 15,
            'start': [0, 0],
            'goal': [14, 14],
            'obstacles': [[5, i] for i in range(5, 10)] + [[i, 7] for i in range(8, 13)],
            'terrain_costs': [[random.randint(1, 3) for _ in range(15)] for _ in range(15)],
            'moving_obstacles': []
        },
        'large': {
            'width': 25,
            'height': 25,
            'start': [0, 0],
            'goal': [24, 24],
            'obstacles': [[i, 12] for i in range(8, 18)] + [[12, i] for i in range(5, 20)],
            'terrain_costs': [[random.randint(1, 5) for _ in range(25)] for _ in range(25)],
            'moving_obstacles': []
        },
        'dynamic': {
            'width': 12,
            'height': 12,
            'start': [0, 0],
            'goal': [11, 11],
            'obstacles': [[4, 4], [4, 5], [5, 4]],
            'terrain_costs': [[1 for _ in range(12)] for _ in range(12)],
            'moving_obstacles': [
                {
                    'path': [[6, 2], [6, 3], [6, 4], [6, 5], [6, 4], [6, 3]]
                },
                {
                    'path': [[8, 8], [9, 8], [10, 8], [9, 8]]
                }
            ]
        }
    }
    
    # Save maps to files
    for name, map_data in maps.items():
        with open(f'maps/{name}_map.json', 'w') as f:
            json.dump(map_data, f, indent=2)
    
    return maps


def main():
    """Main function for CLI interface."""
    parser = argparse.ArgumentParser(description='Autonomous Delivery Agent')
    parser.add_argument('map_file', help='Path to map file')
    parser.add_argument('algorithm', choices=['bfs', 'ucs', 'astar', 'hill_climbing', 'dynamic'],
                       help='Pathfinding algorithm to use')
    parser.add_argument('--visualize', action='store_true', help='Visualize the solution')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load environment
    env = Environment(10, 10)  # Default size
    try:
        env.load_from_file(args.map_file)
    except FileNotFoundError:
        print(f"Map file {args.map_file} not found!")
        return
    
    # Create agent and solve
    agent = DeliveryAgent(env)
    result = agent.solve_with_algorithm(args.algorithm)
    
    # Print results
    print(f"\nAlgorithm: {result['algorithm']}")
    print(f"Success: {result['success']}")
    print(f"Path Cost: {result['path_cost']}")
    print(f"Nodes Expanded: {result['nodes_expanded']}")
    print(f"Execution Time: {result['execution_time']:.4f} seconds")
    
    if args.verbose and result['path']:
        print(f"Path: {result['path']}")
    
    if 'replan_log' in result and args.verbose:
        print("\nReplanning Log:")
        for log_entry in result['replan_log']:
            print(log_entry)
    
    # Visualization
    if args.visualize and result['path']:
        visualize_solution(env, result['path'], result['algorithm'])


def visualize_solution(env: Environment, path: List[Tuple[int, int]], algorithm: str):
    """Visualize the solution path."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid visualization
    grid_viz = np.zeros((env.height, env.width))
    
    # Set terrain costs as background
    for y in range(env.height):
        for x in range(env.width):
            grid_viz[y, x] = env.grid[y][x].movement_cost
    
    # Mark obstacles
    for y in range(env.height):
        for x in range(env.width):
            if env.grid[y][x].cell_type == CellType.OBSTACLE:
                grid_viz[y, x] = -1
    
    # Create colormap
    im = ax.imshow(grid_viz, cmap='RdYlGn_r', alpha=0.7)
    
    # Plot path
    if path:
        path_x = [pos[0] for pos in path]
        path_y = [pos[1] for pos in path]
        ax.plot(path_x, path_y, 'b-', linewidth=3, label='Path')
        ax.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='Goal')
    
    # Mark moving obstacles at time 0
    for obs in env.moving_obstacles:
        pos = obs.get_position_at_time(0)
        if pos != (-1, -1):
            ax.plot(pos[0], pos[1], 'mo', markersize=8, label='Moving Obstacle')
    
    ax.set_title(f'Delivery Agent Solution - {algorithm.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, label='Movement Cost')
    plt.show()


if __name__ == "__main__":
    # Create test maps directory
    import os
    os.makedirs('maps', exist_ok=True)
    
    # Create test maps
    create_test_maps()
    print("Test maps created in 'maps' directory")
    
    # Run main if arguments provided
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        print("Usage: python delivery_agent.py <map_file> <algorithm> [--visualize] [--verbose]")
        print("Available algorithms: bfs, ucs, astar, hill_climbing, dynamic")

