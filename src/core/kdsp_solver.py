#%%
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import heapq
from numba import jit, types
from numba.typed import Dict as NumbaDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathResult:
    """路径结果类"""
    paths: List[List[int]]
    costs: List[float]
    total_cost: float
    computation_time: float
    success: bool


class KDSPSolver:

    def __init__(self, graph: nx.DiGraph, k: int = 2):
        """
        初始化k-dSPP求解器

        Args:
            graph: 有向图 (NetworkX DiGraph)
            k: 不相交路径数量
        """
        self.graph = graph
        self.k = k
        self.n_nodes = len(graph.nodes())
        self.n_edges = len(graph.edges())

        # 预处理图数据结构以提高性能
        self.adj_matrix, self.node_mapping = self._preprocess_graph()

        logger.info(f"初始化k-dSPP求解器: {self.n_nodes}节点, {self.n_edges}边, k={k}")

    def _preprocess_graph(self) -> Tuple[np.ndarray, Dict]:
        """预处理图为邻接矩阵格式以提高计算效率"""
        nodes = list(self.graph.nodes())
        node_mapping = {node: i for i, node in enumerate(nodes)}
        reverse_mapping = {i: node for node, i in node_mapping.items()}

        # 创建邻接矩阵
        adj_matrix = np.full((self.n_nodes, self.n_nodes), np.inf)

        for u, v, data in self.graph.edges(data=True):
            u_idx = node_mapping[u]
            v_idx = node_mapping[v]
            weight = data.get('weight', 1.0)
            adj_matrix[u_idx, v_idx] = weight

        # 对角线设为0
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix, {'forward': node_mapping, 'reverse': reverse_mapping}

    def solve(self, source: int, target: int,
              algorithm: str = 'dinic') -> PathResult:
        """
        求解k-不相交最短路径问题

        Args:
            source: 源节点
            target: 目标节点
            algorithm: 算法类型 ('dinic', 'ford_fulkerson', 'edmonds_karp')

        Returns:
            PathResult: 包含k条不相交路径的结果
        """
        import time
        start_time = time.time()

        try:
            # 转换节点索引
            source_idx = self.node_mapping['forward'].get(source)
            target_idx = self.node_mapping['forward'].get(target)

            if source_idx is None or target_idx is None:
                raise ValueError(f"源节点{source}或目标节点{target}不在图中")

            if algorithm == 'dinic':
                paths, costs = self._solve_dinic(source_idx, target_idx)
            elif algorithm == 'ford_fulkerson':
                paths, costs = self._solve_ford_fulkerson(source_idx, target_idx)
            elif algorithm == 'edmonds_karp':
                paths, costs = self._solve_edmonds_karp(source_idx, target_idx)
            else:
                raise ValueError(f"未知算法: {algorithm}")

            # 转换回原始节点标识
            original_paths = []
            for path in paths:
                original_path = [self.node_mapping['reverse'][node_idx]
                                 for node_idx in path]
                original_paths.append(original_path)

            total_cost = sum(costs) if costs else float('inf')
            computation_time = time.time() - start_time

            result = PathResult(
                paths=original_paths,
                costs=costs,
                total_cost=total_cost,
                computation_time=computation_time,
                success=len(paths) >= self.k
            )

            logger.info(f"k-dSPP求解完成: {len(paths)}条路径, "
                        f"总成本={total_cost:.4f}, 耗时={computation_time:.4f}s")

            return result

        except Exception as e:
            logger.error(f"k-dSPP求解失败: {str(e)}")
            return PathResult([], [], float('inf'), time.time() - start_time, False)

    def _solve_dinic(self, source: int, target: int) -> Tuple[List[List[int]], List[float]]:
        """
        使用Dinic算法求解k-不相交最短路径

        这是最高效的算法实现，时间复杂度O(V²E)
        """
        paths = []
        costs = []

        # 创建残余图
        residual_graph = self.adj_matrix.copy()

        for _ in range(self.k):
            # 使用Dinic算法找到最短路径
            path, cost = self._dinic_shortest_path(residual_graph, source, target)

            if not path:
                break

            paths.append(path)
            costs.append(cost)

            # 更新残余图：移除使用的边
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual_graph[u, v] = np.inf  # 移除边

        return paths, costs

    @staticmethod
    @jit(nopython=True)
    def _dinic_shortest_path(adj_matrix: np.ndarray, source: int,
                             target: int) -> Tuple[List[int], float]:
        """
        Dinic算法的核心最短路径计算 (使用Numba加速)
        """
        n = adj_matrix.shape[0]

        # BFS构建层次图
        level = np.full(n, -1, dtype=types.int32)
        level[source] = 0

        queue = [source]
        head = 0

        while head < len(queue) and level[target] == -1:
            u = queue[head]
            head += 1

            for v in range(n):
                if level[v] == -1 and adj_matrix[u, v] < np.inf:
                    level[v] = level[u] + 1
                    queue.append(v)

        if level[target] == -1:
            return [], np.inf

        # DFS找到最短路径
        path = []
        if KDSPSolver._dfs_path(adj_matrix, level, source, target, path):
            cost = sum(adj_matrix[path[i], path[i + 1]]
                       for i in range(len(path) - 1))
            return path, cost

        return [], np.inf

    @staticmethod
    @jit(nopython=True)
    def _dfs_path(adj_matrix: np.ndarray, level: np.ndarray,
                  current: int, target: int, path: List[int]) -> bool:
        """DFS寻找路径 (使用Numba加速)"""
        path.append(current)

        if current == target:
            return True

        for next_node in range(adj_matrix.shape[0]):
            if (level[next_node] == level[current] + 1 and
                    adj_matrix[current, next_node] < np.inf):
                if KDSPSolver._dfs_path(adj_matrix, level, next_node, target, path):
                    return True

        path.pop()
        return False

    def _solve_ford_fulkerson(self, source: int, target: int) -> Tuple[List[List[int]], List[float]]:
        """Ford-Fulkerson算法实现"""
        paths = []
        costs = []

        # 创建残余图
        residual_graph = self.adj_matrix.copy()

        for _ in range(self.k):
            # 使用DFS找到增广路径
            path, cost = self._ford_fulkerson_path(residual_graph, source, target)

            if not path:
                break

            paths.append(path)
            costs.append(cost)

            # 更新残余图
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual_graph[u, v] = np.inf

        return paths, costs

    def _ford_fulkerson_path(self, residual_graph: np.ndarray,
                             source: int, target: int) -> Tuple[List[int], float]:
        """Ford-Fulkerson算法的路径查找"""
        visited = np.zeros(self.n_nodes, dtype=bool)
        path = []

        if self._dfs_ford_fulkerson(residual_graph, source, target, visited, path):
            cost = sum(residual_graph[path[i], path[i + 1]]
                       for i in range(len(path) - 1))
            return path, cost

        return [], np.inf

    def _dfs_ford_fulkerson(self, residual_graph: np.ndarray, current: int,
                            target: int, visited: np.ndarray,
                            path: List[int]) -> bool:
        """Ford-Fulkerson的DFS实现"""
        visited[current] = True
        path.append(current)

        if current == target:
            return True

        for next_node in range(self.n_nodes):
            if (not visited[next_node] and
                    residual_graph[current, next_node] < np.inf):
                if self._dfs_ford_fulkerson(residual_graph, next_node, target,
                                            visited, path):
                    return True

        path.pop()
        return False

    def _solve_edmonds_karp(self, source: int, target: int) -> Tuple[List[List[int]], List[float]]:
        """Edmonds-Karp算法实现 (BFS版本的Ford-Fulkerson)"""
        paths = []
        costs = []

        # 创建残余图
        residual_graph = self.adj_matrix.copy()

        for _ in range(self.k):
            # 使用BFS找到最短增广路径
            path, cost = self._edmonds_karp_path(residual_graph, source, target)

            if not path:
                break

            paths.append(path)
            costs.append(cost)

            # 更新残余图
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                residual_graph[u, v] = np.inf

        return paths, costs

    def _edmonds_karp_path(self, residual_graph: np.ndarray,
                           source: int, target: int) -> Tuple[List[int], float]:
        """Edmonds-Karp算法的BFS路径查找"""
        from collections import deque

        parent = np.full(self.n_nodes, -1, dtype=int)
        visited = np.zeros(self.n_nodes, dtype=bool)
        queue = deque([source])
        visited[source] = True

        # BFS寻找路径
        while queue:
            u = queue.popleft()

            for v in range(self.n_nodes):
                if not visited[v] and residual_graph[u, v] < np.inf:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u

                    if v == target:
                        # 重构路径
                        path = []
                        current = target
                        while current != -1:
                            path.append(current)
                            current = parent[current]
                        path.reverse()

                        # 计算成本
                        cost = sum(residual_graph[path[i], path[i + 1]]
                                   for i in range(len(path) - 1))

                        return path, cost

        return [], np.inf

    def solve_batch(self, requests: List[Tuple[int, int]],
                    algorithm: str = 'dinic') -> List[PathResult]:
        """
        批量求解多个k-dSPP问题

        Args:
            requests: 源-目标节点对列表
            algorithm: 使用的算法

        Returns:
            每个请求的PathResult列表
        """
        results = []

        for source, target in requests:
            result = self.solve(source, target, algorithm)
            results.append(result)

        logger.info(f"批量求解完成: {len(requests)}个请求")
        return results

    def get_statistics(self) -> Dict:
        """获取求解器统计信息"""
        return {
            'n_nodes': self.n_nodes,
            'n_edges': self.n_edges,
            'k': self.k,
            'graph_density': self.n_edges / (self.n_nodes * (self.n_nodes - 1)),
            'memory_usage_mb': self.adj_matrix.nbytes / (1024 * 1024)
        }

    def visualize_solution(self, result: PathResult, save_path: Optional[str] = None):
        """可视化k-dSPP解决方案"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            if not result.success or not result.paths:
                logger.warning("没有有效路径可以可视化")
                return

            # 创建图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # 绘制图的基础结构
            pos = nx.spring_layout(self.graph, seed=42)
            nx.draw_networkx_nodes(self.graph, pos, ax=ax,
                                   node_color='lightblue',
                                   node_size=300, alpha=0.7)
            nx.draw_networkx_labels(self.graph, pos, ax=ax)

            # 绘制所有边（灰色）
            nx.draw_networkx_edges(self.graph, pos, ax=ax,
                                   edge_color='gray', alpha=0.3)

            # 为每条路径使用不同颜色
            colors = ['red', 'blue', 'green', 'orange', 'purple']

            for i, path in enumerate(result.paths[:self.k]):
                color = colors[i % len(colors)]

                # 绘制路径边
                path_edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
                nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges,
                                       ax=ax, edge_color=color, width=3,
                                       label=f'路径 {i + 1} (成本: {result.costs[i]:.2f})')

            ax.set_title(f'k-dSPP解决方案 (k={self.k})\n'
                         f'总成本: {result.total_cost:.2f}, '
                         f'计算时间: {result.computation_time:.4f}s')
            ax.legend()
            ax.axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"可视化结果已保存到: {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化")
        except Exception as e:
            logger.error(f"可视化过程中出错: {str(e)}")


class KDSPSolverFactory:
    """k-dSPP求解器工厂类"""

    @staticmethod
    def create_from_adjacency_matrix(adj_matrix: np.ndarray, k: int = 2) -> KDSPSolver:
        """从邻接矩阵创建求解器"""
        graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        return KDSPSolver(graph, k)

    @staticmethod
    def create_from_edge_list(edges: List[Tuple], k: int = 2) -> KDSPSolver:
        """从边列表创建求解器"""
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(edges)
        return KDSPSolver(graph, k)

    @staticmethod
    def create_from_file(file_path: str, k: int = 2) -> KDSPSolver:
        """从文件创建求解器"""
        import pandas as pd
        from pathlib import Path

        file_path = Path(file_path)

        if file_path.suffix == '.csv':
            # CSV格式: source, target, weight
            df = pd.read_csv(file_path)
            edges = [(row['source'], row['target'], row['weight'])
                     for _, row in df.iterrows()]
        elif file_path.suffix == '.txt':
            # 文本格式: 每行 source target weight
            edges = []
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        source, target, weight = parts[0], parts[1], float(parts[2])
                        edges.append((source, target, weight))
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        return KDSPSolverFactory.create_from_edge_list(edges, k)


# 性能优化的辅助函数
@jit(nopython=True)
def compute_shortest_distances(adj_matrix: np.ndarray) -> np.ndarray:
    """使用Floyd-Warshall算法计算所有节点对之间的最短距离"""
    n = adj_matrix.shape[0]
    dist = adj_matrix.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    return dist


@jit(nopython=True)
def check_graph_connectivity(adj_matrix: np.ndarray) -> bool:
    """检查图的连通性"""
    n = adj_matrix.shape[0]
    dist = compute_shortest_distances(adj_matrix)

    # 检查是否所有节点对都可达
    for i in range(n):
        for j in range(n):
            if i != j and dist[i, j] >= np.inf:
                return False

    return True


def benchmark_algorithms(graph: nx.DiGraph, source: int, target: int,
                         k: int = 2, iterations: int = 10) -> Dict:
    """对不同算法进行基准测试"""
    import time

    solver = KDSPSolver(graph, k)
    algorithms = ['dinic', 'ford_fulkerson', 'edmonds_karp']
    results = {}

    for algorithm in algorithms:
        times = []
        costs = []
        successes = 0

        for _ in range(iterations):
            start_time = time.time()
            result = solver.solve(source, target, algorithm)
            end_time = time.time()

            times.append(end_time - start_time)
            costs.append(result.total_cost)
            if result.success:
                successes += 1

        results[algorithm] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_cost': np.mean(costs),
            'success_rate': successes / iterations
        }

    return results