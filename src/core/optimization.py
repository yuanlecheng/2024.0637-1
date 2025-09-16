#%%
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
from numba import jit, types
from numba.typed import Dict as NumbaDict, List as NumbaList
import heapq
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    success: bool
    total_cost: float
    computation_time: float
    solution_details: Dict[str, Any]
    objective_value: float
    num_served: int
    service_rate: float
    decision_variables: Optional[List[float]] = None
    dual_variables: Optional[List[float]] = None
    status: str = "unknown"
    gap: float = 0.0


@dataclass
class Request:
    """出行请求数据结构"""
    id: int
    origin: int
    destination: int
    request_time: float
    pickup_deadline: float
    delivery_deadline: float
    priority: float = 1.0
    passenger_count: int = 1


@dataclass
class Vehicle:
    """车辆数据结构"""
    id: int
    initial_position: int
    initial_time: float
    capacity: int = 4
    speed: float = 1.0  # 速度系数
    operational_cost: float = 1.0


class AbstractOptimizer(ABC):
    """抽象优化器基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def solve(self, *args, **kwargs) -> OptimizationResult:
        """抽象求解方法"""
        pass


class FullInformationSolver(AbstractOptimizer):
    """
    完全信息问题求解器

    求解离线AMoD问题，已知所有请求信息
    使用改进的匈牙利算法和最小费用流算法
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.time_limit = config.get('optimization.time_limit_seconds', 3600)
        self.gap_tolerance = config.get('optimization.gap_tolerance', 0.001)
        self.solver_type = config.get('optimization.solver_type', 'min_cost_flow')

    def solve(self, demands: List[Tuple[int, int, float]],
              vehicles: List[Tuple[int, float]],
              graph: nx.DiGraph) -> OptimizationResult:
        """
        求解完全信息问题

        Args:
            demands: 需求列表 [(origin, destination, time), ...]
            vehicles: 车辆列表 [(position, time), ...]
            graph: 路网图

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()

        try:
            self.logger.info(f"开始求解完全信息问题: {len(demands)}个需求, {len(vehicles)}个车辆")

            # 构建请求和车辆对象
            requests = self._build_requests(demands)
            fleet = self._build_vehicles(vehicles)

            # 预计算距离矩阵
            distance_matrix = self._precompute_distances(graph)

            # 根据求解器类型选择算法
            if self.solver_type == 'min_cost_flow':
                result = self._solve_min_cost_flow(requests, fleet, distance_matrix, graph)
            elif self.solver_type == 'hungarian':
                result = self._solve_hungarian(requests, fleet, distance_matrix)
            elif self.solver_type == 'greedy':
                result = self._solve_greedy(requests, fleet, distance_matrix)
            else:
                raise ValueError(f"未知的求解器类型: {self.solver_type}")

            computation_time = time.time() - start_time
            result.computation_time = computation_time

            self.logger.info(f"完全信息问题求解完成: 成本={result.total_cost:.2f}, "
                             f"服务率={result.service_rate:.2%}, 耗时={computation_time:.4f}s")

            return result

        except Exception as e:
            self.logger.error(f"完全信息问题求解失败: {str(e)}")
            return OptimizationResult(
                success=False,
                total_cost=float('inf'),
                computation_time=time.time() - start_time,
                solution_details={'error': str(e)},
                objective_value=float('inf'),
                num_served=0,
                service_rate=0.0,
                status="error"
            )

    def _build_requests(self, demands: List[Tuple[int, int, float]]) -> List[Request]:
        """构建请求对象"""
        requests = []
        for i, (origin, dest, req_time) in enumerate(demands):
            # 设置截止时间（简化版本）
            pickup_deadline = req_time + self.config.get('request.max_pickup_delay', 10.0)
            delivery_deadline = pickup_deadline + self.config.get('request.max_travel_time', 30.0)

            request = Request(
                id=i,
                origin=origin,
                destination=dest,
                request_time=req_time,
                pickup_deadline=pickup_deadline,
                delivery_deadline=delivery_deadline,
                priority=1.0
            )
            requests.append(request)

        return requests

    def _build_vehicles(self, vehicles: List[Tuple[int, float]]) -> List[Vehicle]:
        """构建车辆对象"""
        fleet = []
        for i, (position, initial_time) in enumerate(vehicles):
            vehicle = Vehicle(
                id=i,
                initial_position=position,
                initial_time=initial_time,
                capacity=self.config.get('vehicle.capacity', 4)
            )
            fleet.append(vehicle)

        return fleet

    def _precompute_distances(self, graph: nx.DiGraph) -> np.ndarray:
        """预计算所有节点对之间的最短距离"""
        try:
            # 使用Floyd-Warshall算法计算所有对最短路径
            nodes = list(graph.nodes())
            n = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}

            # 初始化距离矩阵
            dist_matrix = np.full((n, n), np.inf, dtype=np.float64)
            np.fill_diagonal(dist_matrix, 0)

            # 设置直接连接的边
            for u, v, data in graph.edges(data=True):
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                weight = data.get('weight', 1.0)
                dist_matrix[u_idx, v_idx] = weight

            # Floyd-Warshall算法
            dist_matrix = self._floyd_warshall_numba(dist_matrix)

            return dist_matrix, node_to_idx

        except Exception as e:
            self.logger.error(f"距离矩阵计算失败: {str(e)}")
            # 返回默认距离矩阵
            n = len(graph.nodes())
            return np.ones((n, n), dtype=np.float64), {}

    @staticmethod
    @jit(nopython=True)
    def _floyd_warshall_numba(dist_matrix: np.ndarray) -> np.ndarray:
        """使用Numba加速的Floyd-Warshall算法"""
        n = dist_matrix.shape[0]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i, k] + dist_matrix[k, j] < dist_matrix[i, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

        return dist_matrix

    def _solve_min_cost_flow(self, requests: List[Request], vehicles: List[Vehicle],
                             distance_info: Tuple[np.ndarray, Dict], graph: nx.DiGraph) -> OptimizationResult:
        """使用最小费用流算法求解"""
        try:
            distance_matrix, node_to_idx = distance_info

            # 构建二部图：车辆-请求匹配
            flow_graph = nx.DiGraph()

            # 添加源点和汇点
            source = 'source'
            sink = 'sink'
            flow_graph.add_node(source)
            flow_graph.add_node(sink)

            # 添加车辆节点
            for vehicle in vehicles:
                vehicle_node = f'v_{vehicle.id}'
                flow_graph.add_node(vehicle_node)
                # 源点到车辆的边（容量1，费用0）
                flow_graph.add_edge(source, vehicle_node, capacity=1, weight=0)

            # 添加请求节点
            for request in requests:
                request_node = f'r_{request.id}'
                flow_graph.add_node(request_node)
                # 请求到汇点的边（容量1，费用0）
                flow_graph.add_edge(request_node, sink, capacity=1, weight=0)

            # 添加车辆-请求匹配边
            for vehicle in vehicles:
                for request in requests:
                    # 计算服务成本
                    cost = self._calculate_service_cost(vehicle, request, distance_matrix, node_to_idx)

                    if cost < np.inf:  # 只添加可行的匹配
                        vehicle_node = f'v_{vehicle.id}'
                        request_node = f'r_{request.id}'
                        flow_graph.add_edge(vehicle_node, request_node, capacity=1, weight=cost)

            # 求解最小费用最大流
            flow_cost, flow_dict = nx.network_simplex(flow_graph)

            # 解析解决方案
            assignments = {}
            total_cost = 0
            num_served = 0

            for vehicle in vehicles:
                vehicle_node = f'v_{vehicle.id}'
                if vehicle_node in flow_dict:
                    for request_node, flow in flow_dict[vehicle_node].items():
                        if flow > 0 and request_node.startswith('r_'):
                            request_id = int(request_node.split('_')[1])
                            assignments[vehicle.id] = request_id

                            # 计算实际成本
                            request = requests[request_id]
                            cost = self._calculate_service_cost(vehicle, request, distance_matrix, node_to_idx)
                            total_cost += cost
                            num_served += 1

            service_rate = num_served / len(requests) if requests else 0

            return OptimizationResult(
                success=True,
                total_cost=total_cost,
                computation_time=0,  # 将在外层设置
                solution_details={'assignments': assignments, 'method': 'min_cost_flow'},
                objective_value=total_cost,
                num_served=num_served,
                service_rate=service_rate,
                status="optimal",
                gap=0.0
            )

        except Exception as e:
            self.logger.error(f"最小费用流求解失败: {str(e)}")
            return OptimizationResult(
                success=False,
                total_cost=float('inf'),
                computation_time=0,
                solution_details={'error': str(e)},
                objective_value=float('inf'),
                num_served=0,
                service_rate=0.0,
                status="error"
            )

    def _solve_hungarian(self, requests: List[Request], vehicles: List[Vehicle],
                         distance_info: Tuple[np.ndarray, Dict]) -> OptimizationResult:
        """使用匈牙利算法求解"""
        try:
            from scipy.optimize import linear_sum_assignment

            distance_matrix, node_to_idx = distance_info

            # 构建成本矩阵
            n_vehicles = len(vehicles)
            n_requests = len(requests)

            # 创建方形成本矩阵（添加虚拟节点）
            size = max(n_vehicles, n_requests)
            cost_matrix = np.full((size, size), np.inf, dtype=np.float64)

            # 填充实际成本
            for i, vehicle in enumerate(vehicles):
                for j, request in enumerate(requests):
                    cost = self._calculate_service_cost(vehicle, request, distance_matrix, node_to_idx)
                    cost_matrix[i, j] = cost if cost < np.inf else 1e6  # 替换inf以避免数值问题

            # 对于多余的行/列，使用0成本
            for i in range(n_vehicles, size):
                for j in range(size):
                    cost_matrix[i, j] = 0
            for i in range(size):
                for j in range(n_requests, size):
                    cost_matrix[i, j] = 0

            # 求解匈牙利问题
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            # 解析结果
            assignments = {}
            total_cost = 0
            num_served = 0

            for i, j in zip(row_indices, col_indices):
                if i < n_vehicles and j < n_requests and cost_matrix[i, j] < 1e6:
                    assignments[vehicles[i].id] = requests[j].id
                    total_cost += cost_matrix[i, j]
                    num_served += 1

            service_rate = num_served / len(requests) if requests else 0

            return OptimizationResult(
                success=True,
                total_cost=total_cost,
                computation_time=0,
                solution_details={'assignments': assignments, 'method': 'hungarian'},
                objective_value=total_cost,
                num_served=num_served,
                service_rate=service_rate,
                status="optimal"
            )

        except ImportError:
            self.logger.error("scipy未安装，无法使用匈牙利算法")
            return self._solve_greedy(requests, vehicles, distance_info)
        except Exception as e:
            self.logger.error(f"匈牙利算法求解失败: {str(e)}")
            return self._solve_greedy(requests, vehicles, distance_info)

    def _solve_greedy(self, requests: List[Request], vehicles: List[Vehicle],
                      distance_info: Tuple[np.ndarray, Dict]) -> OptimizationResult:
        """使用贪婪算法求解"""
        try:
            distance_matrix, node_to_idx = distance_info

            assignments = {}
            total_cost = 0
            available_vehicles = set(v.id for v in vehicles)
            unserved_requests = set(r.id for r in requests)

            # 计算所有可能的车辆-请求配对成本
            pairings = []
            for vehicle in vehicles:
                for request in requests:
                    cost = self._calculate_service_cost(vehicle, request, distance_matrix, node_to_idx)
                    if cost < np.inf:
                        pairings.append((cost, vehicle.id, request.id))

            # 按成本排序
            pairings.sort(key=lambda x: x[0])

            # 贪婪分配
            for cost, vehicle_id, request_id in pairings:
                if vehicle_id in available_vehicles and request_id in unserved_requests:
                    assignments[vehicle_id] = request_id
                    total_cost += cost
                    available_vehicles.remove(vehicle_id)
                    unserved_requests.remove(request_id)

            num_served = len(assignments)
            service_rate = num_served / len(requests) if requests else 0

            return OptimizationResult(
                success=True,
                total_cost=total_cost,
                computation_time=0,
                solution_details={'assignments': assignments, 'method': 'greedy'},
                objective_value=total_cost,
                num_served=num_served,
                service_rate=service_rate,
                status="feasible"
            )

        except Exception as e:
            self.logger.error(f"贪婪算法求解失败: {str(e)}")
            return OptimizationResult(
                success=False,
                total_cost=float('inf'),
                computation_time=0,
                solution_details={'error': str(e)},
                objective_value=float('inf'),
                num_served=0,
                service_rate=0.0,
                status="error"
            )

    def _calculate_service_cost(self, vehicle: Vehicle, request: Request,
                                distance_matrix: np.ndarray, node_to_idx: Dict) -> float:
        """计算车辆服务请求的成本"""
        try:
            # 获取节点索引
            vehicle_pos_idx = node_to_idx.get(vehicle.initial_position)
            origin_idx = node_to_idx.get(request.origin)
            dest_idx = node_to_idx.get(request.destination)

            if any(idx is None for idx in [vehicle_pos_idx, origin_idx, dest_idx]):
                return np.inf

            # 计算距离成本
            pickup_distance = distance_matrix[vehicle_pos_idx, origin_idx]
            service_distance = distance_matrix[origin_idx, dest_idx]

            if pickup_distance == np.inf or service_distance == np.inf:
                return np.inf

            # 计算时间成本
            pickup_time = vehicle.initial_time + pickup_distance / vehicle.speed
            delivery_time = pickup_time + service_distance / vehicle.speed

            # 检查时间约束
            if pickup_time > request.pickup_deadline or delivery_time > request.delivery_deadline:
                return np.inf

            # 计算总成本
            travel_cost = (pickup_distance + service_distance) * vehicle.operational_cost
            delay_penalty = max(0, pickup_time - request.request_time) * self.config.get('cost.delay_penalty', 1.0)

            return travel_cost + delay_penalty

        except Exception as e:
            self.logger.error(f"成本计算失败: {str(e)}")
            return np.inf


class OnlineInstanceGenerator(AbstractOptimizer):
    """
    在线实例生成器

    从完全信息解决方案生成在线决策实例
    用于训练结构化学习模型
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.revelation_strategy = config.get('online.revelation_strategy', 'time_based')
        self.lookahead_window = config.get('online.lookahead_window', 5.0)

    def generate(self, demands: List[Tuple[int, int, float]],
                 vehicles: List[Tuple[int, float]],
                 graph: nx.DiGraph,
                 full_information_solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成在线实例

        Args:
            demands: 需求列表
            vehicles: 车辆列表
            graph: 路网图
            full_information_solution: 完全信息解决方案

        Returns:
            在线实例解决方案
        """
        try:
            self.logger.info("开始生成在线实例")

            # 检查完全信息解是否成功
            if not full_information_solution.get('success', False):
                return {'success': False, 'error': '完全信息解决方案无效'}

            # 构建请求和车辆
            requests = self._build_requests_from_demands(demands)
            fleet = self._build_vehicles_from_list(vehicles)

            # 根据启示策略生成在线序列
            if self.revelation_strategy == 'time_based':
                online_solution = self._generate_time_based_sequence(
                    requests, fleet, graph, full_information_solution
                )
            elif self.revelation_strategy == 'batch_based':
                online_solution = self._generate_batch_based_sequence(
                    requests, fleet, graph, full_information_solution
                )
            else:
                online_solution = self._generate_greedy_online_sequence(
                    requests, fleet, graph, full_information_solution
                )

            return online_solution

        except Exception as e:
            self.logger.error(f"在线实例生成失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _build_requests_from_demands(self, demands: List[Tuple[int, int, float]]) -> List[Dict[str, Any]]:
        """从需求列表构建请求对象"""
        requests = []
        for i, (origin, dest, req_time) in enumerate(demands):
            request = {
                'id': i,
                'origin': origin,
                'destination': dest,
                'request_time': req_time,
                'revealed': False,
                'served': False
            }
            requests.append(request)
        return requests

    def _build_vehicles_from_list(self, vehicles: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """从车辆列表构建车辆对象"""
        fleet = []
        for i, (position, initial_time) in enumerate(vehicles):
            vehicle = {
                'id': i,
                'position': position,
                'time': initial_time,
                'busy': False,
                'assigned_request': None
            }
            fleet.append(vehicle)
        return fleet

    def _generate_time_based_sequence(self, requests: List[Dict[str, Any]],
                                      fleet: List[Dict[str, Any]],
                                      graph: nx.DiGraph,
                                      full_info_solution: Dict[str, Any]) -> Dict[str, Any]:
        """生成基于时间的在线决策序列"""
        try:
            # 按请求时间排序
            requests_sorted = sorted(requests, key=lambda r: r['request_time'])

            decision_sequence = []
            current_time = min(r['request_time'] for r in requests_sorted)
            end_time = max(r['request_time'] for r in requests_sorted) + self.lookahead_window

            assignments = full_info_solution['solution_details'].get('assignments', {})

            # 时间步进循环
            time_step = 0.5  # 0.5分钟步长
            while current_time <= end_time:
                # 揭示当前时间窗口内的请求
                newly_revealed = []
                for request in requests_sorted:
                    if (not request['revealed'] and
                            request['request_time'] <= current_time + self.lookahead_window):
                        request['revealed'] = True
                        newly_revealed.append(request)

                # 如果有新请求，做出决策
                if newly_revealed:
                    decisions = self._make_online_decisions(
                        newly_revealed, fleet, assignments, current_time
                    )
                    decision_sequence.append({
                        'time': current_time,
                        'revealed_requests': [r['id'] for r in newly_revealed],
                        'decisions': decisions
                    })

                current_time += time_step

            # 计算在线解决方案的质量
            total_cost, num_served = self._evaluate_online_solution(decision_sequence, requests, fleet)

            return {
                'success': True,
                'decision_sequence': decision_sequence,
                'total_cost': total_cost,
                'num_served': num_served,
                'service_rate': num_served / len(requests) if requests else 0,
                'decision_variables': self._extract_decision_variables(decision_sequence),
                'method': 'time_based_online'
            }

        except Exception as e:
            self.logger.error(f"基于时间的在线序列生成失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _make_online_decisions(self, newly_revealed: List[Dict[str, Any]],
                               fleet: List[Dict[str, Any]],
                               assignments: Dict[int, int],
                               current_time: float) -> List[Dict[str, Any]]:
        """做出在线决策"""
        decisions = []

        for request in newly_revealed:
            # 查找完全信息解中的分配
            assigned_vehicle_id = None
            for vehicle_id, request_id in assignments.items():
                if request_id == request['id']:
                    assigned_vehicle_id = vehicle_id
                    break

            if assigned_vehicle_id is not None:
                # 检查车辆是否可用
                vehicle = next((v for v in fleet if v['id'] == assigned_vehicle_id), None)
                if vehicle and not vehicle['busy']:
                    # 分配车辆
                    vehicle['busy'] = True
                    vehicle['assigned_request'] = request['id']
                    request['served'] = True

                    decision = {
                        'request_id': request['id'],
                        'vehicle_id': assigned_vehicle_id,
                        'action': 'assign',
                        'cost': self._estimate_service_cost(request, vehicle)
                    }
                    decisions.append(decision)
                else:
                    # 车辆不可用，拒绝请求
                    decision = {
                        'request_id': request['id'],
                        'vehicle_id': None,
                        'action': 'reject',
                        'cost': 0
                    }
                    decisions.append(decision)
            else:
                # 完全信息解中未分配此请求，直接拒绝
                decision = {
                    'request_id': request['id'],
                    'vehicle_id': None,
                    'action': 'reject',
                    'cost': 0
                }
                decisions.append(decision)

        return decisions

    def _estimate_service_cost(self, request: Dict[str, Any], vehicle: Dict[str, Any]) -> float:
        """估算服务成本（简化版本）"""
        # 简化的距离估算
        origin_dist = abs(request['origin'] - vehicle['position'])
        service_dist = abs(request['destination'] - request['origin'])
        return origin_dist + service_dist

    def _evaluate_online_solution(self, decision_sequence: List[Dict[str, Any]],
                                  requests: List[Dict[str, Any]],
                                  fleet: List[Dict[str, Any]]) -> Tuple[float, int]:
        """评估在线解决方案的质量"""
        total_cost = 0
        num_served = 0

        for step in decision_sequence:
            for decision in step['decisions']:
                if decision['action'] == 'assign':
                    total_cost += decision['cost']
                    num_served += 1

        return total_cost, num_served

    def _extract_decision_variables(self, decision_sequence: List[Dict[str, Any]]) -> List[float]:
        """提取决策变量用于学习"""
        decision_vars = []

        for step in decision_sequence:
            for decision in step['decisions']:
                # 编码决策：分配=1, 拒绝=0
                decision_var = 1.0 if decision['action'] == 'assign' else 0.0
                decision_vars.append(decision_var)

        return decision_vars

    def _generate_batch_based_sequence(self, requests: List[Dict[str, Any]],
                                       fleet: List[Dict[str, Any]],
                                       graph: nx.DiGraph,
                                       full_info_solution: Dict[str, Any]) -> Dict[str, Any]:
        """生成基于批次的在线决策序列"""
        # 简化实现：将请求分批处理
        batch_size = self.config.get('online.batch_size', 5)
        decision_sequence = []

        # 按时间排序请求
        requests_sorted = sorted(requests, key=lambda r: r['request_time'])

        # 分批处理
        for i in range(0, len(requests_sorted), batch_size):
            batch = requests_sorted[i:i + batch_size]
            batch_time = batch[0]['request_time']

            # 为批次做出决策
            decisions = self._make_online_decisions(
                batch, fleet, full_info_solution['solution_details'].get('assignments', {}), batch_time
            )

            decision_sequence.append({
                'time': batch_time,
                'revealed_requests': [r['id'] for r in batch],
                'decisions': decisions
            })

        total_cost, num_served = self._evaluate_online_solution(decision_sequence, requests, fleet)

        return {
            'success': True,
            'decision_sequence': decision_sequence,
            'total_cost': total_cost,
            'num_served': num_served,
            'service_rate': num_served / len(requests) if requests else 0,
            'decision_variables': self._extract_decision_variables(decision_sequence),
            'method': 'batch_based_online'
        }

    def _generate_greedy_online_sequence(self, requests: List[Dict[str, Any]],
                                         fleet: List[Dict[str, Any]],
                                         graph: nx.DiGraph,
                                         full_info_solution: Dict[str, Any]) -> Dict[str, Any]:
        """生成贪婪在线决策序列"""
        try:
            decision_sequence = []
            requests_sorted = sorted(requests, key=lambda r: r['request_time'])

            for request in requests_sorted:
                # 找到最近的可用车辆
                available_vehicles = [v for v in fleet if not v['busy']]
                if not available_vehicles:
                    # 没有可用车辆，拒绝请求
                    decision = {
                        'request_id': request['id'],
                        'vehicle_id': None,
                        'action': 'reject',
                        'cost': 0
                    }
                else:
                    # 选择最近的车辆
                    best_vehicle = min(available_vehicles,
                                       key=lambda v: abs(v['position'] - request['origin']))

                    best_vehicle['busy'] = True
                    best_vehicle['assigned_request'] = request['id']
                    request['served'] = True

                    decision = {
                        'request_id': request['id'],
                        'vehicle_id': best_vehicle['id'],
                        'action': 'assign',
                        'cost': self._estimate_service_cost(request, best_vehicle)
                    }

                decision_sequence.append({
                    'time': request['request_time'],
                    'revealed_requests': [request['id']],
                    'decisions': [decision]
                })

            total_cost, num_served = self._evaluate_online_solution(decision_sequence, requests, fleet)

            return {
                'success': True,
                'decision_sequence': decision_sequence,
                'total_cost': total_cost,
                'num_served': num_served,
                'service_rate': num_served / len(requests) if requests else 0,
                'decision_variables': self._extract_decision_variables(decision_sequence),
                'method': 'greedy_online'
            }

        except Exception as e:
            self.logger.error(f"贪婪在线序列生成失败: {str(e)}")
            return {'success': False, 'error': str(e)}


# 性能优化工具函数

@jit(nopython=True)
def compute_assignment_cost_matrix(vehicle_positions: np.ndarray,
                                   request_origins: np.ndarray,
                                   request_destinations: np.ndarray,
                                   distance_matrix: np.ndarray,
                                   position_to_idx: np.ndarray) -> np.ndarray:
    """
    计算车辆-请求分配成本矩阵 (Numba加速)

    Args:
        vehicle_positions: 车辆位置数组
        request_origins: 请求起点数组
        request_destinations: 请求终点数组
        distance_matrix: 距离矩阵
        position_to_idx: 位置到索引的映射

    Returns:
        成本矩阵
    """
    n_vehicles = len(vehicle_positions)
    n_requests = len(request_origins)
    cost_matrix = np.full((n_vehicles, n_requests), np.inf)

    for i in range(n_vehicles):
        vehicle_pos_idx = position_to_idx[vehicle_positions[i]]

        for j in range(n_requests):
            origin_idx = position_to_idx[request_origins[j]]
            dest_idx = position_to_idx[request_destinations[j]]

            # 计算接客距离 + 服务距离
            pickup_dist = distance_matrix[vehicle_pos_idx, origin_idx]
            service_dist = distance_matrix[origin_idx, dest_idx]

            if pickup_dist < np.inf and service_dist < np.inf:
                cost_matrix[i, j] = pickup_dist + service_dist

    return cost_matrix


@jit(nopython=True)
def greedy_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    贪婪分配算法 (Numba加速)

    Args:
        cost_matrix: 成本矩阵

    Returns:
        (分配数组, 总成本)
    """
    n_vehicles, n_requests = cost_matrix.shape
    assignments = np.full(n_vehicles, -1)  # -1表示未分配
    used_requests = np.zeros(n_requests, dtype=np.bool_)
    total_cost = 0.0

    # 创建成本-索引对列表并排序
    costs_with_indices = []
    for i in range(n_vehicles):
        for j in range(n_requests):
            if cost_matrix[i, j] < np.inf:
                costs_with_indices.append((cost_matrix[i, j], i, j))

    # 简单排序 (Numba不支持复杂排序)
    # 使用冒泡排序
    n = len(costs_with_indices)
    for i in range(n):
        for j in range(0, n - i - 1):
            if costs_with_indices[j][0] > costs_with_indices[j + 1][0]:
                costs_with_indices[j], costs_with_indices[j + 1] = costs_with_indices[j + 1], costs_with_indices[j]

    # 贪婪分配
    for cost, vehicle_idx, request_idx in costs_with_indices:
        if assignments[vehicle_idx] == -1 and not used_requests[request_idx]:
            assignments[vehicle_idx] = request_idx
            used_requests[request_idx] = True
            total_cost += cost

    return assignments, total_cost


class MultiObjectiveOptimizer:
    """
    多目标优化器

    同时优化多个目标：成本、服务质量、公平性等
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.objectives = config.get('multi_objective.objectives', ['cost', 'service_rate'])
        self.weights = config.get('multi_objective.weights', [0.7, 0.3])

    def solve_pareto_optimal(self, demands: List[Tuple[int, int, float]],
                             vehicles: List[Tuple[int, float]],
                             graph: nx.DiGraph) -> List[OptimizationResult]:
        """求解帕累托最优解集"""
        try:
            # 生成多个解决方案（不同权重组合）
            solutions = []

            # 测试不同的权重组合
            weight_combinations = [
                [1.0, 0.0],  # 纯成本优化
                [0.8, 0.2],  # 成本主导
                [0.6, 0.4],  # 平衡
                [0.4, 0.6],  # 服务主导
                [0.0, 1.0]  # 纯服务优化
            ]

            for weights in weight_combinations:
                # 更新权重
                temp_config = self.config.copy()
                temp_config['multi_objective.weights'] = weights

                # 求解
                solver = FullInformationSolver(temp_config)
                solution = solver.solve(demands, vehicles, graph)

                if solution.success:
                    solution.solution_details['weights'] = weights
                    solutions.append(solution)

            # 过滤帕累托最优解
            pareto_solutions = self._filter_pareto_optimal(solutions)

            return pareto_solutions

        except Exception as e:
            logger.error(f"多目标优化失败: {str(e)}")
            return []

    def _filter_pareto_optimal(self, solutions: List[OptimizationResult]) -> List[OptimizationResult]:
        """过滤帕累托最优解"""
        if not solutions:
            return []

        pareto_solutions = []

        for i, sol_i in enumerate(solutions):
            is_dominated = False

            for j, sol_j in enumerate(solutions):
                if i != j:
                    # 检查sol_i是否被sol_j支配
                    if (sol_j.total_cost <= sol_i.total_cost and
                            sol_j.service_rate >= sol_i.service_rate and
                            (sol_j.total_cost < sol_i.total_cost or sol_j.service_rate > sol_i.service_rate)):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_solutions.append(sol_i)

        return pareto_solutions


class RebalancingOptimizer:
    """
    车辆重新平衡优化器

    优化空闲车辆的重新定位以满足未来需求
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prediction_horizon = config.get('rebalancing.prediction_horizon', 15.0)  # 分钟
        self.rebalancing_cost = config.get('rebalancing.cost_per_km', 0.5)

    def optimize_rebalancing(self, idle_vehicles: List[Dict[str, Any]],
                             demand_forecast: List[Tuple[int, float]],
                             graph: nx.DiGraph) -> Dict[str, Any]:
        """
        优化车辆重新平衡

        Args:
            idle_vehicles: 空闲车辆列表
            demand_forecast: 需求预测 [(zone, demand_intensity), ...]
            graph: 路网图

        Returns:
            重新平衡方案
        """
        try:
            logger.info(f"开始车辆重新平衡优化: {len(idle_vehicles)}辆空闲车辆")

            # 构建供需不平衡
            supply_demand_gap = self._compute_supply_demand_gap(idle_vehicles, demand_forecast)

            # 构建重新平衡网络
            rebalancing_graph = self._build_rebalancing_network(
                idle_vehicles, supply_demand_gap, graph
            )

            # 求解最小费用流问题
            rebalancing_plan = self._solve_rebalancing_flow(rebalancing_graph)

            return rebalancing_plan

        except Exception as e:
            logger.error(f"车辆重新平衡优化失败: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _compute_supply_demand_gap(self, idle_vehicles: List[Dict[str, Any]],
                                   demand_forecast: List[Tuple[int, float]]) -> Dict[int, float]:
        """计算供需缺口"""
        # 统计各区域的车辆供给
        supply = defaultdict(int)
        for vehicle in idle_vehicles:
            supply[vehicle['position']] += 1

        # 计算供需差
        supply_demand_gap = {}
        for zone, demand in demand_forecast:
            gap = supply[zone] - demand
            supply_demand_gap[zone] = gap

        return supply_demand_gap

    def _build_rebalancing_network(self, idle_vehicles: List[Dict[str, Any]],
                                   supply_demand_gap: Dict[int, float],
                                   graph: nx.DiGraph) -> nx.DiGraph:
        """构建重新平衡网络"""
        rebalancing_graph = nx.DiGraph()

        # 添加源点和汇点
        source = 'source'
        sink = 'sink'
        rebalancing_graph.add_node(source)
        rebalancing_graph.add_node(sink)

        # 添加车辆节点（供给）
        for vehicle in idle_vehicles:
            vehicle_node = f'v_{vehicle["id"]}'
            rebalancing_graph.add_node(vehicle_node)
            rebalancing_graph.add_edge(source, vehicle_node, capacity=1, weight=0)

        # 添加需求节点
        for zone, gap in supply_demand_gap.items():
            if gap < 0:  # 需求大于供给
                demand_node = f'demand_{zone}'
                rebalancing_graph.add_node(demand_node)
                rebalancing_graph.add_edge(demand_node, sink, capacity=int(abs(gap)), weight=0)

                # 连接车辆到需求区域
                for vehicle in idle_vehicles:
                    vehicle_node = f'v_{vehicle["id"]}'
                    try:
                        # 计算重新平衡成本
                        distance = nx.shortest_path_length(graph, vehicle['position'], zone, weight='weight')
                        cost = distance * self.rebalancing_cost
                        rebalancing_graph.add_edge(vehicle_node, demand_node, capacity=1, weight=cost)
                    except nx.NetworkXNoPath:
                        # 无法到达，不添加边
                        continue

        return rebalancing_graph

    def _solve_rebalancing_flow(self, rebalancing_graph: nx.DiGraph) -> Dict[str, Any]:
        """求解重新平衡流问题"""
        try:
            flow_cost, flow_dict = nx.network_simplex(rebalancing_graph)

            # 解析重新平衡计划
            rebalancing_moves = []
            total_cost = 0

            for vehicle_node, targets in flow_dict.items():
                if vehicle_node.startswith('v_'):
                    vehicle_id = int(vehicle_node.split('_')[1])

                    for target_node, flow in targets.items():
                        if flow > 0 and target_node.startswith('demand_'):
                            target_zone = int(target_node.split('_')[1])

                            # 获取移动成本
                            edge_data = rebalancing_graph[vehicle_node][target_node]
                            move_cost = edge_data['weight']

                            rebalancing_moves.append({
                                'vehicle_id': vehicle_id,
                                'target_zone': target_zone,
                                'cost': move_cost
                            })
                            total_cost += move_cost

            return {
                'success': True,
                'rebalancing_moves': rebalancing_moves,
                'total_cost': total_cost,
                'num_moves': len(rebalancing_moves)
            }

        except Exception as e:
            logger.error(f"重新平衡流问题求解失败: {str(e)}")
            return {'success': False, 'error': str(e)}


# 工厂类
class OptimizerFactory:
    """优化器工厂类"""

    @staticmethod
    def create_optimizer(optimizer_type: str, config: Dict[str, Any]) -> AbstractOptimizer:
        """创建优化器实例"""
        if optimizer_type == 'full_information':
            return FullInformationSolver(config)
        elif optimizer_type == 'online_generator':
            return OnlineInstanceGenerator(config)
        elif optimizer_type == 'multi_objective':
            return MultiObjectiveOptimizer(config)
        elif optimizer_type == 'rebalancing':
            return RebalancingOptimizer(config)
        else:
            raise ValueError(f"未知的优化器类型: {optimizer_type}")


# 性能基准测试
def benchmark_optimizers(demands: List[Tuple[int, int, float]],
                         vehicles: List[Tuple[int, float]],
                         graph: nx.DiGraph,
                         config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """对不同优化器进行基准测试"""
    results = {}

    # 测试不同求解器
    solver_types = ['min_cost_flow', 'hungarian', 'greedy']

    for solver_type in solver_types:
        test_config = config.copy()
        test_config['optimization.solver_type'] = solver_type

        optimizer = FullInformationSolver(test_config)

        # 多次运行取平均
        times = []
        costs = []
        service_rates = []

        for _ in range(5):
            result = optimizer.solve(demands, vehicles, graph)
            if result.success:
                times.append(result.computation_time)
                costs.append(result.total_cost)
                service_rates.append(result.service_rate)

        if times:
            results[solver_type] = {
                'avg_time': np.mean(times),
                'avg_cost': np.mean(costs),
                'avg_service_rate': np.mean(service_rates),
                'std_time': np.std(times)
            }

    return results