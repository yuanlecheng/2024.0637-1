#%%
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    path: str
    taxi_data_dir: str
    preprocessed_dir: str
    time_horizon_days: int
    geographical_bounds: Dict[str, float]
    demand_scaling_factor: float
    vehicle_capacity: int


@dataclass
class ModelConfig:
    """模型配置"""
    learning_rate: float
    batch_size: int
    num_epochs: int
    hidden_layers: list
    activation_function: str
    optimizer: str
    loss_function: str
    regularization_lambda: float
    dropout_rate: float


@dataclass
class OptimizationConfig:
    """优化配置"""
    solver_type: str  # 'dinic', 'ford_fulkerson', 'edmonds_karp'
    k_paths: int
    time_limit_seconds: int
    memory_limit_gb: float
    gap_tolerance: float
    parallel_threads: int


@dataclass
class EvaluationConfig:
    """评估配置"""
    validation_split: float
    test_split: float
    metrics: list
    benchmark_policies: list
    evaluation_episodes: int
    save_detailed_results: bool


class ExperimentConfig:
    """
    实验配置管理器

    替换原始项目中的bash变量和配置文件系统
    支持不同的实验类型：small, medium, large, test
    """

    # 预定义的实验配置
    EXPERIMENT_CONFIGS = {
        'test': {
            'description': '快速测试配置',
            'dataset': DatasetConfig(
                name='manhattan_test',
                path='data/taxi_data_Manhattan_2015_preprocessed/test',
                taxi_data_dir='data/taxi_data_Manhattan_2015_preprocessed/month-01',
                preprocessed_dir='data/processed/test',
                time_horizon_days=1,
                geographical_bounds={
                    'min_lat': 40.700, 'max_lat': 40.800,
                    'min_lon': -74.020, 'max_lon': -73.920
                },
                demand_scaling_factor=0.1,
                vehicle_capacity=4
            ),
            'model': ModelConfig(
                learning_rate=0.001,
                batch_size=32,
                num_epochs=10,
                hidden_layers=[64, 32],
                activation_function='relu',
                optimizer='adam',
                loss_function='mse',
                regularization_lambda=0.001,
                dropout_rate=0.1
            ),
            'optimization': OptimizationConfig(
                solver_type='dinic',
                k_paths=2,
                time_limit_seconds=60,
                memory_limit_gb=2.0,
                gap_tolerance=0.01,
                parallel_threads=2
            ),
            'evaluation': EvaluationConfig(
                validation_split=0.2,
                test_split=0.2,
                metrics=['total_cost', 'computation_time', 'success_rate'],
                benchmark_policies=['greedy', 'random', 'offline'],
                evaluation_episodes=10,
                save_detailed_results=True
            )
        },

        'small': {
            'description': '小规模实验配置',
            'dataset': DatasetConfig(
                name='manhattan_small',
                path='data/taxi_data_Manhattan_2015_preprocessed/small',
                taxi_data_dir='data/taxi_data_Manhattan_2015_preprocessed/month-01',
                preprocessed_dir='data/processed/small',
                time_horizon_days=7,
                geographical_bounds={
                    'min_lat': 40.700, 'max_lat': 40.850,
                    'min_lon': -74.050, 'max_lon': -73.900
                },
                demand_scaling_factor=0.3,
                vehicle_capacity=4
            ),
            'model': ModelConfig(
                learning_rate=0.001,
                batch_size=64,
                num_epochs=50,
                hidden_layers=[128, 64, 32],
                activation_function='relu',
                optimizer='adam',
                loss_function='structured_loss',
                regularization_lambda=0.001,
                dropout_rate=0.2
            ),
            'optimization': OptimizationConfig(
                solver_type='dinic',
                k_paths=3,
                time_limit_seconds=300,
                memory_limit_gb=4.0,
                gap_tolerance=0.005,
                parallel_threads=4
            ),
            'evaluation': EvaluationConfig(
                validation_split=0.15,
                test_split=0.15,
                metrics=['total_cost', 'computation_time', 'success_rate', 'path_quality'],
                benchmark_policies=['greedy', 'random', 'offline', 'sampling'],
                evaluation_episodes=50,
                save_detailed_results=True
            )
        },

        'medium': {
            'description': '中等规模实验配置',
            'dataset': DatasetConfig(
                name='manhattan_medium',
                path='data/taxi_data_Manhattan_2015_preprocessed/medium',
                taxi_data_dir='data/taxi_data_Manhattan_2015_preprocessed/month-01',
                preprocessed_dir='data/processed/medium',
                time_horizon_days=30,
                geographical_bounds={
                    'min_lat': 40.680, 'max_lat': 40.900,
                    'min_lon': -74.100, 'max_lon': -73.850
                },
                demand_scaling_factor=0.7,
                vehicle_capacity=4
            ),
            'model': ModelConfig(
                learning_rate=0.0005,
                batch_size=128,
                num_epochs=100,
                hidden_layers=[256, 128, 64, 32],
                activation_function='relu',
                optimizer='adamw',
                loss_function='structured_loss',
                regularization_lambda=0.0005,
                dropout_rate=0.3
            ),
            'optimization': OptimizationConfig(
                solver_type='dinic',
                k_paths=4,
                time_limit_seconds=900,
                memory_limit_gb=8.0,
                gap_tolerance=0.002,
                parallel_threads=8
            ),
            'evaluation': EvaluationConfig(
                validation_split=0.1,
                test_split=0.1,
                metrics=['total_cost', 'computation_time', 'success_rate',
                         'path_quality', 'vehicle_utilization'],
                benchmark_policies=['greedy', 'random', 'offline', 'sampling', 'policy_SB'],
                evaluation_episodes=100,
                save_detailed_results=True
            )
        },

        'large': {
            'description': '大规模实验配置（论文复现）',
            'dataset': DatasetConfig(
                name='manhattan_full',
                path='data/taxi_data_Manhattan_2015_preprocessed/full',
                taxi_data_dir='data/taxi_data_Manhattan_2015_preprocessed',
                preprocessed_dir='data/processed/full',
                time_horizon_days=365,
                geographical_bounds={
                    'min_lat': 40.477399, 'max_lat': 40.917577,
                    'min_lon': -74.259090, 'max_lon': -73.700272
                },
                demand_scaling_factor=1.0,
                vehicle_capacity=4
            ),
            'model': ModelConfig(
                learning_rate=0.0001,
                batch_size=256,
                num_epochs=200,
                hidden_layers=[512, 256, 128, 64, 32],
                activation_function='relu',
                optimizer='adamw',
                loss_function='structured_loss',
                regularization_lambda=0.0001,
                dropout_rate=0.4
            ),
            'optimization': OptimizationConfig(
                solver_type='dinic',
                k_paths=5,
                time_limit_seconds=3600,
                memory_limit_gb=16.0,
                gap_tolerance=0.001,
                parallel_threads=16
            ),
            'evaluation': EvaluationConfig(
                validation_split=0.1,
                test_split=0.1,
                metrics=['total_cost', 'computation_time', 'success_rate',
                         'path_quality', 'vehicle_utilization', 'demand_satisfaction'],
                benchmark_policies=['greedy', 'random', 'offline', 'sampling',
                                    'policy_SB', 'policy_CB'],
                evaluation_episodes=200,
                save_detailed_results=True
            )
        }
    }

    def __init__(self, experiment_type: str, custom_config_path: Optional[str] = None):
        """
        初始化实验配置

        Args:
            experiment_type: 实验类型 ('test', 'small', 'medium', 'large')
            custom_config_path: 自定义配置文件路径
        """
        self.experiment_type = experiment_type

        if custom_config_path and Path(custom_config_path).exists():
            self.config = self._load_custom_config(custom_config_path)
        elif experiment_type in self.EXPERIMENT_CONFIGS:
            self.config = self._create_config_from_template(experiment_type)
        else:
            raise ValueError(f"未知的实验类型: {experiment_type}。"
                             f"支持的类型: {list(self.EXPERIMENT_CONFIGS.keys())}")

        logger.info(f"加载实验配置: {experiment_type}")

    def _create_config_from_template(self, experiment_type: str) -> Dict[str, Any]:
        """从模板创建配置"""
        template = self.EXPERIMENT_CONFIGS[experiment_type]
        config = {}

        for key, value in template.items():
            if hasattr(value, '__dict__'):  # dataclass实例
                config[key] = asdict(value)
            else:
                config[key] = value

        return config

    def _load_custom_config(self, config_path: str) -> Dict[str, Any]:
        """加载自定义配置文件"""
        config_path = Path(config_path)

        try:
            if config_path.suffix == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号分隔的层级访问

        Args:
            key: 配置键，支持 'dataset.name' 形式的层级访问
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        设置配置值，支持点号分隔的层级设置

        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self.config

        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value

        logger.debug(f"设置配置: {key} = {value}")

    def save(self, save_path: str, format: str = 'json'):
        """
        保存配置到文件

        Args:
            save_path: 保存路径
            format: 保存格式 ('json' 或 'yaml')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == 'json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            elif format == 'yaml':
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False,
                              allow_unicode=True, indent=2)
            else:
                raise ValueError(f"不支持的保存格式: {format}")

            logger.info(f"配置已保存到: {save_path}")

        except Exception as e:
            logger.error(f"保存配置失败: {str(e)}")
            raise

    def update(self, updates: Dict[str, Any]):
        """
        批量更新配置

        Args:
            updates: 更新的配置字典
        """

        def deep_update(base_dict: Dict, update_dict: Dict):
            """递归更新字典"""
            for key, value in update_dict.items():
                if (key in base_dict and
                        isinstance(base_dict[key], dict) and
                        isinstance(value, dict)):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(self.config, updates)
        logger.info(f"批量更新配置: {len(updates)} 项")

    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            # 验证必需字段
            required_sections = ['dataset', 'model', 'optimization', 'evaluation']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"缺少必需的配置段: {section}")
                    return False

            # 验证数据集路径
            dataset_path = Path(self.get('dataset.path', ''))
            if not dataset_path.exists():
                logger.warning(f"数据集路径不存在: {dataset_path}")

            # 验证数值范围
            lr = self.get('model.learning_rate', 0)
            if not 0 < lr < 1:
                logger.error(f"学习率超出有效范围: {lr}")
                return False

            # 验证k_paths
            k_paths = self.get('optimization.k_paths', 0)
            if k_paths < 1:
                logger.error(f"k_paths必须大于0: {k_paths}")
                return False

            logger.info("配置验证通过")
            return True

        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'experiment_type': self.experiment_type,
            'description': self.get('description', ''),
            'dataset_name': self.get('dataset.name', ''),
            'time_horizon_days': self.get('dataset.time_horizon_days', 0),
            'num_epochs': self.get('model.num_epochs', 0),
            'k_paths': self.get('optimization.k_paths', 0),
            'evaluation_episodes': self.get('evaluation.evaluation_episodes', 0)
        }

    def __str__(self) -> str:
        """字符串表示"""
        summary = self.get_summary()
        return (f"ExperimentConfig({summary['experiment_type']}: "
                f"{summary['description']})")


# 配置验证器
class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_paths(config: ExperimentConfig):
        """验证路径配置"""
        errors = []

        paths_to_check = [
            'dataset.path',
            'dataset.taxi_data_dir',
            'dataset.preprocessed_dir'
        ]

        for path_key in paths_to_check:
            path_value = config.get(path_key)
            if path_value:
                path = Path(path_value)
                if not path.exists():
                    errors.append(f"路径不存在: {path_key} = {path_value}")

        return errors

    @staticmethod
    def validate_model_params(config: ExperimentConfig):
        """验证模型参数"""
        errors = []

        # 学习率检查
        lr = config.get('model.learning_rate', 0)
        if not 0 < lr < 1:
            errors.append(f"学习率超出范围 (0, 1): {lr}")

        # 批量大小检查
        batch_size = config.get('model.batch_size', 0)
        if batch_size < 1 or batch_size > 1024:
            errors.append(f"批量大小超出合理范围 [1, 1024]: {batch_size}")

        # 隐藏层检查
        hidden_layers = config.get('model.hidden_layers', [])
        if not isinstance(hidden_layers, list) or len(hidden_layers) == 0:
            errors.append("隐藏层配置无效")

        return errors