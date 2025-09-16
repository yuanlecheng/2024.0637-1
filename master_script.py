#!/usr/bin/env python3


import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import multiprocessing as mp

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.experiment_config import ExperimentConfig
from config.system_config import SystemConfig
from src.utils.logging_utils import setup_logging
from src.utils.system_utils import get_system_info, check_dependencies

class MasterController:
    """主控制器类 - 管理整个ML-CO流水线"""

    def __init__(self, experiment_type: str, config_path: Optional[str] = None):
        self.experiment_type = experiment_type
        self.project_root = PROJECT_ROOT
        self.config = ExperimentConfig(experiment_type, config_path)
        self.system_config = SystemConfig()

        # 设置日志
        self.logger = setup_logging(
            log_level=self.config.get('logging.level', 'INFO'),
            log_file=self.project_root / 'logs' / f'{experiment_type}_master.log'
        )

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            'logs', 'results', 'data/processed', 'data/temp',
            'results/training', 'results/evaluation', 'results/benchmarks'
        ]

        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"创建目录结构完成: {len(directories)} 个目录")

    def run_create_instances(self) -> bool:
        """
        创建训练实例
        替换原始的run_bash_createInstances.cmd
        """
        self.logger.info("开始创建训练实例...")

        try:
            from scripts.create_training_instances import TrainingInstanceCreator

            creator = TrainingInstanceCreator(
                experiment_type=self.experiment_type,
                config=self.config,
                project_root=self.project_root
            )

            success = creator.run()

            if success:
                self.logger.info("训练实例创建成功")
                return True
            else:
                self.logger.error("训练实例创建失败")
                return False

        except Exception as e:
            self.logger.error(f"创建训练实例时出错: {str(e)}")
            return False

    def run_training(self) -> bool:
        """
        运行训练过程
        替换原始的run_bash_training.cmd
        """
        self.logger.info("开始训练过程...")

        try:
            from scripts.run_training import TrainingRunner

            trainer = TrainingRunner(
                experiment_type=self.experiment_type,
                config=self.config,
                project_root=self.project_root
            )

            success = trainer.run()

            if success:
                self.logger.info("训练过程完成")
                return True
            else:
                self.logger.error("训练过程失败")
                return False

        except Exception as e:
            self.logger.error(f"训练过程中出错: {str(e)}")
            return False

    def run_benchmarks(self) -> bool:
        """
        运行基准测试
        替换原始的run_bash_benchmarks.cmd
        """
        self.logger.info("开始基准测试...")

        try:
            from scripts.run_benchmarks import BenchmarkRunner

            benchmark_runner = BenchmarkRunner(
                experiment_type=self.experiment_type,
                config=self.config,
                project_root=self.project_root
            )

            success = benchmark_runner.run()

            if success:
                self.logger.info("基准测试完成")
                return True
            else:
                self.logger.error("基准测试失败")
                return False

        except Exception as e:
            self.logger.error(f"基准测试中出错: {str(e)}")
            return False

    def run_evaluation(self) -> bool:
        """运行完整评估"""
        self.logger.info("开始完整评估...")

        try:
            from scripts.evaluation_runner import EvaluationRunner

            evaluator = EvaluationRunner(
                experiment_type=self.experiment_type,
                config=self.config,
                project_root=self.project_root
            )

            success = evaluator.run()

            if success:
                self.logger.info("完整评估完成")
                return True
            else:
                self.logger.error("完整评估失败")
                return False

        except Exception as e:
            self.logger.error(f"完整评估中出错: {str(e)}")
            return False

    def run_full_pipeline(self) -> bool:
        """运行完整的流水线"""
        self.logger.info(f"开始运行完整流水线 - 实验类型: {self.experiment_type}")

        # 检查系统依赖
        if not check_dependencies():
            self.logger.error("系统依赖检查失败")
            return False

        # 显示系统信息
        system_info = get_system_info()
        self.logger.info(f"系统信息: {system_info}")

        pipeline_steps = [
            ("创建训练实例", self.run_create_instances),
            ("训练模型", self.run_training),
            ("运行基准测试", self.run_benchmarks),
            ("完整评估", self.run_evaluation)
        ]

        for step_name, step_func in pipeline_steps:
            self.logger.info(f"执行步骤: {step_name}")

            if not step_func():
                self.logger.error(f"步骤 '{step_name}' 执行失败，终止流水线")
                return False

            self.logger.info(f"步骤 '{step_name}' 执行成功")

        self.logger.info("完整流水线执行成功！")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ML-CO-pipeline-AMoD-control Master Script - Windows版本"
    )

    parser.add_argument(
        '--experiment-type', '-e',
        type=str,
        required=True,
        choices=['small', 'medium', 'large', 'test'],
        help="实验类型 (对应原RUNNING_TYPE变量)"
    )

    parser.add_argument(
        '--step', '-s',
        type=str,
        choices=['create-instances', 'training', 'benchmarks', 'evaluation', 'full'],
        default='full',
        help="要执行的步骤"
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help="配置文件路径"
    )

    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=None,
        help="并行进程数 (默认使用所有CPU核心)"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="详细输出"
    )

    args = parser.parse_args()

    # 设置并行进程数
    if args.parallel is not None:
        mp.set_start_method('spawn', force=True)  # Windows兼容
        os.environ['OMP_NUM_THREADS'] = str(args.parallel)

    # 创建主控制器
    try:
        controller = MasterController(args.experiment_type, args.config)
    except Exception as e:
        print(f"初始化主控制器失败: {str(e)}")
        return 1

    # 执行相应步骤
    success = False

    if args.step == 'create-instances':
        success = controller.run_create_instances()
    elif args.step == 'training':
        success = controller.run_training()
    elif args.step == 'benchmarks':
        success = controller.run_benchmarks()
    elif args.step == 'evaluation':
        success = controller.run_evaluation()
    elif args.step == 'full':
        success = controller.run_full_pipeline()

    return 0 if success else 1

if __name__ == '__main__':
    # Windows特定设置
    if sys.platform == 'win32':
        # 设置控制台编码为UTF-8
        import locale
        if locale.getpreferredencoding().upper() != 'UTF-8':
            os.environ['PYTHONIOENCODING'] = 'utf-8'

    sys.exit(main())