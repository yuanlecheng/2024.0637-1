#%%
import os
import sys
import platform
import psutil
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """系统信息"""
    platform: str
    platform_version: str
    python_version: str
    cpu_count: int
    memory_total_gb: float
    memory_available_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_info: Optional[str] = None


@dataclass
class ResourceLimits:
    """资源限制配置"""
    max_memory_gb: float
    max_cpu_cores: int
    max_disk_usage_gb: float
    enable_gpu: bool
    gpu_memory_fraction: float


class SystemConfig:
    """
    系统配置管理器

    处理Windows特定配置，替换原始项目中的Linux/slurm依赖
    """

    def __init__(self):
        """初始化系统配置"""
        self.platform = platform.system().lower()
        self.is_windows = self.platform == 'windows'
        self.is_linux = self.platform == 'linux'
        self.is_macos = self.platform == 'darwin'

        # 获取系统信息
        self.system_info = self._get_system_info()

        # 设置默认资源限制
        self.resource_limits = self._get_default_resource_limits()

        # Windows特定设置
        if self.is_windows:
            self._setup_windows_environment()

        logger.info(f"系统配置初始化完成: {self.platform}")

    def _get_system_info(self) -> SystemInfo:
        """获取系统信息"""
        # 内存信息
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)

        # 磁盘信息
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)

        # GPU信息
        gpu_available, gpu_info = self._detect_gpu()

        return SystemInfo(
            platform=self.platform,
            platform_version=platform.platform(),
            python_version=sys.version,
            cpu_count=mp.cpu_count(),
            memory_total_gb=memory_total_gb,
            memory_available_gb=memory_available_gb,
            disk_free_gb=disk_free_gb,
            gpu_available=gpu_available,
            gpu_info=gpu_info
        )

    def _detect_gpu(self) -> tuple[bool, Optional[str]]:
        """检测GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_name(0)
                return True, gpu_info
        except ImportError:
            pass

        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                return True, f"TensorFlow GPU: {len(gpus)} device(s)"
        except ImportError:
            pass

        return False, None

    def _get_default_resource_limits(self) -> ResourceLimits:
        """获取默认资源限制"""
        # 保守的资源使用策略
        max_memory_gb = min(self.system_info.memory_available_gb * 0.8, 16.0)
        max_cpu_cores = max(1, self.system_info.cpu_count - 1)
        max_disk_usage_gb = min(self.system_info.disk_free_gb * 0.5, 100.0)

        return ResourceLimits(
            max_memory_gb=max_memory_gb,
            max_cpu_cores=max_cpu_cores,
            max_disk_usage_gb=max_disk_usage_gb,
            enable_gpu=self.system_info.gpu_available,
            gpu_memory_fraction=0.8
        )

    def _setup_windows_environment(self):
        """设置Windows特定环境"""
        # 设置编码为UTF-8
        if 'PYTHONIOENCODING' not in os.environ:
            os.environ['PYTHONIOENCODING'] = 'utf-8'

        # 设置多进程启动方法
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)

        # 设置NumPy线程数
        if 'OMP_NUM_THREADS' not in os.environ:
            os.environ['OMP_NUM_THREADS'] = str(self.resource_limits.max_cpu_cores)

        # 设置临时目录
        temp_dir = Path.home() / 'AppData' / 'Local' / 'Temp' / 'amod_pipeline'
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TEMP_DIR'] = str(temp_dir)

        logger.info("Windows环境配置完成")

    def get_path_separator(self) -> str:
        """获取路径分隔符"""
        return '\\' if self.is_windows else '/'

    def normalize_path(self, path: str) -> Path:
        """标准化路径"""
        return Path(path).resolve()

    def get_executable_extension(self) -> str:
        """获取可执行文件扩展名"""
        return '.exe' if self.is_windows else ''

    def get_script_extension(self) -> str:
        """获取脚本文件扩展名"""
        return '.bat' if self.is_windows else '.sh'

    def create_temp_dir(self, prefix: str = 'amod_') -> Path:
        """创建临时目录"""
        import tempfile

        if self.is_windows:
            base_temp = Path(os.environ.get('TEMP_DIR', tempfile.gettempdir()))
        else:
            base_temp = Path(tempfile.gettempdir())

        temp_dir = base_temp / f"{prefix}{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        return temp_dir

    def run_command(self, command: List[str], cwd: Optional[Path] = None,
                    timeout: Optional[int] = None) -> tuple[int, str, str]:
        """
        运行系统命令（跨平台）

        Args:
            command: 命令列表
            cwd: 工作目录
            timeout: 超时时间（秒）

        Returns:
            (返回码, stdout, stderr)
        """
        import subprocess

        try:
            # Windows特定处理
            if self.is_windows:
                # 确保使用正确的shell
                if command[0].endswith('.py'):
                    command = [sys.executable] + command

                # 设置创建标志以避免控制台窗口
                creationflags = subprocess.CREATE_NO_WINDOW if self.is_windows else 0
            else:
                creationflags = 0

            result = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                capture_output=True,
                text=True,
                encoding='utf-8',
                creationflags=creationflags
            )

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return -1, "", f"命令超时: {timeout}秒"
        except Exception as e:
            return -1, "", f"命令执行失败: {str(e)}"

    def get_available_memory_gb(self) -> float:
        """获取当前可用内存（GB）"""
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)

    def get_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        return psutil.cpu_percent(interval=1)

    def get_disk_usage(self, path: Path) -> Dict[str, float]:
        """获取磁盘使用情况"""
        usage = psutil.disk_usage(str(path))
        return {
            'total_gb': usage.total / (1024 ** 3),
            'used_gb': usage.used / (1024 ** 3),
            'free_gb': usage.free / (1024 ** 3),
            'percent': (usage.used / usage.total) * 100
        }

    def check_resource_availability(self, required_memory_gb: float = 1.0,
                                    required_cpu_cores: int = 1) -> Dict[str, bool]:
        """检查资源可用性"""
        available_memory = self.get_available_memory_gb()
        cpu_usage = self.get_cpu_usage()

        results = {
            'memory_sufficient': available_memory >= required_memory_gb,
            'cpu_available': cpu_usage < 80.0,  # CPU使用率低于80%
            'cpu_cores_sufficient': self.system_info.cpu_count >= required_cpu_cores
        }

        results['all_sufficient'] = all(results.values())
        return results

    def optimize_for_computation(self):
        """为计算密集型任务优化系统设置"""
        # 设置进程优先级
        if self.is_windows:
            try:
                import psutil
                p = psutil.Process()
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.info("设置高优先级成功")
            except Exception as e:
                logger.warning(f"设置进程优先级失败: {str(e)}")

        # 优化NumPy设置
        try:
            import numpy as np
            # 使用所有可用核心
            os.environ['OMP_NUM_THREADS'] = str(self.resource_limits.max_cpu_cores)
            os.environ['MKL_NUM_THREADS'] = str(self.resource_limits.max_cpu_cores)
            logger.info(f"设置NumPy线程数: {self.resource_limits.max_cpu_cores}")
        except ImportError:
            pass

        # 优化PyTorch设置
        try:
            import torch
            torch.set_num_threads(self.resource_limits.max_cpu_cores)
            if self.system_info.gpu_available:
                torch.backends.cudnn.benchmark = True
                logger.info("启用PyTorch优化")
        except ImportError:
            pass

    def setup_logging_directory(self, base_path: Path) -> Path:
        """设置日志目录"""
        if self.is_windows:
            log_dir = base_path / 'logs'
        else:
            log_dir = base_path / 'logs'

        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_environment_variables(self) -> Dict[str, str]:
        """获取相关环境变量"""
        env_vars = {}

        # Python相关
        for var in ['PYTHONPATH', 'PYTHONIOENCODING', 'PYTHONHOME']:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        # 计算相关
        for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        # GPU相关
        for var in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER']:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        return env_vars

    def validate_dependencies(self) -> Dict[str, bool]:
        """验证系统依赖"""
        dependencies = {}

        # Python包依赖
        required_packages = [
            'numpy', 'scipy', 'pandas', 'networkx',
            'scikit-learn', 'matplotlib', 'numba'
        ]

        for package in required_packages:
            try:
                __import__(package)
                dependencies[package] = True
            except ImportError:
                dependencies[package] = False

        # 可选包依赖
        optional_packages = ['torch', 'tensorflow', 'cvxpy', 'ortools']

        for package in optional_packages:
            try:
                __import__(package)
                dependencies[f"{package}_optional"] = True
            except ImportError:
                dependencies[f"{package}_optional"] = False

        return dependencies

    def get_recommended_settings(self, experiment_size: str) -> Dict[str, Any]:
        """获取推荐的系统设置"""
        settings = {
            'test': {
                'max_memory_gb': min(2.0, self.resource_limits.max_memory_gb),
                'max_cpu_cores': min(2, self.resource_limits.max_cpu_cores),
                'enable_gpu': False,
                'parallel_jobs': 1
            },
            'small': {
                'max_memory_gb': min(4.0, self.resource_limits.max_memory_gb),
                'max_cpu_cores': min(4, self.resource_limits.max_cpu_cores),
                'enable_gpu': self.system_info.gpu_available,
                'parallel_jobs': 2
            },
            'medium': {
                'max_memory_gb': min(8.0, self.resource_limits.max_memory_gb),
                'max_cpu_cores': min(8, self.resource_limits.max_cpu_cores),
                'enable_gpu': self.system_info.gpu_available,
                'parallel_jobs': 4
            },
            'large': {
                'max_memory_gb': self.resource_limits.max_memory_gb,
                'max_cpu_cores': self.resource_limits.max_cpu_cores,
                'enable_gpu': self.system_info.gpu_available,
                'parallel_jobs': 8
            }
        }

        return settings.get(experiment_size, settings['small'])

    def create_conda_environment_script(self, env_name: str,
                                        requirements_file: Path) -> Path:
        """创建conda环境安装脚本"""
        script_ext = '.bat' if self.is_windows else '.sh'
        script_name = f"setup_environment{script_ext}"

        if self.is_windows:
            script_content = f"""@echo off
echo 创建conda环境: {env_name}
conda create -n {env_name} python=3.8 -y
conda activate {env_name}
pip install -r "{requirements_file}"
echo 环境创建完成！
pause
"""
        else:
            script_content = f"""#!/bin/bash
echo "创建conda环境: {env_name}"
conda create -n {env_name} python=3.8 -y
conda activate {env_name}
pip install -r "{requirements_file}"
echo "环境创建完成！"
"""

        script_path = Path(script_name)
        script_path.write_text(script_content, encoding='utf-8')

        if not self.is_windows:
            os.chmod(script_path, 0o755)  # 给予执行权限

        return script_path

    def __str__(self) -> str:
        """字符串表示"""
        return (f"SystemConfig({self.platform}, "
                f"CPU:{self.system_info.cpu_count}, "
                f"RAM:{self.system_info.memory_total_gb:.1f}GB, "
                f"GPU:{self.system_info.gpu_available})")


# 系统检查工具
class SystemChecker:
    """系统检查工具"""

    @staticmethod
    def check_minimum_requirements() -> Dict[str, tuple[bool, str]]:
        """检查最低系统要求"""
        checks = {}

        # Python版本检查
        python_version = sys.version_info
        python_ok = python_version >= (3, 8)
        checks['python_version'] = (
            python_ok,
            f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        )

        # 内存检查
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        memory_ok = memory_gb >= 4.0
        checks['memory'] = (memory_ok, f"{memory_gb:.1f}GB")

        # 磁盘空间检查
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        disk_ok = disk_free_gb >= 10.0
        checks['disk_space'] = (disk_ok, f"{disk_free_gb:.1f}GB free")

        # CPU检查
        cpu_count = mp.cpu_count()
        cpu_ok = cpu_count >= 2
        checks['cpu_cores'] = (cpu_ok, f"{cpu_count} cores")

        return checks

    @staticmethod
    def generate_system_report(config: SystemConfig) -> str:
        """生成系统报告"""
        info = config.system_info
        limits = config.resource_limits

        report = f"""
=== 系统配置报告 ===
平台: {info.platform} ({info.platform_version})
Python: {info.python_version.split()[0]}
CPU: {info.cpu_count} 核心
内存: {info.memory_total_gb:.1f}GB 总计, {info.memory_available_gb:.1f}GB 可用
磁盘: {info.disk_free_gb:.1f}GB 可用空间
GPU: {'是' if info.gpu_available else '否'}{f' ({info.gpu_info})' if info.gpu_info else ''}

=== 资源限制 ===
最大内存: {limits.max_memory_gb:.1f}GB
最大CPU核心: {limits.max_cpu_cores}
最大磁盘使用: {limits.max_disk_usage_gb:.1f}GB
GPU启用: {'是' if limits.enable_gpu else '否'}
"""

        return report