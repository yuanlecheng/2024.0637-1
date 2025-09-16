
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import platform


# 读取版本信息
def read_version():
    version_file = Path(__file__).parent / 'src' / '__init__.py'
    if version_file.exists():
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'


# 读取README
def read_readme():
    readme_file = Path(__file__).parent / 'README.md'
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "ML-CO-pipeline-AMoD-control: 自动出行需求系统的机器学习和组合优化流水线"


# 读取依赖
def read_requirements():
    req_file = Path(__file__).parent / 'requirements.txt'
    if req_file.exists():
        with open(req_file, 'r') as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # 处理条件依赖
                    if ';' in line:
                        req, condition = line.split(';', 1)
                        requirements.append(f"{req.strip()}; {condition.strip()}")
                    else:
                        requirements.append(line)
            return requirements
    return []


# 平台特定设置
def get_platform_specific_settings():
    """获取平台特定的编译设置"""
    settings = {
        'include_dirs': [],
        'library_dirs': [],
        'libraries': [],
        'extra_compile_args': [],
        'extra_link_args': []
    }

    if platform.system() == 'Windows':
        # Windows特定设置
        settings['extra_compile_args'].extend([
            '/O2',  # 优化
            '/W3',  # 警告级别
            '/std:c++14'  # C++标准
        ])

        # 如果安装了Visual Studio
        if 'VCINSTALLDIR' in os.environ:
            settings['include_dirs'].append(os.path.join(os.environ['VCINSTALLDIR'], 'include'))

    elif platform.system() == 'Linux':
        # Linux设置
        settings['extra_compile_args'].extend([
            '-O3',
            '-std=c++14',
            '-fPIC'
        ])

    elif platform.system() == 'Darwin':
        # macOS设置
        settings['extra_compile_args'].extend([
            '-O3',
            '-std=c++14',
            '-stdlib=libc++'
        ])
        settings['extra_link_args'].append('-stdlib=libc++')

    return settings


# 可选的C++扩展（如果用户想要性能优化）
def get_extensions():
    """获取C++扩展模块（可选）"""
    extensions = []

    # 检查是否可以编译C++扩展
    try:
        from pybind11 import get_cmake_dir, get_pybind_include
        import pybind11

        platform_settings = get_platform_specific_settings()

        # k-dSPP算法的C++加速扩展（可选）
        kdsp_ext = Extension(
            'amod_pipeline.core.kdsp_cpp',
            sources=[
                'src/core/cpp/kdsp_solver.cpp',
                'src/core/cpp/graph_algorithms.cpp',
                'src/core/cpp/python_bindings.cpp'
            ],
            include_dirs=[
                             get_pybind_include(),
                             'src/core/cpp/include'
                         ] + platform_settings['include_dirs'],
            library_dirs=platform_settings['library_dirs'],
            libraries=platform_settings['libraries'],
            extra_compile_args=platform_settings['extra_compile_args'],
            extra_link_args=platform_settings['extra_link_args'],
            language='c++'
        )
        extensions.append(kdsp_ext)

        print("找到pybind11，将编译C++扩展以提升性能")

    except ImportError:
        print("未找到pybind11，跳过C++扩展编译（将使用纯Python实现）")

    return extensions


# 自定义构建命令
class CustomBuildExt(build_ext):
    """自定义构建扩展命令"""

    def build_extensions(self):
        # 检查编译器可用性
        try:
            super().build_extensions()
        except Exception as e:
            print(f"警告: C++扩展编译失败，将使用纯Python实现: {str(e)}")
            # 移除失败的扩展
            self.extensions = []


# 入口点脚本
entry_points = {
    'console_scripts': [
        'amod-pipeline=amod_pipeline.master_script:main',
        'amod-create-instances=amod_pipeline.scripts.create_training_instances:main',
        'amod-train=amod_pipeline.scripts.run_training:main',
        'amod-evaluate=amod_pipeline.scripts.run_benchmarks:main',
        'amod-visualize=amod_pipeline.visualization.visualization_results:main'
    ]
}

# 包数据
package_data = {
    'amod_pipeline': [
        'data/taxi_data_format.txt',
        'config/*.yaml',
        'config/*.json',
        'visualization/templates/*.html',
        'visualization/static/css/*.css',
        'visualization/static/js/*.js'
    ]
}

# 额外数据文件
data_files = [
    ('config', ['config/default_experiment.yaml']),
    ('scripts', ['scripts/setup_environment.bat', 'scripts/setup_environment.sh']),
    ('docs', ['README.md', 'LICENSE'])
]

# 分类器
classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: C++',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Software Development :: Libraries :: Python Modules'
]

# 主要设置
setup(
    name='amod-pipeline',
    version=read_version(),
    description='机器学习和组合优化流水线用于自动出行需求系统控制',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='ML-CO Pipeline Team',
    author_email='contact@amod-pipeline.org',
    url='https://github.com/tumBAIS/ML-CO-pipeline-AMoD-control',
    project_urls={
        'Bug Reports': 'https://github.com/tumBAIS/ML-CO-pipeline-AMoD-control/issues',
        'Source': 'https://github.com/tumBAIS/ML-CO-pipeline-AMoD-control',
        'Documentation': 'https://amod-pipeline.readthedocs.io'
    },

    # 包配置
    packages=find_packages(exclude=['tests*', 'docs*']),
    package_dir={'amod_pipeline': 'src'},
    package_data=package_data,
    data_files=data_files,
    include_package_data=True,

    # 依赖
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.6.0',
            'flake8>=3.9.0',
            'isort>=5.9.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0'
        ],
        'gpu': [
            'torch>=1.9.0',
            'tensorflow-gpu>=2.6.0'
        ],
        'optimization': [
            'cvxpy>=1.2.0',
            'ortools>=9.0.0',
            'gurobipy>=9.5.0'
        ],
        'cpp': [
            'pybind11>=2.6.0',
            'cmake>=3.16.0'
        ],
        'all': [
            'pytest>=6.2.0', 'pytest-cov>=2.12.0', 'black>=21.6.0',
            'torch>=1.9.0', 'cvxpy>=1.2.0', 'pybind11>=2.6.0'
        ]
    },

    # Python版本要求
    python_requires='>=3.8',

    # 扩展模块
    ext_modules=get_extensions(),
    cmdclass={'build_ext': CustomBuildExt},

    # 元数据
    classifiers=classifiers,
    keywords='machine-learning optimization autonomous-vehicles mobility transportation',
    license='MIT',

    # 入口点
    entry_points=entry_points,

    # 测试配置
    test_suite='tests',
    tests_require=['pytest>=6.2.0', 'pytest-cov>=2.12.0'],

    # 其他配置
    zip_safe=False,
    platforms=['Windows', 'Linux', 'MacOS'])


# 安装后处理
def post_install_message():
    """安装后信息提示"""
    print("\n" + "=" * 60)
    print("AMoD Pipeline 安装完成！")
    print("=" * 60)
    print("\n开始使用:")
    print("1. 创建训练实例: amod-create-instances --experiment-type test")
    print("2. 训练模型: amod-train --experiment-type test")
    print("3. 运行评估: amod-evaluate --experiment-type test")
    print("4. 完整流水线: amod-pipeline --experiment-type test --step full")
    print("\n配置文件位置: ~/.amod_pipeline/")
    print("日志文件位置: ./logs/")
    print("结果文件位置: ./results/")
    print("\n详细文档: https://amod-pipeline.readthedocs.io")
    print("问题报告: https://github.com/tumBAIS/ML-CO-pipeline-AMoD-control/issues")

    # 检查Windows特定问题
    if platform.system() == 'Windows':
        print("\nWindows用户注意事项:")
        print("- 确保已安装Visual Studio Build Tools或完整的Visual Studio")
        print("- 如需GPU支持，请安装CUDA toolkit")
        print("- 建议使用Anaconda或Miniconda管理Python环境")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # 运行设置
    setup()

    # 如果是直接安装，显示后续信息
    if len(sys.argv) > 1 and 'install' in sys.argv:
        post_install_message()