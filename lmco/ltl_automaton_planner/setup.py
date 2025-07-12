from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'ltl_automaton_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*_launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'ltl_automaton_planner'), glob('ltl_automaton_planner/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nanli',
    maintainer_email='nan.li@gatech.edu',
    description='TODO: Package description',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = ltl_automaton_planner.nodes.planner_node:main',
            'benchmark_node = ltl_automaton_planner.nodes.benchmark_node:main',
            'relay_node = ltl_automaton_planner.nodes.relay_node:main',
            'taskassign_node = ltl_automaton_planner.nodes.taskassign_node:main',
            'showmove_node = ltl_automaton_planner.nodes.showmove_node:main',
            'planner_cluster_node = ltl_automaton_planner.nodes.planner_cluster_node:main',
            'benchmark_cluster_node = ltl_automaton_planner.nodes.benchmark_cluster_node:main',
            'taskassign_cluster_node = ltl_automaton_planner.nodes.taskassign_cluster_node:main'
        ],
    },
)
