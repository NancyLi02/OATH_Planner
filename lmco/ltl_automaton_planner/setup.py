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
        # (os.path.join('lib', package_name, 'ltl_tools'), [
        #     os.path.join(package_name, 'ltl_tools', f) for f in os.listdir(os.path.join(package_name, 'ltl_tools')) if f.endswith('.py')
        # ])
        # (os.path.join('share', package_name, 'nodes'), glob('nodes/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jren313',
    maintainer_email='jren313@gatech.edu',
    description='TODO: Package description',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planner_node = ltl_automaton_planner.planner_node:main',
            'benchmark_node = ltl_automaton_planner.benchmark_node:main',
            'relay_node = ltl_automaton_planner.relay_node:main'
        ],
    },
)
