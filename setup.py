import os
from glob import glob
from setuptools import setup

package_name = 's2m2_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Mederic Fourmy',
    author_email='mederic.fourmy@gmail.com',
    description='ROS2 wrapper around s2m2 deep stereo depth estimation.',
    license='CC BY-NC 4.0',
    entry_points={
        'console_scripts': [
            's2m2_node = s2m2_ros.s2m2_node:main'
        ],
    },
)