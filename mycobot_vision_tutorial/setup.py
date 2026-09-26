from setuptools import find_packages, setup
from glob import glob

package_name = 'mycobot_vision_tutorial'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gz',
    maintainer_email='gz@todo.todo',
    description='Stage-1 vision warm-up: color segmentation + 3D centroid for mycobot Gazebo sim.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'color_detector = mycobot_vision_tutorial.color_detector:main',
        ],
    },
)
