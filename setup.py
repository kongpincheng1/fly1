from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fly = control.latest_single_drop:main',
            'pub = control.sim_pub:main',
            'mul = control.test_mulitiDrop:main',
            'mul1 = control.test_mulitiDrop1:main',
            'servo = control.servo_test:main',
            'mission_mul = control.mission_mul:main',
        ],
    },
)
