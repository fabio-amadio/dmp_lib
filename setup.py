from setuptools import setup

package_name = 'dmp_lib'

setup(
    name=package_name,
    version='1.0',
    packages=[package_name],
    data_files=[],
    install_requires=['numpy', 'matplotlib', 'scikit-learn'],
    zip_safe=True,
    maintainer='fabio-amadio',
    maintainer_email='fabioamadio93@gmail.com',
    description='Dynamic Movement Primitive (DMP) Library',
    license='GNU GENERAL PUBLIC LICENSE v3',
    entry_points={},
)
