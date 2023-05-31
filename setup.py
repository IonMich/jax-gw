from setuptools import setup

def get_requirements():
    with open("requirements-dev.txt", "r") as ff:
        requirements = ff.readlines()
    return requirements


VERSION = "0.0.1"

setup(name='jax-gw', 
      version=VERSION, 
      author='Ioannis Michaloliakos',
      author_email='ioannis.michalol@gmail.com',
      python_requires='==3.9.*',
      install_requires=get_requirements(),    
      packages=['detector','signal','sources'],
      package_dir={'': 'jax_gw'})