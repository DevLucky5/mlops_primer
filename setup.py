from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'
def get_requirements(file:str) -> List[str]:
    '''
    Return the list of requirements.
    '''

    with open(file) as f:
        requirements = [req.strip() for req in f.readlines()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup (
    name = 'mlops_primer',
    version = '0',
    author = 'Debasish',
    author_email='luckypadhy011@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)