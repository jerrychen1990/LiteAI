# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     setup
   Description :
   Author :       chenhao
   date：          2021/4/6
-------------------------------------------------
   Change Activity:
                   2021/4/6:
-------------------------------------------------
"""
import sys

from setuptools import find_packages, setup

from snippets.utils import get_latest_version, get_next_version, read2list


def get_install_req():
    req = read2list("requirements.txt")
    return req


if __name__ == "__main__":
    name = "liteai"
    if len(sys.argv) >= 4 and sys.argv[-1].startswith("v"):
        version = sys.argv.pop(-1)
    else:
        latest_version = get_latest_version(name)
        version = get_next_version(latest_version)
    print(f"version: {version}")
    install_req = get_install_req()
    print(f"install_req: {install_req}")
    setup(
        name=name,
        version=version,
        install_requires=install_req,
        packages=find_packages(exclude=['tests*']),
        package_dir={"": "."},
        package_data={},
        url='https://github.com/jerrychen1990/LiteAI',
        license='MIT',
        author='Chen Hao',
        author_email='jerrychen1990@gmail.com',
        zip_safe=True,
        description='use ai lite',
        long_description="use ai lite"
    )
