from setuptools import setup, find_packages

setup(
        name='offlinerlkit',
        version="0.0.1",
        description=(
            'OfflineRL-kit'
        ),
        author='Yihao Sun',
        author_email='sunyh@lamda.nju.edu.cn',
        maintainer='yihaosun1124',
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            "gym>=0.15.4,<=0.24.1",
            "mujoco==2.3.3",
            "cython<3",
            "matplotlib",
            "h5py",
            "opencv2-python",
            "numpy<1.24",
            "pandas",
            "scikit-learn",
            #"ray==1.13.0",
            "torch",
            "tensorboard",
            "tqdm",
            "wandb"
        ]
    )
