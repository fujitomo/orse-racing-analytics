from setuptools import setup, find_packages

setup(
    name="horse_racing",
    version="0.1.0",
    description="競馬のレースデータを分析し、競走経験質指数（REQI）や成績の傾向を可視化するツール",
    author="Tomohiro Fujino",
    author_email="",
    url="https://github.com/yourusername/horse-racing-analysis",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'analyze-race=horse_racing.cli.analyze_race:main',
            'process-bac=horse_racing.cli.process_bac:main',
            'process-sed=horse_racing.cli.process_sed:main',
        ],
    },
) 