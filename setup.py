from setuptools import setup, find_packages

setup(
    name='webookcare',
    version="0.1.0",
    author='Arian Yavari',
    author_email='arianyavari@protonmail.com',
    description='a search and match engine conneccting HCW and patients',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/arian47/webookcare',
    packages=find_packages(include=['webookcare', 'webookcare.*']),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy==1.23.5",
        'pandas==2.2.2',
        'pydantic==1.10.9',
        'tensorflow==2.12.0',
        'python-dotenv==1.0.1',
        'tensorflow_docs',
        'scikit-learn==1.1.3',
        'docx2pdf==0.1.8',
        'scikit-surprise==1.1.4',
        'fastapi==0.110.1',
        'Faker==25.3.0',
        'mysql-connector-python==9.0.0',
    ]
)