import setuptools

setuptools.setup(
    name='galcem',
    version="0.0.2b",
    author="Eda Gjergo, Aleksei Sorokin",
    author_email="eda.gjergo@gmail.com",
    license='GNU General Public License',
    description="Galactic Chemical Evolution",
    #long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/egjergo/GalCEM",
    #download_url='',
    packages=['galcem','galcem.classes'],
    install_requires=[ # looser? 
        'numpy >= 1.17.0', 
        'scipy >= 1.0.0', 
        'pandas >= 1.0.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"],
    keywords=['galactic', 'chemical', 'evolution', 'simulation'],
    python_requires=">=3.6", # looser? 
    include_package_data=True,
    )