# Hierarchical Bayesian Inference Tool for Knowledge DAGs


## Description

Hierarchical Bayesian Inference Tool for Knowledge DAGs is a tool for inferring individuals' causal scientific knowledge from the individuals' responses to a set of questions. This tool is based on a probabilistic [graph-based logistic (GrL) model](cite_paper) that uses directed acyclic graphs (DAGs) to represent individuals' causal scientific knowledge for a given theory. This tool can be used for:
+ Visualizing population level knowledge distribution
+ Visualizing individual level knowledge distribution
+ Visualizing clusters across the population

## Running the notebook on your personal computer

Find and download the right version of
[Anaconda for Python 3.7](https://www.anaconda.com/distribution) from Continuum Analytics.
**Note:** You do need Python 3 and note Python 2. The notebooks will not work
with Python 2.

### OS Specific Instructions

#### Microsoft Windows

+ We need C, C++, Fortran compilers, as well as the Python sources.
Start the command line by opening "Anaconda Prompt" from the
start menu. In the command line type:
```
conda config --append channels https://repo.continuum.io/pkgs/free
conda install mingw libpython
```
+ Finally, you need [git](https://git-scm.com/downloads). As you install it,
make sure to indicate that you want to use "Git from the command line and
also from 3rd party software".

#### Apple OS X

+ Download and install the latest version of [Xcode](https://developer.apple.com/xcode/download/).

#### Linux

If you are using Linux, I am sure that you can figure it out on your own.

### Installation of Required Python Packages

Independently of the operating system, use the command line to install the following Python packages:
+ [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/), for beautiful graphics:
```
conda install seaborn
```

+ [PyMC3](https://docs.pymc.io/) for MCMC sampling:
```
conda install pymc3
```

+ [graphviz](https://www.graphviz.org/download/) for visualizing probabilistic graphical models:
```
pip install graphviz
```

+ [scikit-learn](https://scikit-learn.org/stable/) for some standard machine learning algorithms implemented in Python:
```
conda install scikit-learn
```

+ [dill](https://pypi.org/project/dill/) for serializing and de-serializing python objects:
```
pip install dill
```

+ [pandas](https://pypi.org/project/pandas/) for fast, flexible, and expressive data structures:
```
pip install pandas
```

+ [plotly](https://plotly.com/python/) for visualizing interactive graphs:
```
pip install plotly
```

+ [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_guide.html) for GUI:
```
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Running the notebooks

+ Open the command line.
+ `cd` to your favorite folder.
+ Then, type:
```
git clone https://github.com/PredictiveScienceLab/knowledge-dag.git
```
+ This will download the contents of this repository in a folder called `knowledge-dag`.
+ Enter the ``knowledge-dag`` folder:
```
cd knowledge-dag
```
+ Start the jupyter notebook by typing the command:
```
jupyter notebook
```
+ Read the user guide (`user_guide.ipynb`) in the folder called `user_guide`.
+ If the tool content has been updated, type the following command (while being inside `knowledge-dag`) to get the latest version:
```
git pull origin master
```
Keep in mind, that if you have made local changes to the repository, you may have to commit them before moving on.