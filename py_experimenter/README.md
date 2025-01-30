[![Project Homepage](https://img.shields.io/badge/Project%20Homepage-###.github.io/py_experimenter-0092CD)](https://###.github.io/py_experimenter)
[![Pypi](https://img.shields.io/pypi/v/py_experimenter)](https://pypi.org/project/py-experimenter/)
[![License](https://img.shields.io/github/license/###/py_experimenter)](https://###.github.io/py_experimenter/license.html)
[![DOI](###/status.svg)](https://doi.org/###)

![Tests](https://github.com/###/py_experimenter/actions/workflows/tests.yml/badge.svg)
![GitHub Pages](https://github.com/###/py_experimenter/actions/workflows/github-pages.yml/badge.svg)

<img src="docs/source/_static/py-experimenter-logo.png" alt="PyExperimenter Logo: Python biting a database" width="200px"/>

# PyExperimenter

`PyExperimenter` is a tool to facilitate the setup, documentation, execution, and subsequent evaluation of results from an empirical study of algorithms and in particular is designed to reduce the involved manual effort significantly.
It is intended to be used by researchers in the field of artificial intelligence, but is not limited to those.

The empirical analysis of algorithms is often accompanied by the execution of algorithms for different inputs and variants of the algorithms (specified via parameters) and the measurement of non-functional properties.
Since the individual evaluations are usually independent, the evaluation can be performed in a distributed manner on an HPC system.
However, setting up, documenting, and evaluating the results of such a study is often file-based.
Usually, this requires extensive manual work to create configuration files for the inputs or to read and aggregate measured results from a report file.
In addition, monitoring and restarting individual executions is tedious and time-consuming.

These challenges are addressed by `PyExperimenter` by means of a single well defined configuration file and a central database for managing massively parallel evaluations, as well as collecting and aggregating their results.
Thereby, `PyExperimenter` alleviates the aforementioned overhead and allows experiment executions to be defined and monitored with ease.

![General schema of `PyExperimenter`.](docs/source/_static/workflow.png)

For more details check out the [`PyExperimenter` documentation](https://###.github.io/py_experimenter/):

- [Installation](https://###.github.io/py_experimenter/installation.html)
- [Examples](https://###.github.io/py_experimenter/examples/example_general_usage.html)

## Cite PyExperimenter

If you use `PyExperimenter` in a scientific publication, we would appreciate a citation in one of the following ways.

### Citation String

### et al., (2023). PyExperimenter: Easily distribute experiments and track results. Journal of Open Source Software, 8(84), 5149, https://doi.org/###

### BibTex
```
@article{###2023, 
    title = {{PyExperimenter}: Easily distribute experiments and track results}, 
    author = {### ### and ### ### and ### ### and ### ### and ### ### and ### ### and ### ### and ### ###}, 
    journal = {Journal of Open Source Software},
    publisher = {The Open Journal},  
    year = {2023}, 
    volume = {8}, 
    number = {84}, 
    pages = {5149}, 
    doi = {###}, 
    url = {https://doi.org/###}
}
```
