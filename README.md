# Fooling Least Square Based DAG Learners Using Variance Manipulation (FooLS) 
This repository contains all experiments and analysis of the paper *Fooling Least Square Based DAG Learners Using Variance Manipulation*. In the following we give a brief overview of what's contained in this repsoitory.

## manipulations
In this notebook we investigate the perfect manipulation scenario described in our paper for the multiple settings. We consider linear and non-linear 3-node systems, linear and non-linear 10-node systems as well as a real-worl example. We show that DAG learners using Least Square losses and related ones do predict different graphs depending on the variable's scale. Further we show that it is perfectly controllable what is predicted under some circumstances. We employ NOTEARS and DAG-GNN as instantiations of DAG learners using Least Square based (or related) losses.

## varsort_mmse-3nodes
In this notebook we further investigate properties of _varsortability_ by "simulating" NOTEARS with a set of linear models defined for each possible DAG. We empirically analyze the dependency between varsortability and MMSE.

## nt\_linear
Contains the NOTEARS implementation provided by [The author's of NOTEARS](https://github.com/xunzheng/notears). 

## flip-experiment-imperfect-scenario and flip-experiment-fixed-data-imperfect-scenario
These two files implement the same experiment. We have given a chain graph $G$ as well as data from a distirbution $p$ that factorizes according to $G$, we then try to manipulate the variable's scale s.t. NOTEARS reverses $G$ in an imperfect regime (in this case we only have access to one variables). The file _flip-experiment-fixed-data-imperfect-scenario_ does this using fixed noise sampled only once so that each manipulation is performed with the exact same data. The file _flip-experiment-imperfect-scenario_ samples new data for each run.

## flip-experiment-analysis-imperfect-scenario
Here we analyze the results obtained by the experiments in the imperfect setting.

## dag\_generator
This file contains a DAGGenerator class to automatically generate random DAGs.
