# Learning large DAGs is harder than you Think: Many Losses are Minimal for the wrong DAG
This repository contains all experiments and analysis of the paper *Learning large DAGs is harder than you Think: Many Losses are Minimal for the wrong DAG* published at ICLR 2024.

## Abstract
Structure learning is a crucial task in science, especially in fields such as medicine and biology, where the wrong identification of (in)dependencies among random variables can have significant implications. The primary objective of structure learning is to learn a Directed Acyclic Graph (DAG) that represents the underlying probability distribution of the data. Many prominent DAG learners rely on least square losses or log-likelihood losses for optimization. It is well-known from regression models that least square losses are heavily influenced by the scale of the variables. Recently it has been demonstrated that the scale of data also affects performance of structure learning algorithms, though with a strong focus on linear 2-node systems and simulated data. Moving beyond these results, we provide conditions under which square-based losses are minimal for wrong DAGs in $d$-dimensional cases. Furthermore, we also show that scale can impair performance of structure learners if relations among variables are non-linear for both square based and log-likelihood based losses. We confirm our theoretical findings through extensive experiments on synthetic and real-world data.

## Notebooks and Files

### manipulations
In this notebook we investigate the perfect manipulation scenario described in our paper for the multiple settings. We consider linear and non-linear 3-node systems, linear and non-linear 10-node systems as well as a real-worl example. We show that DAG learners using Least Square losses and related ones do predict different graphs depending on the variable's scale. Further we show that it is perfectly controllable what is predicted under some circumstances. We employ NOTEARS and DAG-GNN as instantiations of DAG learners using Least Square based (or related) losses.

### varsort_mmse-3nodes
In this notebook we further investigate properties of _varsortability_ by "simulating" NOTEARS with a set of linear models defined for each possible DAG. We empirically analyze the dependency between varsortability and MMSE.

### flip-experiment-imperfect-scenario and flip-experiment-fixed-data-imperfect-scenario
These two files implement the same experiment. We have given a chain graph $G$ as well as data from a distirbution $p$ that factorizes according to $G$, we then try to manipulate the variable's scale s.t. NT/DG reverses $G$ in an imperfect regime (in this case we only have access to one variables). The file _flip-experiment-fixed-data-imperfect-scenario_ does this using fixed noise sampled only once so that each manipulation is performed with the exact same data. The file _flip-experiment-imperfect-scenario_ samples new data for each run.

### flip-experiment-analysis-imperfect-scenario
Here we analyze the results obtained by the experiments in the imperfect setting.

### dag\_generator
This file contains a DAGGenerator class to automatically generate random DAGs.

## Directories
### notears
Contains the NOTEARS implementation provided by [The author's of NOTEARS](https://github.com/xunzheng/notears). 

### dag-gnn
Contains the DAG-GNN implementation provided by [The author's of DAG-GNN](https://github.com/fishmoon1234/DAG-GNN).

### data
Data used to perform our manipulations in case we used fixed datasets. 

### experiment-logs
Detailed logs of our experiments conducted. 

### figures
Figures that show DAGs predicted by NT/DG in different scenarios (scenario should be clear from sub-directory names).

### graphs
Graphs used to generated data in the `data` directory.



