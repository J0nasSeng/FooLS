# Tearing Apart NOTEARS (TANT)
This repository contains all experiments and analysis of the paper [Tearing Apart NOTEARS: Controlling the Graph Prediction via Variance Manipulation](https://openreview.net/forum?id=jgQw9lTAxj-). In the following we give a brief overview of what's contained in this repsoitory.

## attacks
In this notebook we investigate the perfect attack scenario described in our paper for the 3-node case. We show that we are able to perfectly predict NOTEARS' output.

## varsort-3nodes
In this notebook we further investigate properties of _varsortability_ by "simulating" NOTEARS with a set of linear models defined for each possible DAG.

## nt\_linear
Contains the NOTEARS implementation provided by [The author's of NOTEARS](https://github.com/xunzheng/notears). 

## flip-experiment and flip-experiment-fixed-data
These two files implement the same experiment. We have given a causal chain structure as well as data from this structure, we then try to attack NOTEARS in the restricted setting described in our paper to flip the entire chain by just manipulating the variance of one node in the graph. The file _flip-experiment-fixed-data_ does this using fixed noise sampled only once so that each attack is performed with the exact same data. The file _flip-experiment_ samples new data for each attack.

## flip-experiment-analysis
Here we analyze the results obtained by the experiments in the restricted setting.

## dag\_generator
This file contains a DAGGenerator class to automatically generate random DAGs.
