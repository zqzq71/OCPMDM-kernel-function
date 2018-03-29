# OCPMDM-kernel-function
This repository provided kernel function for OCPMDM (for "OCPMDM: Online Computation Platform for Materials Data Ming"). If you want to use the full version of OCPMDM, please use http://materialdata.shu.edu.cn.

## Provided function
* Filling perovskite material descriptors.
* Relevance vector machine.
* Virtual Screening perovskite material candidates.

## Introduction
In demo, first, the perovskite materials will convert to descriptors, then RVM have been used to build the machine learning model. Using virtual screening to find a material with higher property than those in training set.

## How to use
Install requirements

```shell
pip install -r requirements.txt
```

Run demo

```shell
python demo.py
```

