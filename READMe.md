# Orthogonal LDA
Implements linear discriminant analysis (LDA) using functionality from the Manopt package.

![CI](https://github.com/grero/OrthogonalLDA.jl/actions/workflows/ci.yml/badge.svg)

## Introduction

Basically, LDA searches for transformation matrix $W$ that maximizes the projection of the between-classs scatter matrix $S_b$, while minimizing the within-class scatter matrix $S_w$, in other words maximises that ratio

$\frac{Tr\[W^TS_bW^t\]}{Tr\[W^TS_wW\]}$.

Here, we directly maximise this ratio while ensuring that transformation matrix $W$ is orthonormal.

