# Greedy Poisson Rejection Sampling

![Gaussian-Gaussian GPRS](./img/gprs.gif)

This repository contains experimental data and reference implementations of the relative entropy coding algorithms that were used in the paper

[Greedy Poisson Rejection Sampling](https://arxiv.org/abs/2305.15313)

by Gergely Flamich.

# Introduction

The paper proposes the Greedy Poisson Rejection Sampler (GPRS), a rejection sampling algorithm based on Poisson Processes.
The main goal of GPRS is to be utilized in a channel simulation protocol for encoding random samples from a target distribution using as few bits as necessary. 
Concretely, given a target distribution $Q$ and a proposal / coding distribution $P$, GPRS encodes a **single** random sample $x \sim Q$ using 
$$
  D_{KL}[Q\,\Vert\,P] + \log_2 (D_{KL}[Q\,\Vert\,P] + 1) + \mathcal{O}(1) 
$$
bits, where $D_{KL}[Q\,\Vert\,P]$ denotes the Kullback-Leibler divergence of $Q$ from $P$ measured in bits.

The paper proposes three variants of GPRS:
 - **Global GPRS:** The most general variant, as it is applicable to distributions $Q$ and $P$ over arbitrary Polish spaces, whenever the target distribution $Q$ is abosolutely continuous with respect to the coding distribution $P$.
 The minimum requirements for implementation are:
   - We can simulate $P$-distributed samples.
   - We can evaluate the density ratio (Radon-Nikodym derivative) $r = dQ/dP$.
   - We can evaluate the complementary CDFs $w_P(h) = \mathbb{P}_{Z \sim P}[r(Z) \geq h]$ and $w_Q(h) = \mathbb{P}_{Z \sim Q}[r(Z) \geq h]$.
   
   The number of steps $N$ this variant of GPRS takes on average is
   $$
   \mathbb{E}[N] = \Vert dQ/dP \Vert_\infty = 2^{D_\infty[Q \, \Vert \, P]},
   $$ 
   where $\Vert \cdot \Vert_\infty$ denotes the ($P$-essential) infinity norm and $D_\infty[Q \, \Vert \, P]$ denotes the Renyi $\infty$-divergence of $Q$ from $P$ in bits. 

   Pseudo-code for global GPRS is given in Algorithm 3 in the paper.
 - **Parallelized GPRS:**
 - **Branch-and-bound GPRS:**

The repository provides JAX implementation of all of the above mentioned variants.

Please cit the paper if you find it useful in your research:

```bibtex
@inproceedings{flamich2023gprs,
  title = {Greedy Poisson Rejection Sampling},
  author = {Flamich, Gergely},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2023},
}
```

# Requirements

# Structure and Contents of the Repository

The repository contains 
