# Simple Steppest descent and gradient problem

It requires `python >= 3.9` and `numpy`.

Execute using:
```shell
python3 gradients.py
```

The script implements simple Steppest descent algorithm as well as the
conjugate gradient.

Two approaches are proposed:
* Steppest descent as first method to solve the poisson problem. The algorithm is based on a Taylor expansion of second order.
It's main drawback is on the number of iterations.
* Conjugate Gradient as a second attempt, which forced descent by using an Arnoldi procedure, which guaratees minimal amount of descent diretions.
Relative convergence is guarateed.

TODOS:
- [ ] Benchmark against analytical solution
- [ ] Verbose readme
- [ ] Test case with multiple sine, cosine functions
- [ ] Mesh convergence analysis
