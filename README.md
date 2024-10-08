# Neural network solvers for parametrized elasticity problems that conserve linear and angular momentum
### Wietse M. Boon, Nicola R. Franco and Alessio Fumagalli

The [examples](./examples/) folder contains the source code for replicating the three test cases. See [arXiv pre-print](XXX).

# Abstract
We consider a mixed formulation of parametrized elasticity problems in terms of stress, displacement, and rotation. The latter two variables act as Lagrange multipliers to enforce conservation of linear and angular momentum. Due to the saddle-point structure, the resulting system is computationally demanding to solve directly, and we therefore propose an efficient solution strategy based on a decomposition of the stress variable. First, a triangular system is solved to obtain a stress field that balances the body and boundary forces. Second, a trained neural network is employed to provide a correction without affecting the conservation equations. The displacement and rotation can be obtained by post-processing, if necessary. The potential of the approach is highlighted by three numerical test cases, including a non-linear model.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv pre-print](XXX).

# PorePy and PyGeoN version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and [PyGeoN](https://github.com/compgeo-mox/pygeon) and might revert them.
Newer versions of may not be compatible with this repository.<br>
PorePy: Release v0.5.0 <br>
PyGeoN valid tag: 31654ffd1c1de609bd138a2b6061051af6236816

# License
See [license](./LICENSE).
