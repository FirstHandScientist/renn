# Environment Setup
Create a virtual environment with a python3 interpreter at 'path/to/your/evn/'
```bash
$ virtualenv -p python3.6 pyenv
```
Then activate your environment:

``` bash
$ source pyenv/bin/activate
```
and install the requirement file:

``` bash
$ pip install -r requirements.txt

```

``` bash
$ python setup.py develop

```
# experiment examples
In order to comparing difference algorithms, run inference tasks on grid graphs:
```
make ising_infer
```
 Run inference tasks on complete graphs:
```
make -f completeMakefile ising_infer
```
The default way of sampling potential factors is from Gaussian. The option can be changed by feeding argument to make:

``` bash
make ising_infer MRF_PARA='{U5,U15}' GRID_N=15 PENALTY='{1,3,5,10}' LOG=uniformGrid15Log -j2
```
which would run inference on grid MRFs of size 15 with uniformly-sampled potentials, i.e., (-5,5), (-15,15). The consistency regularization parameter is enumerated from {1,3,5,10}. The experiment log is directed into the directory 'uniformGrid15Log'. Other experiments may be conducted similarly with the corresponding arguments fed.

To learn a MRF of a grid graph:

``` bash
make ising_train

```

To learn a MRF of a complete graph:

``` bash
make -f completeMakefile ising_train

```
In order to compare RENN performance with different architecture, firstly active only RENN for testing by set args.method = ['kikuchi'] in infer_ising_marginals.py, then use the Makefile to run tasks, e.g., 

``` bash
make net_arch ARCH='{att0mlp0,att1mlp2,att2mlp2}' PENALTY='{1,3,5,10}' GRID_N=10 UNARY_STD=1 -j2
```
where ARCH is the argument indicating the structure of RENN to perform inference in this set of experiments, with 'att1mlp2' denoting to build RENN with 1 layer of attention structure and 2 layers of residual structures. UNARY_STD denotes the standard deviation of univariate potentials in MRFs.
