# Code for our paper:
[Region-based Energy Neural Network for Approximate Inference](https://arxiv.org/abs/2006.09927)

# Environment Setup
Create a virtual environment with a python2 interpreter at 'path/to/your/evn/'
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
To compare marginals for on grid graphs:
```
make ising_infer
```

To compare marginals for on complete graphs:
```
make -f completeMakefile ising_infer
```

To compare train a MRF of a grid graph:

``` bash
make ising_train

```

To compare train MRF of a complete graph:

``` bash
make -f completeMakefile ising_train

```
