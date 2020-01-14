#!/bin/sh
python revise_ising_marginals.py --gpu 0 --n 5 --exp_iters 10 > results5n.txt &

python revise_ising_marginals.py --gpu 0 --n 10 --exp_iters 10 > results10n.txt &

python revise_ising_marginals.py --gpu 0 --n 15 --exp_iters 10 > results15n.txt &

wait
