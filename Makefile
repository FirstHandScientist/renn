SHELL=/bin/bash
PYTHON=pyenv/bin/python

# global variable for directories
LOG=log
GRID_N={5,10,15,20}
UNARY_STD={0.1,1.0}
RESULTS=result

init:
	mkdir -p ${LOG}

ising: init
	for std in $(shell echo ${UNARY_STD}); do \
		for n in $(shell echo ${GRID_N}); do echo "${PYTHON} revise_ising_marginals.py --gpu 0 --n $$n --exp_iters 2 --unary_std $$std > ${LOG}/${RESULTS}_$${n}_$${std}.txt"; done \
		done
