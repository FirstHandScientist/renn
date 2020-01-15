SHELL=/bin/bash
PYTHON=pyenv/bin/python

# global variable for directories
LOG=log
GRID_N=4
UNARY_STD=0.9

EXP=infer_grid
RESULTS=${EXP}_score

#experiment argument
DEVICE=cpu
EXP_ITERS=2

init:
	mkdir -p ${LOG}

ising_infer: $(shell echo ${LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}.txt)


${LOG}/${RESULTS}%.txt: init
	${PYTHON} infer_ising_marginals.py --device ${DEVICE} --exp_iters ${EXP_ITERS} --task $@ > ${LOG}/$@	
