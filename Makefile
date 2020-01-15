SHELL=/bin/bash
PYTHON=pyenv/bin/python

# global variable for directories
LOG=log
GRID_N={5,10,15,20,30}
UNARY_STD={0.1,1.0}

EXP=infer_grid
RESULTS=${EXP}_score

#experiment argument
DEVICE=cuda:0
EXP_ITERS=20
SLEEP=10
init:
	mkdir -p ${LOG}

ising_infer: $(shell echo ${LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}.txt)


${LOG}/${RESULTS}%.txt: init
	${PYTHON} infer_ising_marginals.py --sleep ${SLEEP} --device ${DEVICE} --exp_iters ${EXP_ITERS} --task $@ > $@	
