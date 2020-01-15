SHELL=/bin/bash
PYTHON=pyenv/bin/python

# global variable for directories
LOG=log
GRID_N={5,10,15,20,30}
UNARY_STD={0.1,1.0}

# STRUCTURE should be: grid | full_connected
STRUCTURE=full_connected
EXP=infer_${STRUCTURE}
RESULTS=${EXP}_score

#experiment argument
DEVICE=cuda:0
EXP_ITERS=20
SLEEP=10
init:
	mkdir -p ${LOG}

ising_infer: $(shell echo ${LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}.txt)


${LOG}/${RESULTS}%.txt: init
	${PYTHON} infer_ising_marginals.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --exp_iters ${EXP_ITERS} --task $@ > $@	
