SHELL=/bin/bash
PYTHON=pyenv/bin/python

# global variable for directories
LOG=log
TRAIN_LOG_MSG=tlog_msg
TRAIN_LOG_NET=tlog_net
GRID_N={5,10}
UNARY_STD={0.1,1.0}
PENALTY={1,5}
# STRUCTURE should be: grid | full_connected
STRUCTURE=grid
EXP=train_${STRUCTURE}
RESULTS=${EXP}_score

#experiment argument
DEVICE=cpu

EXP_ITERS=20
SLEEP=10

ALGORITHM_MSG={ve,mf,lbp,dbp,gbp}
ALGORITHM_NET={bethe,kikuchi}

# data set
DATA_DIR=data
TRAIN_SIZE=1000
TEST_SIZE=1000
BATCH_SIZE=100

init:
	mkdir -p ${LOG} ${TRAIN_LOG_MSG} ${TRAIN_LOG_NET} ${DATA_DIR}

ising_infer: $(shell echo ${LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen${PENALTY}.txt)


${LOG}/${RESULTS}%.txt: init
	${PYTHON} infer_ising_marginals.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --exp_iters ${EXP_ITERS} --task $@ > $@	


# training with inference method

ising_train: $(shell echo ${TRAIN_LOG_MSG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen0_algo.${ALGORITHM_MSG}.txt) $(shell echo ${TRAIN_LOG_NET}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen${PENALTY}_algo${ALGORITHM_NET}.txt)

# training with massage passing algorithms
${TRAIN_LOG_MSG}/${RESULTS}%.txt: init
	${PYTHON} train_ising.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --train_size ${TRAIN_SIZE} --test_size ${TEST_SIZE} --batch_size ${BATCH_SIZE} --task $@ > $@

# training with renn or bethe inference net
${TRAIN_LOG_NET}/${RESULTS}%.txt: init
	${PYTHON} train_ising.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --train_size ${TRAIN_SIZE} --test_size ${TEST_SIZE} --batch_size ${BATCH_SIZE} --task $@ > $@

