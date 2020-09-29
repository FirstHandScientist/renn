SHELL=/bin/bash
PYTHON=pyenv/bin/python

# global variable for directories
LOG=log
ARCH_LOG=${LOG}/arch
TRAIN_LOG_MSG=tlog_msg
TRAIN_LOG_NET=tlog_net
GRID_N={5,10}
UNARY_STD=0.1
PENALTY={3,5,10}
# STRUCTURE should be: grid | full_connected
STRUCTURE=grid
EXP=train_${STRUCTURE}
RESULTS=${EXP}_score

#experiment argument
DEVICE=cuda:0

EXP_ITERS=20

SLEEP=15

ALGORITHM_MSG={ve,mf,lbp,dbp,gbp}
ALGORITHM_NET={bethe,kikuchi}

LR=0.0005
TRAIN_ITERS=50
# data set
DATA_DIR=data
TRAIN_SIZE=4000
TEST_SIZE=1000
BATCH_SIZE=500
# architecture setting
ARCH={att0mlp0,att1mlp2}



test:
	echo ${ARCH_LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen${PENALTY}.txt
	echo ${A}

init:
	mkdir -p ${LOG} ${ARCH_LOG} ${TRAIN_LOG_MSG} ${TRAIN_LOG_NET} ${DATA_DIR}

# inference banchmarks
ising_infer: $(shell echo ${LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen${PENALTY}.txt)


${LOG}/${RESULTS}%.txt: init
	${PYTHON} infer_ising_marginals.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --exp_iters ${EXP_ITERS} --task $@ > $@	

# architecture comparisons
net_arch: $(shell echo ${ARCH_LOG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen${PENALTY}.txt,${ARCH})

${ARCH_LOG}/${RESULTS}%: init
	${PYTHON} infer_ising_marginals.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --exp_iters ${EXP_ITERS} --net $(shell cut -d',' -f2 <<<$@) --task $(shell cut -d',' -f1 <<<$@) > $(shell cut -d',' -f1 <<<$@).$(shell cut -d',' -f2 <<<$@)

# training with inference method

ising_train: $(shell echo ${TRAIN_LOG_NET}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen${PENALTY}_algo.${ALGORITHM_NET}.txt) $(shell echo ${TRAIN_LOG_MSG}/${RESULTS}_n${GRID_N}_std${UNARY_STD}_pen0_algo.${ALGORITHM_MSG}.txt)

# training with massage passing algorithms
${TRAIN_LOG_MSG}/${RESULTS}%.txt: init
	${PYTHON} train_ising.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --train_size ${TRAIN_SIZE} --test_size ${TEST_SIZE} --batch_size ${BATCH_SIZE} --train_iters ${TRAIN_ITERS} --lr ${LR} --task $@ > $@

# training with renn or bethe inference net
${TRAIN_LOG_NET}/${RESULTS}%.txt: init
	${PYTHON} train_ising.py --structure ${STRUCTURE} --sleep ${SLEEP} --device ${DEVICE} --train_size ${TRAIN_SIZE} --test_size ${TEST_SIZE} --batch_size ${BATCH_SIZE} --train_iters ${TRAIN_ITERS} --lr ${LR} --task $@ > $@


.SECONDARY:

.PRECIOUS:
