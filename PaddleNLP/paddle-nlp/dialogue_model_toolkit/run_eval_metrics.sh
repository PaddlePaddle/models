#!/bin/bash

TASK_NAME=$1
PRED_FILE="./pred_"${TASK_NAME}
PYTHON_PATH="python"

echo "run predict............................"
sh run_predict.sh ${TASK_NAME} > ${PRED_FILE}

echo "eval_metrics..........................."
${PYTHON_PATH} eval_metrics.py ${TASK_NAME} ${PRED_FILE}

