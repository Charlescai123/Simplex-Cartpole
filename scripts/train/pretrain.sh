#!/bin/bash

# Train
ID="Pretrain"
MODE='train'
CHECKPOINT=null
TEACHER_ENABLE=false
TEACHER_LEARN=false
WITH_FRICTION=false
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  ha_teacher.teacher_learn=${TEACHER_LEARN}