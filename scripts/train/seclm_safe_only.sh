#!/bin/bash

# Train
ID="SeCLM-Safe-Only"
MODE='train'
CHECKPOINT="results/models/Pretrain"
TEACHER_ENABLE=true
TEACHER_LEARN=false
WITH_FRICTION=true
FRICTION_CART=20
ACTUATOR_NOISE=true
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.random_noise.actuator.apply=${ACTUATOR_NOISE} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  coordinator.teacher_learn=${TEACHER_LEARN}