#!/bin/bash

# Train
ID="Unsafe-Continual-Learn"
MODE='train'
CHECKPOINT="results/models/Pretrain"
TEACHER_ENABLE=false
TEACHER_LEARN=false
WITH_FRICTION=true
FRICTION_CART=20
ACTUATOR_NOISE=true
TRAIN_RANDOM_RESET=true
EVAL_RANDOM_RESET=true
#TRAIN_RANDOM_RESET=false
#EVAL_RANDOM_RESET=false

TRAINING_BY_STEPS=true
MAX_TRAINING_EPISODES=1e3
#TRAINING_BY_STEPS=false
#MAX_TRAINING_EPISODES=10

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  general.training_by_steps=${TRAINING_BY_STEPS} \
  general.max_training_episodes=${MAX_TRAINING_EPISODES} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.random_reset.train=${TRAIN_RANDOM_RESET} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.random_noise.actuator.apply=${ACTUATOR_NOISE} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  ha_teacher.teacher_learn=${TEACHER_LEARN}