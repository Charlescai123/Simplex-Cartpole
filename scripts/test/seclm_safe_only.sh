#!/bin/bash

# Test
ID="SeCLM-Safe-Only"
MODE='test'
CHECKPOINT="results/models/SeCLM-Safe-Only"
TEACHER_ENABLE=true
TEACHER_LEARN=false
WITH_FRICTION=true
FRICTION_CART=18


python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  coordinator.teacher_learn=${TEACHER_LEARN}