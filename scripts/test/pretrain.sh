#!/bin/bash

# Test
ID="Pretrain"
MODE='test'
CHECKPOINT="results/models/Pretrain"    # Also try the best trained model (results/models/xxx-best)
TEACHER_ENABLE=false
TEACHER_LEARN=false
WITH_FRICTION=false
FRICTION_CART=20

PLOT_PHASE=true
PLOT_TRAJECTORY=true
ANIMATION_SHOW=true
LIVE_TRAJECTORY_SHOW=true
EVAL_RANDOM_RESET=false
ACTUATOR_NOISE=false
SAMPLE_POINTS=150

python main.py \
  general.id=${ID} \
  general.mode=${MODE} \
  general.checkpoint=${CHECKPOINT} \
  general.max_evaluation_steps=${SAMPLE_POINTS} \
  logger.fig_plotter.phase.plot=${PLOT_PHASE} \
  logger.fig_plotter.trajectory.plot=${PLOT_TRAJECTORY} \
  logger.live_plotter.animation.show=${ANIMATION_SHOW} \
  logger.live_plotter.live_trajectory.show=${LIVE_TRAJECTORY_SHOW} \
  cartpole.random_reset.eval=${EVAL_RANDOM_RESET} \
  cartpole.with_friction=${WITH_FRICTION} \
  cartpole.friction_cart=${FRICTION_CART} \
  cartpole.random_noise.actuator.apply=${ACTUATOR_NOISE} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  ha_teacher.teacher_learn=${TEACHER_LEARN}