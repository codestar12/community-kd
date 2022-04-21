#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh


# python run.py experiment=imagenet_baseline_hydra.yaml model.backbone="Hydra29" name=hydra29_baseline_test logger.wandb.project="hydra_imagenet"
# python run.py experiment=imagenet_baseline_hydra.yaml model.backbone="Hydra50" name=hydra50_baseline_test logger.wandb.project="hydra_imagenet"
python run.py \
    experiment=two_student_all2all.yaml \
    name=two_student_11_22_delay2_codistill \
    model.kd_delay=2 \
    model.student_layers="[[1,1], [2,2]]" \
    logger.wandb.project="hydra_imagenet"

python run.py \
    experiment=two_student_all2all.yaml \
    name=two_student_11_33_delay2_codistill \
    model.kd_delay=2 \
    model.student_layers="[[1,1], [3,3]]" \
    logger.wandb.project="hydra_imagenet"




