#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh


# python run.py experiment=imagenet_baseline_hydra.yaml model.backbone="Hydra29" name=hydra29_baseline_test logger.wandb.project="hydra_imagenet"
# python run.py experiment=imagenet_baseline_hydra.yaml model.backbone="Hydra50" name=hydra50_baseline_test logger.wandb.project="hydra_imagenet"
python run.py experiment=imagenet_one_student_codistill.yaml name=one_student_codistill logger.wandb.project="hydra_imagenet"

