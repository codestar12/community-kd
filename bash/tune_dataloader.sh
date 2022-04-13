#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh


# python run.py experiment=baseline.yaml name=resnet34_baseline logger.wandb.project="community-scaling"
# python run.py experiment=baseline.yaml model.teacher_model='resnet18' name=resnet18_baseline logger.wandb.project="community-scaling"
#no delay
# python run.py experiment=one_student_hardonly.yaml name=one_student_hardonly_delay_1 logger.wandb.project="hydra_classifier_community"
# python run.py experiment=one_student_kd.yaml name=one_student_kd_delay_1 logger.wandb.project="hydra_classifier_community"
# python run.py experiment=one_student_codistill.yaml name=one_student_codistill_delay_1 logger.wandb.project="hydra_classifier_community"
# python run.py experiment=baseline.yaml name=hydra50_baseline logger.wandb.project="hydra_classifier_community"

# python run.py experiment=one_student_hardonly.yaml name=one_student_hardonly_delay_5 model.kd_delay=5 logger.wandb.project="hydra_classifier_community"

# python run.py experiment=baseline.yaml name=baseline_seed_42 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=baseline.yaml name=baseline_seed_0 logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=baseline.yaml name=baseline_seed_12345 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=one_student_codistill.yaml name=one_student_codistill_seed_42 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml name=one_student_codistill_seed_0 logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=one_student_codistill.yaml name=one_student_codistill_seed_12345 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=one_student_codistill.yaml model.student_layers="[[2,2]]" name=one_student_codistill_seed_42_22 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.student_layers="[[2,2]]" name=one_student_codistill_seed_0_22 logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=one_student_codistill.yaml model.student_layers="[[2,2]]" name=one_student_codistill_seed_12345_22 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=one_student_codistill.yaml model.student_layers="[[6,3]]" name=one_student_codistill_seed_42_63 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.student_layers="[[6,3]]" name=one_student_codistill_seed_0_22 logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=one_student_codistill.yaml model.student_layers="[[6,3]]" name=one_student_codistill_seed_12345_63 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=two_student_all2all.yaml name=two_student_all2all_seed_42 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=two_student_all2all.yaml name=two_student_all2all_seed_0 logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=two_student_all2all.yaml name=two_student_all2all_seed_12345 logger.wandb.project="hydra_half_community" seed=1234

# python run.py experiment=baseline.yaml name=baseline_35_seed_42 model.backbone="Hydra35" logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=baseline.yaml name=baseline_41_seed_42 model.backbone="Hydra41" logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=baseline.yaml name=baseline_35_seed_12345 model.backbone="Hydra35" logger.wandb.project="hydra_half_community" seed=12345
# python run.py experiment=baseline.yaml name=baseline_41_seed_12345 model.backbone="Hydra41" logger.wandb.project="hydra_half_community" seed=12345



# python run.py experiment=baseline.yaml model.periods=2 name=baseline_29_t2_seed_42 model.backbone="Hydra29" logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=baseline.yaml model.periods=2 name=baseline_29_t2_seed_0 model.backbone="Hydra29" logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=baseline.yaml model.periods=2 name=baseline_29_t2_seed_12345 model.backbone="Hydra29" logger.wandb.project="hydra_half_community" seed=12345
# python run.py experiment=baseline.yaml model.periods=2 name=baseline_t2_seed_42 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=baseline.yaml model.periods=2 name=baseline_t2_seed_0 logger.wandb.project="hydra_half_community" seed=0
# python run.py experiment=baseline.yaml model.periods=2 name=baseline_t2_seed_12345 logger.wandb.project="hydra_half_community" seed=12345
# python run.py experiment=one_student_codistill.yaml model.periods=2 model.student_layers="[[1,1]]" name=one_student_codistill_t2_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.periods=2 model.student_layers="[[1,1]]" name=one_student_codistill_t2_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.periods=2 model.student_layers="[[1,1]]" name=one_student_codistill_t2_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

python run.py experiment=baseline_hydra_imagenette.yaml name=160_train_bs256_workers9_pin_memory_false datamodule.pin_memory=False datamodule.num_workers=9 logger.wandb.project="hydra_half_community" seed=12345
python run.py experiment=baseline_hydra_imagenette.yaml name=160_train_bs256_workers6_pin_memory_false datamodule.pin_memory=False datamodule.num_workers=6 logger.wandb.project="hydra_half_community" seed=12345
python run.py experiment=baseline_hydra_imagenette.yaml name=160_train_bs256_workers4_pin_memory_false datamodule.pin_memory=False datamodule.num_workers=4 logger.wandb.project="hydra_half_community" seed=12345