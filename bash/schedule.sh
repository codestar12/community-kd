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

# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.2 model.student_layers="[[1,1]]" name=one_student_codistill_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.2 model.student_layers="[[1,1]]" name=one_student_codistill_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.3 model.student_layers="[[1,1]]" name=one_student_codistill_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.3 model.student_layers="[[1,1]]" name=one_student_codistill_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.4 model.student_layers="[[1,1]]" name=one_student_codistill_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.4 model.student_layers="[[1,1]]" name=one_student_codistill_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.5 model.student_layers="[[1,1]]" name=one_student_codistill_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=one_student_codistill.yaml model.hard_label_end=0.5 model.student_layers="[[1,1]]" name=one_student_codistill_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.2 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.2 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.3 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.3 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.4 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.4 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.5 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_42_11 logger.wandb.project="hydra_half_community" seed=42
# python run.py experiment=two_student_all2all.yaml model.hard_label_end=0.5 model.num_students=2 model.student_layers="[[1,1], [1,1]]" name=two_student_all2all_seed_12345_11 logger.wandb.project="hydra_half_community" seed=12345

# python run.py experiment=resnet_imagenette_prune.yaml \
#     callbacks.pruning.amount.final_sparsity=0.5 \
#     callbacks.pruning.amount.starting_epoch=99 \
#     callbacks.pruning.amount.ending_epoch=149 \
#     callbacks.pruning.amount.freq=10 \
#     logger.wandb.name="prune_eight_gpus_0.5"

# python run.py experiment=resnet_imagenette_resize_prune.yaml \
#     callbacks.pruning.amount.final_sparsity=0.3 \
#     callbacks.pruning.amount.starting_epoch=99 \
#     callbacks.pruning.amount.ending_epoch=149 \
#     callbacks.pruning.amount.freq=10 \
#     logger.wandb.name="resize_prune_eight_gpus_0.3"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.3 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.3"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.3 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.3"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.3 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.3"


python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.3 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.3"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.35 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=1234 \
    logger.wandb.name="prune_eight_gpus_0.35"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.35 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.35"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.35 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.35"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.35 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=1234 \
    logger.wandb.name="resize_prune_eight_gpus_0.35"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.35 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.35"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.35 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.35"


python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.4 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.4"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.4 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.4"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.4 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.4"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.4 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.4"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.5 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.5"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.5 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.5"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.5 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.5"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.5 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.5"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.6 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.6"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.6 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.6"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.6 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.6"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.6 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.6"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.7 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.7"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.7 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.7"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.7 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.7"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.7 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.7"


python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.8 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="prune_eight_gpus_0.8"

python run.py experiment=resnet_imagenette_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.8 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="prune_eight_gpus_0.8"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.8 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=42 \
    logger.wandb.name="resize_prune_eight_gpus_0.8"

python run.py experiment=resnet_imagenette_resize_prune.yaml \
    callbacks.pruning.amount.final_sparsity=0.8 \
    callbacks.pruning.amount.starting_epoch=99 \
    callbacks.pruning.amount.ending_epoch=149 \
    callbacks.pruning.amount.freq=10 \
    seed=35 \
    logger.wandb.name="resize_prune_eight_gpus_0.8"












