# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.0001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=75 \
# callbacks.pruning.amount.freq=15 \
# logger.wandb.name="resnet34_self_kd_lr_0.0001_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.0001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=75 \
# callbacks.pruning.amount.freq=15 \
# logger.wandb.name="resnet34_self_kd_lr_0.0001_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.0005 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=75 \
# callbacks.pruning.amount.freq=15 \
# logger.wandb.name="resnet34_self_kd_lr_0.0005_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.0005 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=75 \
# callbacks.pruning.amount.freq=15 \
# logger.wandb.name="resnet34_self_kd_lr_0.0005_100_epochs"



# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# callbacks.pruning.amount.freq=10 \
# logger.wandb.name="resnet34_self_kd_lr_0.001_si_0.25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# callbacks.pruning.amount.freq=10 \
# logger.wandb.name="resnet34_self_kd_lr_0.001_si_0.25_100_epochs"
python run.py experiment=resnet_imagenette_prune_kd.yaml \
model.lr=0.003 \
model.hard_weight=0.1 \
model.soft_weight=0.9 \
model.warmup=0 \
model.student_model="resnet50" \
model.teacher_model="resnet50" \
seed=42 \
trainer.max_epochs=100 \
callbacks.pruning.amount.ending_epoch=31 \
callbacks.pruning.amount.freq=10 \
callbacks.pruning.amount.initial_sparsity=0.25 \
logger.wandb.name="resnet50_self_kd_lr_0.001_freq_20_100_epochs"

python run.py experiment=resnet_imagenette_prune_kd.yaml \
model.lr=0.003 \
model.hard_weight=0.1 \
model.soft_weight=0.9 \
model.warmup=0 \
model.student_model="resnet50" \
model.teacher_model="resnet50" \
seed=42 \
trainer.max_epochs=100 \
callbacks.pruning.amount.ending_epoch=31 \
callbacks.pruning.amount.freq=10 \
callbacks.pruning.amount.initial_sparsity=0.1 \
logger.wandb.name="resnet50_self_kd_lr_0.001_freq_20_100_epochs"



python run.py experiment=resnet_imagenette_prune_kd.yaml \
model.lr=0.003 \
model.hard_weight=0.1 \
model.soft_weight=0.9 \
model.warmup=0 \
model.student_model="resnet50" \
model.teacher_model="resnet50" \
seed=42 \
trainer.max_epochs=100 \
callbacks.pruning.amount.ending_epoch=51 \
callbacks.pruning.amount.freq=20 \
callbacks.pruning.amount.initial_sparsity=0.1 \
logger.wandb.name="resnet50_self_kd_lr_0.003_freq_20_100_epochs"



python run.py experiment=resnet_imagenette_prune_kd.yaml \
model.lr=0.001 \
model.hard_weight=0.1 \
model.soft_weight=0.9 \
model.warmup=0 \
model.student_model="resnet50" \
model.teacher_model="resnet50" \
seed=42 \
trainer.max_epochs=100 \
callbacks.pruning.amount.ending_epoch=51 \
callbacks.pruning.amount.freq=20 \
callbacks.pruning.amount.initial_sparsity=0.1 \
logger.wandb.name="resnet50_self_kd_lr_0.001_freq_20_100_epochs"



# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# model.student_model="resnet50" \
# model.teacher_model="resnet50" \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=51 \
# callbacks.pruning.amount.freq=20 \
# callbacks.pruning.amount.initial_sparsity=0.1 \
# logger.wandb.name="resnet50_self_kd_lr_0.001_freq_25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.0001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# model.student_model="resnet50" \
# model.teacher_model="resnet50" \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=51 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.1 \
# logger.wandb.name="resnet50_self_kd_lr_0.001_freq_25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.0001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# model.student_model="resnet50" \
# model.teacher_model="resnet50" \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=51 \
# callbacks.pruning.amount.freq=20 \
# callbacks.pruning.amount.initial_sparsity=0.1 \
# logger.wandb.name="resnet50_self_kd_lr_0.001_freq_25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# model.student_model="resnet50" \
# model.teacher_model="resnet50" \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=25 \
# callbacks.pruning.amount.initial_sparsity=0.2 \
# logger.wandb.name="resnet50_self_kd_lr_0.001_freq_25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# callbacks.pruning.amount.freq=20 \
# logger.wandb.name="resnet34_self_kd_lr_0.001_freq_20_si_0.25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# callbacks.pruning.amount.freq=20 \
# logger.wandb.name="resnet34_self_kd_lr_0.001_freq_20_si_0.25_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=10 \
# logger.wandb.name="resnet34_self_kd_lr_0.001_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.001 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=10 \
# logger.wandb.name="resnet34_self_kd_lr_0.001_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=7 \
# logger.wandb.name="resnet34_self_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=7 \
# logger.wandb.name="resnet34_self_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=7 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=7 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=7 \
# logger.wandb.name="resnet34_self_kd_lr_0.01_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=7 \
# logger.wandb.name="resnet34_self_kd_lr_0.01_100_epochs"


# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=7 \
# logger.wandb.name="resnet34_self_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=7 \
# logger.wandb.name="resnet34_self_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=7 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_0.01_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=7 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_0.01_100_epochs"






####################################################################################




# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.01_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=50 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.01_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.01_100_epochs"

# python run.py experiment=resnet_imagenette_prune_kd.yaml \
# model.lr=0.01 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# callbacks.pruning.amount.ending_epoch=30 \
# callbacks.pruning.amount.freq=10 \
# callbacks.pruning.amount.initial_sparsity=0.25 \
# logger.wandb.name="resnet34_self_kd_lr_si_0.25_0.01_100_epochs"

############### Epoch Length ###################

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=10 \
# logger.wandb.name="resnet18_kd_lr_0.003_10_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=10 \
# logger.wandb.name="resnet18_kd_lr_0.003_10_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=25 \
# logger.wandb.name="resnet18_kd_lr_0.003_25_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=25 \
# logger.wandb.name="resnet18_kd_lr_0.003_25_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=50 \
# logger.wandb.name="resnet18_kd_lr_0.003_50_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=50 \
# logger.wandb.name="resnet18_kd_lr_0.003_50_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=100 \
# logger.wandb.name="resnet18_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=100 \
# logger.wandb.name="resnet18_kd_lr_0.003_100_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=42 \
# trainer.max_epochs=150 \
# logger.wandb.name="resnet18_kd_lr_0.003_150_epochs"

# python run.py experiment=resnet_imagenette_kd.yaml \
# model.lr=0.003 \
# model.hard_weight=0.1 \
# model.soft_weight=0.9 \
# seed=12345 \
# trainer.max_epochs=150 \
# logger.wandb.name="resnet18_kd_lr_0.003_150_epochs"







