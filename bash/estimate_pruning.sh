# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="25_epochs_res_192_spar_0.3" \
# trainer.max_epochs=25 \
# model.in_place_prune_ratio=0.3 \
# datamodule.train_size=192 \
# datamodule.test_size=288 

# python run.py \
# experiment=resnet_imagenette.yaml \
# logger.wandb.name="25_epochs_res_192_base" \
# trainer.max_epochs=25 \
# datamodule.train_size=192 \
# datamodule.test_size=288 

# python run.py \
# experiment=resnet_imagenette.yaml \
# logger.wandb.name="25_epochs_res_224_base" \
# trainer.max_epochs=25 \
# datamodule.train_size=224 \
# datamodule.test_size=312 

# python run.py \
# experiment=resnet_imagenette.yaml \
# logger.wandb.name="10_epochs_res_128_base" \
# trainer.max_epochs=10 \
# datamodule.train_size=128 \
# datamodule.test_size=192 

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="25_epochs_res_224_spar_0.3" \
# trainer.max_epochs=25 \
# model.in_place_prune_ratio=0.3 \
# datamodule.train_size=224 \
# datamodule.test_size=312

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.02" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.02 \
# datamodule.train_size=128 \
# datamodule.test_size=192

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.157" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.157 \
# datamodule.train_size=128 \
# datamodule.test_size=192

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.2414" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.2414 \
# datamodule.train_size=128 \
# datamodule.test_size=192

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.2835" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.2835 \
# datamodule.train_size=128 \
# datamodule.test_size=192
# python run.py \
# experiment=resnet_imagenette.yaml \
# logger.wandb.name="10_epochs_res_224_base" \
# trainer.max_epochs=10 \
# datamodule.train_size=224 \
# datamodule.test_size=312 

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.02" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.02 \
# datamodule.train_size=128 \
# datamodule.test_size=192

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.157" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.157 \
# datamodule.train_size=128 \
# datamodule.test_size=192

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.2414" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.2414 \
# datamodule.train_size=128 \
# datamodule.test_size=192

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_128_spar_0.2835" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.2835 \
# datamodule.train_size=128 \
# datamodule.test_size=192
# python run.py \
# experiment=resnet_imagenette.yaml \
# logger.wandb.name="10_epochs_res_224_base" \
# trainer.max_epochs=10 \
# datamodule.train_size=224 \
# datamodule.test_size=312 


# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_224_spar_0.02" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.02 \
# datamodule.train_size=224 \
# datamodule.test_size=312

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_224_spar_0.157" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.157 \
# datamodule.train_size=224 \
# datamodule.test_size=312

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_224_spar_0.2414" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.2414 \
# datamodule.train_size=224 \
# datamodule.test_size=312

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="10_epochs_res_224_spar_0.2835" \
# trainer.max_epochs=10 \
# model.in_place_prune_ratio=0.2835 \
# datamodule.train_size=224 \
# datamodule.test_size=312

# python run.py \
# experiment=resnet_imagenette_prune_in_place.yaml \
# logger.wandb.name="50_epochs_res_224_spar_0.3" \
# trainer.max_epochs=50 \
# model.in_place_prune_ratio=0.3 \
# datamodule.train_size=224 \
# datamodule.test_size=312

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="25_epochs_res_224_spar_0.4" \
trainer.max_epochs=25 \
model.in_place_prune_ratio=0.4 \
datamodule.train_size=224 \
datamodule.test_size=312



python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="25_epochs_res_192_spar_0.4" \
trainer.max_epochs=25 \
model.in_place_prune_ratio=0.4 \
datamodule.train_size=192 \
datamodule.test_size=288 

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_128_spar_0.023523200000000022" \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.023523200000000022 \
datamodule.train_size=128 \
datamodule.test_size=192

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_128_spar_0.2101792" \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.2101792 \
datamodule.train_size=128 \
datamodule.test_size=192

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_128_spar_0.3219552" \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.3219552 \
datamodule.train_size=128 \
datamodule.test_size=192

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_128_spar_0.37805120000000003," \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.37805120000000003, \
datamodule.train_size=128 \
datamodule.test_size=192

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="50_epochs_res_224_spar_0.4" \
trainer.max_epochs=50 \
model.in_place_prune_ratio=0.4 \
datamodule.train_size=224 \
datamodule.test_size=312

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_224_spar_0.023523200000000022" \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.023523200000000022 \
datamodule.train_size=224 \
datamodule.test_size=312

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_224_spar_0.2101792" \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.2101792 \
datamodule.train_size=224 \
datamodule.test_size=312

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_224_spar_0.3219552" \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.3219552 \
datamodule.train_size=223 \
datamodule.test_size=312

python run.py \
experiment=resnet_imagenette_prune_in_place.yaml \
logger.wandb.name="10_epochs_res_224_spar_0.37805120000000003," \
trainer.max_epochs=10 \
model.in_place_prune_ratio=0.37805120000000003, \
datamodule.train_size=224 \
datamodule.test_size=312