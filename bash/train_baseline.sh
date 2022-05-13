# python run.py experiment=resnet_imagenette.yaml model.lr=0.001 seed=42 logger.wandb.name="resnet34_base_lr_0.001"
# python run.py experiment=resnet_imagenette.yaml model.lr=0.001 seed=12345 logger.wandb.name="resnet34_base_lr_0.001"
# python run.py experiment=resnet_imagenette.yaml model.lr=0.003 seed=42 logger.wandb.name="resnet34_base_lr_0.003"
# python run.py experiment=resnet_imagenette.yaml model.lr=0.003 seed=12345 logger.wandb.name="resnet34_base_lr_0.003"
# python run.py experiment=resnet_imagenette.yaml model.lr=0.01 seed=42 logger.wandb.name="resnet34_base_lr_0.01"
# python run.py experiment=resnet_imagenette.yaml model.lr=0.01 seed=12345 logger.wandb.name="resnet34_base_lr_0.01"

python run.py experiment=resnet_imagenette.yaml model.lr=0.0005 seed=42 logger.wandb.name="resnet34_base_lr_0.0005"
python run.py experiment=resnet_imagenette.yaml model.lr=0.0005 seed=12345 logger.wandb.name="resnet34_base_lr_0.0005"

python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.0005 seed=42 logger.wandb.name="resnet18_base_lr_0.0005"
python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.0005 seed=12345 logger.wandb.name="resnet18_base_lr_0.0005"

python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.001 seed=42 logger.wandb.name="resnet18_base_lr_0.001"
python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.001 seed=12345 logger.wandb.name="resnet18_base_lr_0.001"

python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.005 seed=42 logger.wandb.name="resnet18_base_lr_0.005"
python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.005 seed=12345 logger.wandb.name="resnet18_base_lr_0.005"

python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.01 seed=42 logger.wandb.name="resnet18_base_lr_0.01"
python run.py experiment=resnet_imagenette.yaml model.model="resnet18" model.lr=0.01 seed=12345 logger.wandb.name="resnet18_base_lr_0.01"