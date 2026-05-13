# PowerShell version for Windows.
# 1. pip install -r requirements.txt
# 2. python download_data.py --data_root ./data/flowers102
# 3. $env:LOGGER="wandb"   # or swanlab / none

if (-not $env:LOGGER) { $env:LOGGER = "wandb" }
if (-not $env:EPOCHS) { $env:EPOCHS = "30" }
if (-not $env:BATCH_SIZE) { $env:BATCH_SIZE = "32" }

python train.py --data_root ./data/flowers102 --out_dir ./outputs --model resnet18 --pretrained --epochs $env:EPOCHS --batch_size $env:BATCH_SIZE --lr_backbone 1e-4 --lr_head 1e-3 --logger $env:LOGGER --run_name baseline_resnet18_ep$($env:EPOCHS)_bb1e-4_head1e-3

python train.py --data_root ./data/flowers102 --out_dir ./outputs --model resnet18 --pretrained --epochs $env:EPOCHS --batch_size $env:BATCH_SIZE --lr_backbone 3e-5 --lr_head 3e-4 --logger $env:LOGGER --run_name baseline_resnet18_ep$($env:EPOCHS)_bb3e-5_head3e-4

python train.py --data_root ./data/flowers102 --out_dir ./outputs --model resnet18 --pretrained --epochs $env:EPOCHS --batch_size $env:BATCH_SIZE --lr_backbone 1e-5 --lr_head 1e-3 --logger $env:LOGGER --run_name baseline_resnet18_ep$($env:EPOCHS)_bb1e-5_head1e-3

python train.py --data_root ./data/flowers102 --out_dir ./outputs --model resnet18 --no-pretrained --epochs $env:EPOCHS --batch_size $env:BATCH_SIZE --lr_backbone 1e-3 --lr_head 1e-3 --logger $env:LOGGER --run_name ablation_resnet18_random_ep$($env:EPOCHS)

python train.py --data_root ./data/flowers102 --out_dir ./outputs --model se_resnet18 --pretrained --epochs $env:EPOCHS --batch_size $env:BATCH_SIZE --lr_backbone 1e-4 --lr_head 1e-3 --logger $env:LOGGER --run_name attention_se_resnet18_ep$($env:EPOCHS)

python train.py --data_root ./data/flowers102 --out_dir ./outputs --model cbam_resnet18 --pretrained --epochs $env:EPOCHS --batch_size $env:BATCH_SIZE --lr_backbone 1e-4 --lr_head 1e-3 --logger $env:LOGGER --run_name attention_cbam_resnet18_ep$($env:EPOCHS)
