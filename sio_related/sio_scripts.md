# CIFAR-10/CIFAR-100

For all CIFAR experiments we were using one Quadro RTX 6000 GPU (24GB memory).

## Inference techniques
```bash
DEVICE=5
SEED=0

DSET=cifar10 # cifar100
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 200 \
    --seed ${SEED}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark extradata-stylegan-ratio${R}

# test
DIR=./results/${DSET}_resnet18_32x32_base_e200_lr0.1_extradata-stylegan-ratio${R}
#DIR=./results/${DSET}_resnet18_32x32_base_e200_lr0.1_default

for m in msp openmax temp_scaling odin ebo mls klm vim knn dice; do
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python main.py \
        --config configs/datasets/${DSET}/${DSET}.yml \
        configs/datasets/${DSET}/${DSET}_ood.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/test/test_ood_multiruns.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/${m}.yml \
        --num_workers 8 \
        --network.ckpt_dir ${DIR} \
        --merge_option merge
done

# react
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/react_net.yml \
    configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/react.yml \
    --network.pretrained False \
    --network.backbone.name resnet18_32x32 \
    --network.backbone.pretrained True \
    --network.backbone.ckpt_dir ${DIR} \
    --num_workers 8 \
    --merge_option merge
```

## Training algorithms
- G-ODIN
```bash
DEVICE=5
SEED=0

DSET=cifar10
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/networks/godin_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/godin.yml \
    --network.backbone.name resnet18_32x32 \
    --num_workers 8 \
    --trainer.name godin \
    --optimizer.num_epochs 200 \
    --seed ${SEED}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/networks/godin_net.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/godin.yml \
    --network.backbone.name resnet18_32x32 \
    --num_workers 8 \
    --trainer.name godin \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark extradata-stylegan-ratio${R}

# test
DIR=./results/${DSET}_godin_net_godin_e200_lr0.1_extradata-stylegan-ratio${R}
#DIR=./results/${DSET}_godin_net_godin_e200_lr0.1_default

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/godin_net.yml \
    configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/godin.yml \
    --network.ckpt_dir ${DIR} \
    --num_workers 8 \
    --merge_option merge
```


- VOS
```bash
DEVICE=2
SEED=0

DSET=cifar10
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_vos.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --optimizer.num_epochs 200 \
    --seed ${SEED}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_vos.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark extradata-stylegan-ratio${R}

# test
DIR=./results/${DSET}_resnet18_32x32_vos_e200_lr0.1_extradata-stylegan-ratio${R}
#DIR=./results/${DSET}_resnet18_32x32_vos_e200_lr0.1_default

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/resnet18_32x32.yml \
    custom_configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/ebo.yml \
    --num_workers 8 \
    --network.ckpt_dir ${DIR} \
    --merge_option merge
```


- LogitNorm
```bash
DEVICE=5
SEED=0

DSET=cifar10
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_logitnorm.yml \
    --optimizer.num_epochs 200 \
    --seed ${SEED}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_logitnorm.yml \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark extradata-stylegan-ratio${R}

# test
DIR=${DSET}_resnet18_32x32_logitnorm_e200_lr0.1_alpha0.04_extradata-stylegan-ratio${R}
#DIR=${DSET}_resnet18_32x32_logitnorm_e200_lr0.1_alpha0.04_default

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/resnet18_32x32.yml \
    custom_configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.ckpt_dir ./results/ \
    --merge_option merge
```


- CutMix
```bash
DEVICE=5
SEED=0

DSET=cifar10
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_cutmix.yml \
    --optimizer.num_epochs 200 \
    --seed ${SEED}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_cutmix.yml \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark extradata-stylegan-ratio${R}

# test
DIR=${DSET}_resnet18_32x32_cutmix_e200_lr0.1_extradata-stylegan-ratio${R}
#DIR=${DSET}_resnet18_32x32_cutmix_e200_lr0.1_default

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/resnet18_32x32.yml \
    custom_configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.ckpt_dir ./results/ \
    --merge_option merge
```

- PixMix
```bash
DEVICE=5
SEED=0

DSET=cifar10
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/preprocessors/pixmix_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    --preprocessor.name pixmix \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --mark pixmix

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/preprocessors/pixmix_preprocessor.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    --preprocessor.name pixmix \
    --seed ${SEED} \
    --optimizer.num_epochs 200 \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark pixmix_extradata-stylegan-ratio${R}

# test
DIR=${DSET}_resnet18_32x32_base_e200_lr0.1_pixmix_extradata-stylegan-ratio${R}
#DIR=${DSET}_resnet18_32x32_base_e200_lr0.1_pixmix

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/resnet18_32x32.yml \
    custom_configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.ckpt_dir ./results/ \
    --merge_option merge
```

- OE
```bash
DEVICE=6
SEED=0

DSET=cifar10
R=0.8

# train
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_oe.yml \
    configs/networks/resnet18_32x32.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_oe.yml \
    --dataset.oe.batch_size 256 \
    --optimizer.num_epochs 200 \
    --num_workers 8 \
    --seed ${SEED}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}_extra_data.yml \
    configs/datasets/${DSET}/${DSET}_oe.yml \
    configs/networks/resnet18_32x32.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_oe.yml \
    --dataset.oe.batch_size 256 \
    --optimizer.num_epochs 200 \
    --num_workers 8 \
    --seed ${SEED} \
    --dataset.train.extra_data_pth ./data/images_classic/${DSET}_extra/stylegan/images.npy \
    --dataset.train.extra_label_pth ./data/images_classic/${DSET}_extra/stylegan/labels.npy \
    --dataset.train.orig_ratio ${R} \
    --mark extradata-stylegan-ratio${R}

# test
DIR=${DSET}_resnet18_32x32_oe_e200_lr0.1_extradata-stylegan-ratio${R}
#DIR=${DSET}_resnet18_32x32_oe_e200_lr0.1_default

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/${DSET}/${DSET}.yml \
    configs/datasets/${DSET}/${DSET}_ood.yml \
    configs/networks/resnet18_32x32.yml \
    custom_configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/msp.yml \
    --num_workers 8 \
    --network.ckpt_dir ./results/ \
    --merge_option merge
```

# ImageNet

For ImageNet experiments we were using two Quadro RTX 6000 GPUs.

## Inference techniques
```bash
DIST_URL="tcp://127.0.0.1:8880"
INET_DIR="/home/public/ImageNet"

DEVICES=6,7
BS=256
LR=0.1
WORKERS=8

SEED=0
SPLIT="10" # dogs
EXTRA_DATA_DIR="./data/images_largescale/imagenet_extra/imagenet_${SPLIT}"

# train
CUDA_VISIBLE_DEVICES=${DEVICES} \
python imagenet_train_subclasses.py \
    --dist-url ${DIST_URL} \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --workers ${WORKERS} \
    --data ${INET_DIR} \
    --batch-size ${BS} --lr ${LR} \
    --epochs 60 \
    --seed ${seed} \
    --sub-classes-split ${SPLIT}

# train with SIO
CUDA_VISIBLE_DEVICES=${DEVICES} \
python imagenet_train_subclasses.py \
    --dist-url ${DIST_URL} \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --workers ${WORKERS} \
    --data #{INET_DIR} \
    --batch-size ${BS} --lr ${LR} \
    --epochs 60 \
    --seed ${seed} \
    --sub-classes-split ${SPLIT} \
    --extra-data ${EXTRA_DATA_DIR} \
    --real-ratio 0.9


# test
DEVICE=7
BS=200
DIR=./results/imagenet-${SPLIT}_resnet18_224x224_e60_bs256_lr0.1_extradata-biggan-ratio0.90
#DIR=./results/imagenet-${SPLIT}_resnet18_224x224_e60_bs256_lr0.1_default

for m in temp_scaling odin ebo gradnorm mls klm openmax knn dice; do
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    python main.py \
        --config configs/datasets/imagenet/imagenet_${SPLIT}.yml \
        configs/datasets/imagenet/imagenet_non-${SPLIT}.yml \
        configs/networks/resnet18_224x224.yml \
        configs/pipelines/test/test_ood_multiruns.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/${m}.yml \
        --num_workers 8 \
        --network.ckpt_dir ${DIR} \
        --ood_dataset.batch_size ${BS} \
        --dataset.test.batch_size ${BS} \
        --dataset.val.batch_size ${BS} \
        --merge_option merge
done

# react
CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --config configs/datasets/imagenet/imagenet_${SPLIT}.yml \
    configs/datasets/imagenet/imagenet_non-${SPLIT}.yml \
    configs/networks/react_net.yml \
    configs/pipelines/test/test_ood_multiruns.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/react.yml \
    --network.pretrained False \
    --network.backbone.name resnet18_224x224 \
    --network.backbone.pretrained True \
    --network.backbone.ckpt_dir ${DIR} \
    --num_workers 8 \
    --ood_dataset.batch_size ${BS} \
    --dataset.test.batch_size ${BS} \
    --dataset.val.batch_size ${BS} \
    --merge_option merge
```
