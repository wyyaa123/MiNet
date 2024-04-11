# MiNet
Implement for image restoration🚀. the dataset used in the project, include GoPro, HIDE, RealBlur. You can download them on datasets folder or other path. It is recommended to symlink your dataset root to this folder - datasets with the command ln -s xxx yyy.

## Installation
```
python 3.9
pytorch 1.13.0
cuda 11.3
cudnn 8.6.2
```
```
git clone https://github.com/wyyaa123/MiNet.git
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

## train
```
python -m torch.distributed.run --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/ReformerNet/ReformerNet.yml --launcher pytorch
```
or you can use:
```
python basicsr/train.py -opt options/train/ReformerNet/ReformerNet.yml
```

## test
for a dataset:
```
python basicsr/test.py -opt options/test/ReformerNet/ReformerNet.yml
```

## infer
```
python scripts/inference/infer_simple.py -m your_model_path -i your_images_path -o saved_path
```

