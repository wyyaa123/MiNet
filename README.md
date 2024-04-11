# MiNet
implement for image restoration🚀


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

