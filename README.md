# Inference Code for RetinaFace with MobileNet Backend in PyTorch

### Step 1:
```Shell
cd cython
python setup.py build_ext --inplace
```

### Step 2:
```Shell
python inference.py
```

### Test Results:
<img src="https://github.com/bogireddytejareddy/retinaface-pytorch/blob/master/test_results/t6.jpg" width="480" height="300">
<img src="https://github.com/bogireddytejareddy/retinaface-pytorch/blob/master/test_results/t3.jpg" width="480" height="300">

### References:
@inproceedings{deng2019retinaface, title={RetinaFace: Single-stage Dense Face Localisation in the Wild}, author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos}, booktitle={arxiv}, year={2019} }
