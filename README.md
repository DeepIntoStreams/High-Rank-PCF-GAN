# High Rank Path Development: an approach of learning the filtration of stochastic processes

## Environment Setup
The code has been successfully tested using Python 3.10 and PyTorch 1.11 with GPU support. Therefore, we suggest using this version or a later version of Python and PyTorch. A typical process for installing the package dependencies involves creating a new Python virtual environment.

```console
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install signatory==1.2.6.1.9.0 --no-cache-dir --force-reinstall
pip install git+https://github.com/tgcsaba/ksig.git --no-deps
pip install -r requirements.txt
```
The source code of the HRPCF-GAN is in the [src](src/) subdirectory.

## Reproducing experiments
 To reproduce the experiments involved with High-Rank PCFGAN illustrated in the paper, one can use [run.py](run.py) script, which takes two main arguments to specify the GAN model and benchmark dataset, i.e. 
 ```console
 python3 run.py --dataset DATASET
 ```   
 where `DATASET` is either `fBM`, or `Stock`.

 To reproduce the benchmarking models illustrated in the paper, run [baseline.py](baseline.py) using the following command:
 ```console
 python3 baseline.py
 ```

For the hypothesis experiment, one can run
```console
python3 HT.py
```

For the american put option pricing experiment, one can run
```console
python3 american_option.py
```
