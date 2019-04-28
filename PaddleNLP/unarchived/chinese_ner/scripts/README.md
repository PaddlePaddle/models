## Purpose of this directory
The purpose of this directory is to provide exemplary execution commands. They are inside bash scripts described below.

## Preparation
To add execution permissions for shell scripts, run in this directory:
`chmod +x *.sh`

## Performance tips
Use the below environment flags for best performance:
```
KMP_AFFINITY=granularity=fine,compact,1,0
OMP_NUM_THREADS=<num_of_physical_cores>
```
For example, you can export them, or add them inside the specific files.

## Training
### CPU with mkldnn
Run:
`./train.sh MKLDNN`
### CPU without mkldnn
Run:
`./train.sh CPU`
### GPU
Run:
`./train.sh GPU`

## Inference
### CPU with mkldnn
Run:
`./infer.sh MKLDNN`
### CPU without mkldnn
Run:
`./infer.sh CPU`
### GPU
Run:
`./infer.sh GPU`
