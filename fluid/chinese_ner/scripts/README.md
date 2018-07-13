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
`train_profile_mkldnn.sh`
### CPU without mkldnn
Run:
`train_profile.sh`

## Inference
### CPU with mkldnn
Run:
`infer_profile_mkldnn.sh`
### CPU without mkldnn
Run:
`infer_profile.sh`
