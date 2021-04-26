# neural-network-pdes
Neural Network Implicit Representation of Partial Differential Equations

## Training
```
python euler.py gpus=0 mlp.periodicity=2
```

## Plotting
```
python euler.py checkpoint=\"outputs/2021-04-25/18-54-45/lightning_logs/version_0/checkpoints/'epoch=0.ckpt'\" train=False
```