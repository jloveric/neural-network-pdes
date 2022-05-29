# neural-network-pdes
Neural Network Implicit Representation of Partial Differential Equations.  The images are very diffuse solutions to the Sod Shock problem in fluid dynamics, which is 1 dimensional ideal compressible gas dynamics (euler equations), a very classic example.  The solution takes a long time to get to, but if nothing else, shows that this problem can probably be solved using implicit representation.  I'm using fourier layers in the example below and you can see a rise in density before the "shock", that density rise should not be there...  I'll continue to try and refine this to get better solutions.  However the, "shock" and rarefaction wave are visible.

![Sod Shock Density](images/Density-fourier.png)
![Sod Shock Velocity](images/Velocity-fourier.png)

and with continuous piecewise polynomial layers

![Sod Shock Density](images/Density-continuous.png)
![Sod Shock Velocity](images/Velocity-continuous.png)

## Training
```
python examples/euler.py gpus=0 mlp.periodicity=2
```

## Plotting
```
python examples/euler.py checkpoint=\"outputs/2021-04-25/18-54-45/lightning_logs/version_0/checkpoints/'epoch=0.ckpt'\" train=False
```

## Running tests
```
pytest tests
```