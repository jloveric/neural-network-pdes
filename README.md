# neural-network-pdes
Neural Network Implicit Representation of Partial Differential Equations.  The problems here are solved using a simple high order MLP where
the the input is (x,t) in 1D and the output is density, velocity and pressure.  The loss function is partial differential equation for the 1d euler equations of gas dynamics dotted with itself.

## Sod Shock

"Solutions" to the Sod Shock problem in fluid dynamics, which is 1 dimensional ideal compressible gas dynamics (euler equations), a very classic problem.  Solutions need to be much better (and faster) to be 

## Fourier layers

![Sod Shock Density](images/Density-fourier.png)
![Sod Shock Velocity](images/Velocity-fourier.png)

### Continuous piecewise polynomial layers

![Sod Shock Density](images/Density-continuous.png)
![Sod Shock Velocity](images/Velocity-continuous.png)

### Discontinuous piecewise polynomial layers

In this case it gets
the initial condition almost exactly right and can produce genuine discontinuities, but it's clearly showing many wrong and entropy violating shocks (not to mention not conserving mass etc...).  I need to add some penalty function here to prevent bad shocks.  This one converged the fastest by far.  The wave reflects
at the boundary as I'm using dirichlet bcs for now.  I believe this is the best long term approach if I can get rid of these various problems.

![Sod Shock Density](images/Density-discontinuous.png)
![Sod Shock Velocity](images/Velocity-discontinuous.png)

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
## Warning!
Currently this is really slow as I need a batched gradient computation and the [torch implementation](https://pytorch.org/tutorials/prototype/vmap_recipe.html) is still in development and [functorch implementation](https://github.com/pytorch/functorch/issues) seems not to be able to handle my networks right now.  So I'm computing the gradients one element at a time instead of in batch.