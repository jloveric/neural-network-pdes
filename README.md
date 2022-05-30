# neural-network-pdes
Neural Network Implicit Representation of Partial Differential Equations.  The problems here are solved using a simple high order MLP where
the the input is (x,t) in 1D and the output is density, velocity and pressure.  The loss function is partial differential equation for the 1d euler equations of gas dynamics dotted with itself.  The resulting model contains the entire solution at every time point and every space point between start and end.

## Euler equations

$$ 
\frac{\partial}{\partial t} \left[ \begin{array}{c} 
\rho \\
\rho v \\
e \\
\end{array}\right]+
\frac{\partial}{\partial x} \left[
\begin{array}{c}
\rho v \\
\rho v^{2}+p \\
\left(e+p\right)v
\end{array}\right]
=r
$$

where r=0.  The PDE is used as the loss function and r is the residual so that $loss=r\cdot r$.


## Sod Shock

"Solutions" to the Sod Shock problem in fluid dynamics, which is 1 dimensional ideal compressible gas dynamics (euler equations), a very classic problem.  Solutions need to be much better (and faster) before I try and extend to much more complicated systems. I'll try and improve on this as I have time.

## Fourier layers
The solution at t=0 should be a step function.  The model need to learn the initial conditions as well as all
the boundary conditions and the interior solution.  You can see here the initial condition is not perfect.

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
Currently this is really slow as I need a batched gradient computation and the [functorch implementation](https://github.com/pytorch/functorch/issues) seems not to be able to handle my networks right now.  So I'm computing the gradients one element at a time instead of in batch.