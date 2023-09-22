# uav_control
This repository implements simulation of a quadrotor with different controllers and observers.

The inner-loop subsystem is controlled by a Dynamic Sliding Mode Controller (DSMC).
The outer-loop subsystem is controller by a Fast Nonsingular Terminal SMC (FNTSMC).

Disturbances and uncertainties are all considered.
Four kinds of observers are tested.
    1. Appointed Fixed Time Observer (AFTO)
    2. High-order Sliding Mode Observer (HSMO)
    3. Nonlinear Extended State Observer (NESO)
    4. Romberg Observer (RO) 
