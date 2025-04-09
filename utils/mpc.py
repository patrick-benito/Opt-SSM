"""
General Model Predictive Control (MPC) functions.
"""

import jax
import jax.numpy as jnp
from time import time
from tqdm.auto import tqdm
from .misc import RK4_step
from .gusto import GuSTO


def generate_ref_trajectory(t, type='circle', T=2.5, A=0.1, dim=2):
    """
    Generate a reference trajectory to be followed by the positions of interest.
    """
    if dim == 2:
        z_ref = jnp.zeros((len(t), 2))
        if type == 'circle':
            z_ref = z_ref.at[:, 0].set(1.0 * jnp.cos(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(0.2 + 0.5 * jnp.sin(2 * jnp.pi / T * t))
        elif type == 'figure_eight':
            z_ref = z_ref.at[:, 0].set(0.6 * jnp.sin(jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(0.3 + 0.15 * jnp.sin(2 * jnp.pi / T * t))
    elif dim == 3:
        z_ref = jnp.zeros((len(t), 3))
        if type == 'circle':
            z_ref = z_ref.at[:, 0].set(A * jnp.cos(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(A * jnp.sin(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 2].set(0.0 * jnp.ones_like(t))
        elif type == 'figure_eight':
            z_ref = z_ref.at[:, 0].set(A * jnp.sin(2 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 1].set(A * jnp.sin(4 * jnp.pi / T * t))
            z_ref = z_ref.at[:, 2].set(0.0 * jnp.ones_like(t))
    return z_ref


def run_mpc(system, model, config, z_ref, xf0, U, dU, N_exec = 1, adaptive=False, smoothing_func=None):
    """
    Run MPC and simulate dynamical system.
    """
    # n_xf = system.n_xf                            # full state dimension
    n_x = model.n_x                                 # state dimension
    n_u = model.n_u                                 # control dimension
    n_z = model.n_z                                 # performance variable dimension
    dt = config.dt                                  # time step
    N = config.N                                    # optimization horizon
    N_obs_delay = model.ssm.N_obs_delay             # observed delays

    # Simulate the full system for single timestep to get the initial delay embedding
    # t_sim = jnp.linspace(0, dt, 2)
    # xf_sim = system.simulate_autonomous(xf0, t_sim)
    # y_sim = system.get_positions(xf_sim)[-1, :, :] - system.settled_positions[-1, :, jnp.newaxis]
    system.reset()
    
    # y0 = jnp.flip(y_sim, 1).T.flatten()
    y0 = jnp.zeros((N_obs_delay + 1) * n_z)
    x0 = model.encode(y0)
    x = jnp.copy(x0)
    # xf = jnp.copy(xf_sim[:, -1])
    
    x_mpc = jnp.zeros((len(z_ref), N+1, n_x))
    z_mpc = jnp.zeros((len(z_ref), N+1, n_z))
    u_mpc = jnp.zeros((len(z_ref), N, n_u))
    z_true = jnp.zeros((len(z_ref), n_z))
    # xf_true = jnp.zeros((len(z_ref), n_xf))
    
    # Save estimated parameters over time if applicable
    if hasattr(model, 'update_parameter_estimate') and adaptive:
        est_params = jnp.ones(len(z_ref) // model.update_interval) * model.est_param
        covs_params = jnp.ones(len(z_ref) // model.update_interval) * model.param_cov
    else:
        est_params = None
        covs_params = None

    # Store the initial conditions
    # x_mpc = x_mpc.at[0].set(jnp.tile(x0, (N+1, 1)))
    # z_true = z_true.at[:2].set(y_sim.T)
    # z_mpc = z_mpc.at[0].set(jnp.tile(y_sim[:, 0], (N+1, 1)))
    y = jnp.copy(y0)
    # xf_true = xf_true.at[:2].set(xf_sim.T)

    total_time = time()
    total_control_cost = 0.
    
    # Set up GuSTO and run first solve with a simple initial guess
    u_init = jnp.zeros((N, model.n_u))
    x_init = model.rollout(x0, u_init, dt)
    gusto = GuSTO(model, config, x0, u_init, x_init, z=z_ref[0:N+1], zf=z_ref[N], U=U, dU=dU, solver="GUROBI")
    x_opt, u_opt, z_opt, _ = gusto.get_solution()

    for t_idx in tqdm(range(1, len(z_ref) - N)):
        # Re-plan only if we're at the start of an N_exec interval
        if (t_idx - 1) % N_exec == 0:
            # Get the next reference positions over the MPC window
            z_ref_win = z_ref[t_idx:t_idx + N + 1]

            # Update LOCP parameter with the previously applied control
            gusto.locp.u0_prev.value = u_opt[N_exec-1].tolist()

            # Solve the MPC problem at the current time step
            gusto.solve(x, u_init, x_init, z=z_ref_win, zf=z_ref_win[-1])
            x_opt, u_opt, z_opt, _ = gusto.get_solution()

            # Use this solution to warm-start the next iteration
            x_future = x_opt[N_exec:]
            x_new = x_future[-1]
            for _ in range(N_exec):
                x_new = model.discrete_dynamics(x_new, u_opt[-1])
                x_future = jnp.concatenate([x_future, x_new.reshape((1, -1))])
            x_init = x_future
            # We just duplicate the last control N_exec times
            u_init = jnp.concatenate([u_opt[N_exec:], jnp.tile(u_opt[-1:], (N_exec, 1))])

        # Store the MPC solution / prediction
        x_mpc = x_mpc.at[t_idx].set(x_opt)
        z_mpc = z_mpc.at[t_idx].set(z_opt)
        u_mpc = u_mpc.at[t_idx].set(u_opt)

        # Propagate the full state in time with the closed-loop MPC input
        # xf_new = RK4_step(system.controlled_dynamics, xf, u, dt=dt)
        
        xf_new = system.step(u_opt[0])
        y_new = system.get_positions(xf_new) - system.settled_positions
        # y_new = system.get_positions(xf_new)[-1, :] - system.settled_positions[-1, :]
        y = jnp.concatenate([y_new, y[:-n_z]])
        if smoothing_func is not None:
            y = smoothing_func(y.reshape(2, n_z).T).T.flatten()  # TODO: the 2 needs updating

        # xf = xf_new
        # xf_true = xf_true.at[t_idx + 1].set(xf_new)

        # Store the true positions
        z_true = z_true.at[t_idx + 1].set(y_new)

        # Update parametric model if necessary
        if est_params is not None:
            if t_idx % model.update_interval == 0 and t_idx > 0:
                x0_interval = x_mpc[t_idx-model.update_interval, 0]
                us_interval = u_mpc[t_idx-model.update_interval:t_idx, 0]
                zs_interval = z_true[t_idx-model.update_interval:t_idx]
                model.update_parameter_estimate(x0_interval, us_interval, zs_interval)
                model.update_model()
                est_params = est_params.at[t_idx//model.update_interval].set(model.est_param)
                covs_params = covs_params.at[t_idx//model.update_interval].set(model.param_cov)

        # Update the reduced state with the new observation
        x = model.encode(y)

        # Accumulate the actual control cost
        total_control_cost += u_mpc[t_idx, 0].T @ config.R @ u_mpc[t_idx, 0]

    total_time = time() - total_time
    print('Total elapsed time:', total_time, 'seconds')
    print('Total control cost:', total_control_cost)

    # For now no full state is returned
    xf_true = None

    return x_mpc, z_mpc, u_mpc, z_true, xf_true, est_params, covs_params
