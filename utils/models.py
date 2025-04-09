"""
Reduced order models of controlled systems.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from utils.ssm import DelaySSM, OptSSM
from utils.residual import ResidualBr
from utils.misc import trajectories_delay_embedding, trajectories_derivatives, RK4_step


class ReducedOrderModel:
    """
    Base class for reduced order models.
    """
    def __init__(self, n_x, n_u, n_y, n_z):
        self.n_x = n_x
        self.n_u = n_u
        self.n_y = n_y
        self.n_z = n_z

    def continuous_dynamics(self, x, u):
        """
        Continuous dynamics of the system.
        """
        raise NotImplementedError

    def discrete_dynamics(self, x, u, dt=0.01):
        """
        Discrete-time dynamics of the system.
        """
        raise NotImplementedError

    def rollout(self, x0, u, dt=0.01):
        """
        Rollout of the model with a given control sequence at an initial condition.
        """
        raise NotImplementedError

    def performance_mapping(self, x):
        """
        Performance mapping maps the state, x, to the performance output, z.
        """
        raise NotImplementedError
    
    @property
    def H(self):
        """
        Linear transformation from the state, x, to the performance variable, z.
        """
        raise NotImplementedError


class SSMR(ReducedOrderModel):
    """
    SSMR model combining a SSM model with a residual dynamics model.
    """
    def __init__(self, ssm=None, residual_dynamics=None, obs_perf_matrix=None, model_path=None, model_type='delay_ssm'):
        if model_path is not None:
            model_data = np.load(model_path)
            if model_type == 'delay_ssm':
                ssm = DelaySSM(model_data=model_data)
            elif model_type == 'opt_ssm':
                ssm = OptSSM(model_data=model_data)
            residual_dynamics = ResidualBr(model_data=model_data)
            obs_perf_matrix = model_data['obs_perf_matrix']
        n_x = ssm.SSMDim
        n_u = residual_dynamics.n_u
        n_z, n_y = obs_perf_matrix.shape
        
        super().__init__(n_x, n_u, n_y, n_z)

        # Autonomous dynamics model
        self.ssm = ssm

        # Residual dynamics model
        self.residual_dynamics = residual_dynamics
        
        # Observation-performance matrix maps the observations, y, to the performance variable, z
        self.obs_perf_matrix = obs_perf_matrix

    def continuous_dynamics(self, x, u):
        """
        Continuous dynamics of reduced system.
        """
        return self.ssm.reduced_dynamics(x) + self.residual_dynamics(x, u)

    @partial(jax.jit, static_argnums=(0,))
    def discrete_dynamics(self, x, u, dt=0.01):
        """
        Discrete-time dynamics of reduced system using RK4 integration.
        """
        return RK4_step(self.continuous_dynamics, x, u, dt)

    def dynamics_step(self, x, u_dt):
        """
        Perform a single step of the reduced dynamics.
        """
        u, dt = u_dt[:-1], u_dt[-1]
        return self.discrete_dynamics(x, u, dt), x

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, x0, u, dt=0.01):
        """
        Rollout of the discrete-time dynamics model, with u being an array of length N.
        Note that if u has length N, then the output will have length N+1.
        """
        u_dt = jnp.column_stack([u, jnp.full(u.shape[0], dt)])  # shape of u is (N, n_u)
        final_state, xs = jax.lax.scan(self.dynamics_step, x0, u_dt)  # TODO: use lambda function instead of u_dt, if performance the same
        return jnp.vstack([xs, final_state])

    def performance_mapping(self, x):
        """
        Performance mapping maps the state, x, to the performance output, z, through
        z = C @ y = C @ w(x).
        """
        return self.obs_perf_matrix @ self.ssm.decode(x)
    
    @property
    def H(self):
        """
        Linear mapping from the state, x, to the performance variable, z.
        """
        raise AttributeError("SSMR uses a nonlinear performance mapping, hence H is not defined.")
    
    def encode(self, y):
        """
        Encode the observations, y, into the reduced state, x.
        """
        return self.ssm.encode(y)
    
    def decode(self, x):
        """
        Decode the reduced state, x, into the observations, y.
        """
        return self.ssm.decode(x)
    
    def save_model(self, path):
        """
        Save the SSMR model to a file.
        """
        np.savez(path,
                 dynamics_coeff = self.ssm.dynamics_coeff,
                 dynamics_exp = self.ssm.dynamics_exp,
                 encoder_coeff = self.ssm.encoder_coeff,
                 encoder_exp = self.ssm.encoder_exp,
                 decoder_coeff = self.ssm.decoder_coeff,
                 decoder_exp = self.ssm.decoder_exp,
                 B_r_coeff = self.residual_dynamics.learned_B_r.B_r_coeff,
                 obs_perf_matrix = self.obs_perf_matrix)


def get_residual_labels(ssm, trajs, ts, u_func=None, rnd_key=jax.random.PRNGKey(0), us=None):
    """
    Get labels for B_r learning.
    """
    # Either provide the control inputs or the control function
    if us is None and u_func is None:
        raise ValueError("Either control inputs or control function must be provided.")

    N_trajs = len(trajs)
    ys = trajectories_delay_embedding(trajs, ssm.N_obs_delay)
    x_trajs = []
    for traj in ys:
        x_traj = ssm.encode(traj)
        # Apply padding of zeros to the end of the trajectory
        x_traj = x_traj.at[:, -(ssm.N_obs_delay):].set(jnp.zeros((ssm.SSMDim, ssm.N_obs_delay)))
        x_trajs.append(x_traj)
    x_trajs = jnp.array(x_trajs)

    x_dots_ctrl = trajectories_derivatives(x_trajs, ts)
    x_dots_aut = []
    for traj in x_trajs:
        x_dot_aut = ssm.reduced_dynamics(traj)
        x_dots_aut.append(x_dot_aut)
    x_dots_aut = jnp.array(x_dots_aut)

    delta_x_dots = x_dots_ctrl - x_dots_aut
    delta_x_dots_flat = delta_x_dots.transpose(0, 2, 1).reshape(-1, ssm.SSMDim)
    xs_flat = x_trajs.transpose(0, 2, 1).reshape(-1, ssm.SSMDim)
    us = u_func(ts, N_trajs, rnd_key) if us is None else us  # shape of us is (N_trajs, n_u, len(ts))
    us_flat = us.transpose(0, 2, 1).reshape(-1, us.shape[1])
    return xs_flat, us_flat, delta_x_dots_flat


def generate_ssmr_predictions(ssmr, trajs, ts, u_func=None, rnd_key=None, us=None):
    """
    Generate tip positions as predicted by SSMR model for entire trajectories.
    """
    # Either provide the control inputs or the control function
    if us is None and u_func is None:
        raise ValueError("Either control inputs or control function must be provided.")

    N_trajs = len(trajs)
    us = u_func(ts, N_trajs, rnd_key) if us is None else us
    N_input_states = trajs.shape[1]
    ssmr_predictions = jnp.zeros_like(trajs)
    N_obs_delay = ssmr.ssm.N_obs_delay
    for i, traj in enumerate(trajs):
        # Assume first (N_obs_delay + 1) observations are known
        ssmr_predictions = ssmr_predictions.at[i, :, :N_obs_delay+1].set(traj[:, :N_obs_delay+1])
        y0 = jnp.flip(traj[:, :N_obs_delay+1], 1).T.flatten()
        x0 = ssmr.ssm.encode(y0)
        xs = ssmr.rollout(x0, us[i, :, N_obs_delay+1:].T)[:-1].T  # exclude the last, (N+1)th, state 
        ys = ssmr.ssm.decode(xs)
        ssmr_predictions = ssmr_predictions.at[i, :, N_obs_delay+1:].set(ys[:N_input_states, :])  # select the non-delayed predictions
    return ssmr_predictions
