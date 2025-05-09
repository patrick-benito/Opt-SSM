"""
This module defines classes for simulating dynamical systems.
"""
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass, asdict
from trunk_sim.simulator import TrunkSimulator


class DynamicalSystem:
    """
    Base class for dynamical systems.
    """

    def __init__(self, n_xf, n_u):
        self.n_xf = n_xf  # number of states
        self.n_u = n_u    # number of controls inputs

    def autonomous_dynamics(self, xf):
        """
        Compute the autonomous dynamics of the system.
        """
        raise NotImplementedError

    def controlled_dynamics(self, xf, u):
        """
        Compute the dynamics of the system given the state and control.
        """
        raise NotImplementedError

    def simulate_autonomous(self, xf0, ts):
        """
        Simulate the autonomous system.
        """
        raise NotImplementedError

    def simulate_controlled(self, xf0, ts, us):
        """
        Simulate the controlled system.
        """
        raise NotImplementedError
    
    def generate_initial_conditions(self, N, rnd_key):
        """
        Generate N initial conditions.
        """
        n_xf = self.n_xf
        keys = random.split(rnd_key, n_xf)
        ic_ranges = getattr(self, 'ic_ranges', [(-1, 1)] * n_xf)
        ic_shifts = self.settled_states
        
        initial_conditions = []
        for i in range(n_xf):
            ic_range = ic_ranges[i]
            ic_shift = ic_shifts[i]
            ic = ic_shift + random.uniform(keys[i], (N,), minval=ic_range[0], maxval=ic_range[1])
            initial_conditions.append(ic)
        
        ICs = jnp.stack(initial_conditions, axis=-1)
        return ICs
    
    def generate_autonomous_trajs(self, N_aut, ts, rnd_key):
        """
        Generate autonomous trajectories.
        """
        aut_trajs = jnp.zeros((N_aut, self.n_xf, len(ts)))
        ICs = self.generate_initial_conditions(N_aut, rnd_key)
        for i in range(N_aut):
            aut_trajs = aut_trajs.at[i].set(self.simulate_autonomous(ICs[i], ts))
        return aut_trajs
    
    def generate_controlled_trajs(self, N_ctrl, ts, u_func, rnd_key):
        """
        Generate controlled trajectories.
        """
        ctrl_trajs = jnp.zeros((N_ctrl, self.n_xf, len(ts)))
        ICs = self.generate_initial_conditions(N_ctrl, rnd_key)
        us = u_func(ts, N_ctrl, rnd_key)
        for i in range(N_ctrl):
            ctrl_trajs = ctrl_trajs.at[i].set(self.simulate_controlled(ICs[i], ts, us[i]))
        return ctrl_trajs
    
    def get_observations(self, trajs):
        """
        Compute the observations of the system given the full state trajectories.
        """
        raise NotImplementedError
    
    def _compute_settled_states(self):
        """
        Compute the settled states.
        """
        raise NotImplementedError
    
    def _set_ic_ranges(self):
        """
        Set the initial condition ranges for the system.
        """
        self.ic_ranges = None


@dataclass
class TrunkConfig:
    num_segments: int = 3
    tip_mass: float = 0.5
    data_folder: str = "../data/trunk"
    duration: int = 10
    render_video: bool = False
    init_steady_state: bool = True
    stop_at_convergence: bool = False  # NOTE: if True, requires padding trajectories


class Trunk():
    """
    Wrapper for the Trunk Mujoco simulator.
    """
    def __init__(self,
                 config=TrunkConfig(),       # configuration for the trunk
                 ):
        self.config = config
        self._extract_config(config)
        self._create_simulator()
        self._compute_settled_positions()

    def simulate_autonomous(self, xf0, t):
        """
        Simulate the autonomous system.
        """
        pass

    def get_positions(self, xf):
        """
        Return the positions of the trunk.
        """
        return xf[-1, :3]

    def step(self, u):
        """
        Step the simulator with the given control input and return the new state.
        """
        # Reshape control input
        control_input = u.reshape((self.num_segments, 2))
        _, _, _, xf_new = self.simulator.step(control_input)
        return xf_new
    
    def reset(self):
        """
        Reset the simulator.
        """
        self.simulator.reset()

    def _extract_config(self, config):
        """
        Dynamically extract the configuration for the pendulum.
        """
        for key, value in asdict(config).items():
            setattr(self, key, value)
    
    def _create_simulator(self):
        """
        Create the trunk simulator.
        """
        self.simulator = TrunkSimulator(
                            num_segments=self.config.num_segments,
                            tip_mass=self.config.tip_mass,
        )
        self.num_links_per_segment = self.simulator.num_links_per_segment
        self.num_links = self.simulator.num_links
        self.num_segments = self.simulator.num_segments

    def _compute_settled_positions(self):
        """
        Compute the settled positions of the trunk.
        """
        self.settled_positions = jnp.array([0.0, 0.0, -0.32])  # TODO: avoid hardcoding of this
