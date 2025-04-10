"""
Custom plotting functions. Choose if you want to use LaTeX for plotting labels via LATEX_LABELS flag.
"""

LATEX_LABEL = False

import re
import jax.numpy as jnp
import matplotlib.pyplot as plt
if LATEX_LABEL:
    plt.rcParams['text.usetex'] = True


def plot_trajectories(trajs, ts):
    """
    Plot trajectories of observations.
    """
    traj_colormap = plt.cm.cool
    traj_linewidth = 1
    colors = [traj_colormap(i / len(trajs)) for i in range(len(trajs))]
    _, axs = plt.subplots(trajs.shape[1], 1, figsize=(6, 4), sharex=True)

    # Determine if the trajectories are 2D or 3D
    is_3d = trajs.shape[1] == 3
    
    if is_3d:
        axs[0].set_ylabel(r'$x_1$')
        axs[1].set_ylabel(r'$x_2$')
        axs[2].set_ylabel(r'$x_3$')
    else:
        axs[0].set_ylabel(r'$x_1$')
        axs[1].set_ylabel(r'$x_2$')

    for i, traj in enumerate(trajs):
        for j in range(trajs.shape[1]):
            axs[j].plot(ts, traj[j, :], color=colors[i], lw=traj_linewidth)
    axs[-1].set_xlabel(r'$t$')
    plt.tight_layout()
    if is_3d:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, traj in enumerate(trajs):
            ax.plot(traj[0, :], traj[1, :], traj[2, :], color=colors[i], lw=traj_linewidth)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(r'$x_3$')
        set_3D_axes_equal(ax)
    else:
        plt.figure(figsize=(4, 4))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        for i, traj in enumerate(trajs):
            plt.plot(traj[0, :], traj[1, :], color=colors[i], lw=traj_linewidth)
        plt.axis('equal')
    plt.tight_layout()


def plot_mpc_trajectory(ts, z_ref, z_mpc, z_true, u_mpc, N, centers=None, radii=None, plot_controls=True, y_up=False, top_down=False):
    """
    Plot the MPC trajectory for 2D or 3D data.
    """
    # Determine the dimension (2D or 3D)
    dim = z_true.shape[1]

    # Set up the plotting environment
    if plot_controls:
        if dim == 2:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        else:
            fig = plt.figure(figsize=(12, 5))
            ax = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2)]
    else:
        if dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax = [ax]
        else:
            fig = plt.figure(figsize=(6, 5))
            ax = [fig.add_subplot(1, 1, 1, projection='3d')]

    # Plot the obstacles
    if centers is not None and radii is not None:
        if dim == 2:
            for pc, rc in zip(centers, radii):
                ax[0].add_patch(
                    plt.Circle((pc[0], pc[1]), rc, color='r', alpha=0.3)
                )
        else:
            # Plot spheres for 3D obstacles
            raise NotImplementedError("3D obstacle plotting not implemented yet.")

    # Plot the MPC trajectories
    for t_idx in range(len(ts) - 2 * N):
        if dim == 2:
            ax[0].plot(z_mpc[t_idx, :, 0], z_mpc[t_idx, :, 1], '--*', color='k', markersize=3,
                       label=latex_label('MPC') if t_idx == 0 else None)
        else:
            ax[0].plot3D(z_mpc[t_idx, :, 0], z_mpc[t_idx, :, 1], z_mpc[t_idx, :, 2], '--*', color='k', markersize=3,
                         label=latex_label('MPC') if t_idx == 0 else None)

    # Plot the true trajectory, start point, and reference
    if dim == 2:
        ax[0].plot(z_true[:-N, 0], z_true[:-N, 1], '-o', label=latex_label('True'), markersize=3)
        ax[0].plot(z_true[0, 0], z_true[0, 1], 'ro', label=latex_label('Start'), markersize=6)
        ax[0].plot(z_ref[:-N, 0], z_ref[:-N, 1], 'y--', label=latex_label('Reference'))
        ax[0].set_xlabel(latex_label('X [m]'))
        ax[0].set_ylabel(latex_label('Y [m]'))
        ax[0].axis('equal')
    else:
        ax[0].plot3D(z_true[:-N, 0], z_true[:-N, 1], z_true[:-N, 2], '-o', label=latex_label('True'), markersize=3)
        ax[0].scatter(z_true[0, 0], z_true[0, 1], z_true[0, 2], color='r', label=latex_label('Start'), s=36)
        ax[0].plot3D(z_ref[:-N, 0], z_ref[:-N, 1], z_ref[:-N, 2], 'y--', label=latex_label('Reference'))
        ax[0].set_xlabel(latex_label('X [m]'))
        ax[0].set_ylabel(latex_label('Y [m]'))
        ax[0].set_zlabel(latex_label('Z [m]'))
        set_3D_axes_equal(ax[0])
        if y_up:
            ax[0].view_init(elev=115, azim=-115, roll=-25)
        elif top_down:
            ax[0].view_init(elev=90, azim=-90, roll=0)
    ax[0].legend()

    # Plot the control inputs
    if plot_controls:
        for u_idx in range(u_mpc.shape[-1]):
            ax[1].plot(ts, u_mpc[:, 0, u_idx], label=latex_label(f'u_{u_idx+1}(t)'))
        ax[1].set_xlabel(latex_label('t [s]'))
        ax[1].set_ylabel(latex_label('U'))
        ax[1].legend()
    plt.tight_layout()


def plot_slow_fast_results(x_true, x_pred_orth, x_pred_opt):
    """
    Plot the results for the slow-fast system.
    """
    pastel_blue = "#6FA3EF"

    # Critical manifold approx. (in this case, the x2 = x1^2 curve)
    x1_critical = jnp.linspace(-0.15, 1.75, 100)
    x2_critical = x1_critical**2

    plt.figure(figsize=(5, 5))
    plt.plot(x1_critical, x2_critical, color='grey', alpha=0.6, linewidth=2, label=r"", zorder=1)
    plt.scatter(x_true[0, 0], x_true[0, 1], color='black', marker='+', linewidth=3, s=100, label=r'', zorder=2)
    plt.plot(x_true[:, 0], x_true[:, 1], color='black', linewidth=3, linestyle='--', label=r"True", zorder=3)
    plt.scatter(x_pred_orth[0, 0], x_pred_orth[1, 0], color='red', marker='+', linewidth=3, s=100, label=r'', zorder=4)
    plt.plot(x_pred_orth[0, :], x_pred_orth[1, :], color='red', linewidth=3, label=r"Orthogonal", zorder=5)
    plt.scatter(x_pred_opt[0, 0], x_pred_opt[1, 0], color=pastel_blue, marker='+', linewidth=3, s=100, label=r'', zorder=6)
    plt.plot(x_pred_opt[0, :], x_pred_opt[1, :], color=pastel_blue, linewidth=3, label=r"Oblique", zorder=7)
    plt.scatter(x_true[-1, 0], x_true[-1, 1], color='black', marker='o', s=60, zorder=8)

    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    x_ticks = jnp.linspace(0, 2, num=3)
    y_ticks = jnp.linspace(0, 2, num=3)
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(y_ticks, fontsize=14)
    plt.legend(fontsize=14)
    plt.gca().set_aspect('equal')


def plot_fluid_results(x_true, x_pred_orth, x_pred_opt):
    """
    Plot the results for the simplified fluid system.
    """
    pastel_blue = "#6FA3EF"

    # Critical manifold approx. (in this case, the x3 = x1^2 + x2^2 curve)
    x1_critical, x2_critical = jnp.linspace(-1.25, 1.25, 100), jnp.linspace(-1.25, 1.25, 100)
    x1_critical, x2_critical = jnp.meshgrid(x1_critical, x2_critical)
    x3_critical = x1_critical**2 + x2_critical**2

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_critical, x2_critical, x3_critical, color='grey', alpha=0.25, label=r"", zorder=1)
    ax.scatter(x_true[0, 0], x_true[0, 1], x_true[0, 2], color='black', marker='+', linewidth=3, s=100, label=r'', zorder=2)
    ax.plot(x_true[:, 0], x_true[:, 1], x_true[:, 2], color='black', linewidth=2, linestyle='--', label=r"True", zorder=3)
    ax.scatter(x_pred_orth[0, 0], x_pred_orth[1, 0], x_pred_orth[2, 0], color='red', marker='+', linewidth=3, s=100, label=r'', zorder=4)
    ax.plot(x_pred_orth[0, :], x_pred_orth[1, :], x_pred_orth[2, :], color='red', linewidth=2, label=r"Orthogonal", zorder=5)
    ax.scatter(x_pred_opt[0, 0], x_pred_opt[1, 0], x_pred_opt[2, 0], color=pastel_blue, marker='+', linewidth=3, s=100, label=r'', zorder=6)
    ax.plot(x_pred_opt[0, :], x_pred_opt[1, :], x_pred_opt[2, :], color=pastel_blue, linewidth=2, label=r"Oblique", zorder=7)
    ax.scatter(0, 0, 0, color='black', marker='o', s=60, label=r'', zorder=8)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_zlabel('$x_3$', fontsize=14, labelpad=-0.5)
    ax.legend(fontsize=12)
    ax.view_init(elev=32.5, azim=-60, roll=0)


def plot_trunk_results(z_ref, z_true_orth, z_true_opt, N):
    """
    Plot the results for the trunk system.
    """
    pastel_blue = "#6FA3EF"

    plt.figure(figsize=(6, 6))
    plt.plot(z_ref[:-N, 0], z_ref[:-N, 1], linestyle='--', linewidth=2, color='black', label='Reference')
    plt.plot(z_true_orth[:-N, 0], z_true_orth[:-N, 1], linestyle='-', linewidth=3, color='red', label='Orthogonal')
    plt.plot(z_true_opt[:-N, 0], z_true_opt[:-N, 1], linestyle='-', linewidth=3, color=pastel_blue, label='Oblique')
    plt.xlabel(r'$x$ [m]', fontsize=14)
    plt.ylabel(r'$y$ [m]', fontsize=14)
    x_ticks = jnp.linspace(-0.2, 0.2, num=5)
    y_ticks = jnp.linspace(-0.2, 0.2, num=5)
    plt.xticks(x_ticks, fontsize=14)
    plt.yticks(y_ticks, fontsize=14)
    plt.axis('equal')
    plt.legend(fontsize=14)


def set_3D_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = (x_limits[0] + x_limits[1]) / 2
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = (y_limits[0] + y_limits[1]) / 2
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = (z_limits[0] + z_limits[1]) / 2

    # The plot bounding box is a sphere in the sense of the infinity norm, hence radius
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def latex_label(text):
    """
    Converts regular text to a LaTeX-compatible label, handling LaTeX math mode correctly.
    """
    # Split text into math ($...$) and non-math segments
    parts = re.split(r'(\$.+?\$)', text)
    formatted_parts = []

    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            # It's a math mode segment, add it as is
            formatted_parts.append(part)
        else:
            # Replace spaces with ~ and wrap non-math text with \mathrm{}
            clean_fragment = part.replace(' ', '~')
            if clean_fragment:
                latex_command = r"\mathrm{" + clean_fragment + "}"
                formatted_parts.append(f'${latex_command}$')
                # formatted_parts.append(f'${r"\mathrm{" + clean_fragment + "}"}$')

    return ''.join(formatted_parts)
