#!/usr/bin/env python3
"""
Cart-transporter trajectory visualizer.

Reads a saved trajectory file and writes both an animation and static summaries.

The drawing shows:
- Cart (blue rectangle) moving along the track
- Cargo (orange rectangle) on top of the cart
- Friction force between cargo and cart
- Control force applied to cart
- Relative sliding between cargo and cart
"""

import argparse
import os as _os
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))
from visual_utils import resolve_output_dir, resolve_trajectory

import numpy as np
import matplotlib as _mpl
if not _os.environ.get("DISPLAY"):
    _mpl.use("Agg")  # headless-safe: write figures without a display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib.patheffects as path_effects


def visualize_solution_from_file(trajectory_file, l=1.0, x_goal=None, dt=0.02,
                                 minimal_overlay=False, frame_interval=10, output_dir=None):
    """
    Visualize a cart-transporter trajectory from a solver output file.
    
    Args:
        trajectory_file: Path to trajectory file
        l: Gap length (cargo must stay within |x1 - x2| <= l)
        x_goal: Goal state [x1, x2, x1_dot, x2_dot] (optional)
        dt: Time step for animation speed
        minimal_overlay: If True, hide axes, legend, and title in animation
        frame_interval: Frame interval for overlay sequence (every N frames)
        output_dir: Directory for generated images and animation
    """
    
    # Cart and cargo dimensions for visualization
    cart_width = 1.2
    cart_height = 0.4
    cargo_width = 0.6
    cargo_height = 0.4
    
    # Read trajectory file
    with open(trajectory_file, 'r') as f:
        lines = f.readlines()
    
    # Find state and control trajectory sections
    state_start = None
    control_start = None
    for i, line in enumerate(lines):
        if line.startswith('# State Trajectory'):
            state_start = i + 1
        elif line.startswith('# Control Trajectory'):
            control_start = i + 1
            break
    
    # Parse state trajectory (x1, x2, x1_dot, x2_dot)
    state_lines = []
    for i in range(state_start, control_start - 2):
        if lines[i].strip() and not lines[i].startswith('#'):
            state_lines.append(lines[i])
    
    x_data = []
    for line in state_lines:
        values = [float(v) for v in line.strip().split()]
        x_data.append(values)
    x = np.array(x_data)  # Shape: (timesteps, 4) - [x1, x2, x1_dot, x2_dot]
    
    # Parse control trajectory (f, u, v, w)
    control_lines = []
    for i in range(control_start, len(lines)):
        if lines[i].strip() and not lines[i].startswith('#'):
            control_lines.append(lines[i])
    
    u_data = []
    for line in control_lines:
        values = [float(v) for v in line.strip().split()]
        u_data.append(values)
    u = np.array(u_data)  # Shape: (timesteps, 4) - [f, u, v, w]
    
    print(f"Loaded trajectory: {x.shape[0]} timesteps, state dim={x.shape[1]}, control dim={u.shape[1]}")
    
    # Derived quantities used in plots and labels.
    x1 = x[:, 0]  # Cargo position
    x2 = x[:, 1]  # Cart position
    x1_dot = x[:, 2]  # Cargo velocity
    x2_dot = x[:, 3]  # Cart velocity
    relative_pos = x1 - x2  # Relative position of cargo on cart
    relative_vel = x1_dot - x2_dot  # Relative velocity (sliding)
    
    friction_force = u[:, 0]  # Friction force
    control_force = u[:, 1]  # Control force on cart
    
    print(f"Position range: cart [{x2.min():.2f}, {x2.max():.2f}], cargo [{x1.min():.2f}, {x1.max():.2f}]")
    print(f"Relative position range: [{relative_pos.min():.3f}, {relative_pos.max():.3f}] (limit: ±{l})")
    
    # Use a plain light theme for saved figures.
    plt.style.use('default')
    
    # Main figure layout.
    if minimal_overlay:
        # Compact figure for minimal overlay mode.
        fig = plt.figure(figsize=(14, 6), facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax_pos = ax_vel = ax_force = ax_rel = ax_info = ax_state = None
        
        # Draw the track (ground line) for minimal mode
        track_min = min(x1.min(), x2.min()) - 2
        track_max = max(x1.max(), x2.max()) + 2
        ax.axhline(y=0, color='#555555', linewidth=3, zorder=1)
        ax.fill_between([track_min, track_max], [-0.15, -0.15], [0, 0], 
                        color='#dddddd', alpha=0.8, zorder=0)
    else:
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
        # Main visualization axis (top section)
        ax = fig.add_subplot(gs[0:2, :3])
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5, color='#cccccc')
        ax.set_facecolor('#f0f5f0')
        
        # Position plot
        ax_pos = fig.add_subplot(gs[2, :2])
        ax_pos.grid(True, alpha=0.3)
        ax_pos.set_facecolor('#f8f8f8')
        
        # Velocity plot
        ax_vel = fig.add_subplot(gs[2, 2:])
        ax_vel.grid(True, alpha=0.3)
        ax_vel.set_facecolor('#f8f8f8')
        
        # Force plot
        ax_force = fig.add_subplot(gs[3, :2])
        ax_force.grid(True, alpha=0.3)
        ax_force.set_facecolor('#f8f8f8')
        
        # Relative motion plot
        ax_rel = fig.add_subplot(gs[3, 2:])
        ax_rel.grid(True, alpha=0.3)
        ax_rel.set_facecolor('#f8f8f8')
        
        # Info panels (right side)
        ax_info = fig.add_subplot(gs[0, 3])
        ax_info.axis('off')
    
        ax_state = fig.add_subplot(gs[1, 3])
        ax_state.axis('off')

        # Draw the track (ground line)
        track_min = min(x1.min(), x2.min()) - 2
        track_max = max(x1.max(), x2.max()) + 2
        ax.axhline(y=0, color='#555555', linewidth=3, zorder=1)
        ax.fill_between([track_min, track_max], [-0.15, -0.15], [0, 0], 
                        color='#dddddd', alpha=0.8, zorder=0)
    
    # Track bounds are reused by both figure modes.
    if minimal_overlay:
        pass  # Already set above
    else:
        pass  # Already set above
    track_min = min(x1.min(), x2.min()) - 2
    track_max = max(x1.max(), x2.max()) + 2
    
    # Initial positions.
    x2_0 = float(x2[0])
    x1_0 = float(x1[0])
    
    # Cart patch.
    cart = mpatches.FancyBboxPatch(
        (x2_0 - cart_width/2, 0), cart_width, cart_height,
        boxstyle="round,pad=0.02",
        facecolor='#4da6ff', edgecolor='#0066cc', linewidth=2.5,
        alpha=0.9, zorder=5, label='Cart'
    )
    ax.add_patch(cart)
    
    # Cart wheels
    wheel_radius = 0.08
    wheel1 = plt.Circle((x2_0 - cart_width/3, 0), wheel_radius, 
                        color='#333333', zorder=6)
    wheel2 = plt.Circle((x2_0 + cart_width/3, 0), wheel_radius, 
                        color='#333333', zorder=6)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    
    # Cargo patch.
    cargo = mpatches.FancyBboxPatch(
        (x1_0 - cargo_width/2, cart_height), cargo_width, cargo_height,
        boxstyle="round,pad=0.02",
        facecolor='#ffaa44', edgecolor='#cc6600', linewidth=2.5,
        alpha=0.9, zorder=7, label='Cargo'
    )
    ax.add_patch(cargo)
    
    # Force arrows for friction and control.
    force_scale = 0.05  # Arrow length per unit force
    
    # Friction force arrow (between cargo and cart)
    friction_arrow = ax.annotate('', xy=(x1_0, cart_height + cargo_height/2),
                                  xytext=(x1_0 - friction_force[0]*force_scale, cart_height + cargo_height/2),
                                  arrowprops=dict(arrowstyle='->', color='#cc0066', lw=2.5),
                                  zorder=10)
    
    # Control force arrow (on cart)
    control_arrow = ax.annotate('', xy=(x2_0 + cart_width/2, cart_height/2),
                                 xytext=(x2_0 + cart_width/2 + control_force[0]*force_scale, cart_height/2),
                                 arrowprops=dict(arrowstyle='->', color='#009900', lw=3),
                                 zorder=10)
    
    # Goal position markers.
    if x_goal is not None:
        x1_goal, x2_goal = x_goal[0], x_goal[1]
        # Goal cart outline
        goal_cart = mpatches.FancyBboxPatch(
            (x2_goal - cart_width/2, 0), cart_width, cart_height,
            boxstyle="round,pad=0.02",
            facecolor='none', edgecolor='#00cc00', linewidth=2,
            linestyle='--', alpha=0.7, zorder=3, label='Goal Cart'
        )
        ax.add_patch(goal_cart)
        # Goal cargo outline
        goal_cargo = mpatches.FancyBboxPatch(
            (x1_goal - cargo_width/2, cart_height), cargo_width, cargo_height,
            boxstyle="round,pad=0.02",
            facecolor='none', edgecolor='#00cc00', linewidth=2,
            linestyle='--', alpha=0.7, zorder=3, label='Goal Cargo'
        )
        ax.add_patch(goal_cargo)
    
    # Set axis limits
    margin = 1.5
    ax.set_xlim(track_min, track_max)
    ax.set_ylim(-0.5, cart_height + cargo_height + 1)
    
    # Dynamically adjust figure size for minimal mode
    if minimal_overlay:
        data_width = track_max - track_min
        data_height = cart_height + cargo_height + 1.5
        aspect = data_width / data_height if data_height > 0 else 1.0
        fig_height = 6
        fig_width = fig_height * aspect
        fig.set_size_inches(fig_width, fig_height)
    
    # Plot trajectory on position/velocity plots (static) - only if not minimal
    time = np.arange(len(x)) * dt
    
    if not minimal_overlay:
        # Position plot
        ax_pos.plot(time, x1, 'orange', linewidth=2, label='Cargo (x1)', alpha=0.3)
        ax_pos.plot(time, x2, 'blue', linewidth=2, label='Cart (x2)', alpha=0.3)
        pos_line1, = ax_pos.plot([], [], 'o', color='orange', markersize=8)
        pos_line2, = ax_pos.plot([], [], 'o', color='blue', markersize=8)
        pos_vline = ax_pos.axvline(x=0, color='red', linewidth=1.5, alpha=0.7)
        if x_goal is not None:
            ax_pos.axhline(y=x_goal[0], color='orange', linestyle='--', alpha=0.5)
            ax_pos.axhline(y=x_goal[1], color='blue', linestyle='--', alpha=0.5)
        ax_pos.set_xlabel('Time [s]', fontsize=10)
        ax_pos.set_ylabel('Position [m]', fontsize=10)
        ax_pos.set_title('Position vs Time', fontsize=11, fontweight='bold')
        ax_pos.legend(loc='upper right', fontsize=9)
        
        # Velocity plot
        ax_vel.plot(time, x1_dot, 'orange', linewidth=2, label='Cargo vel', alpha=0.3)
        ax_vel.plot(time, x2_dot, 'blue', linewidth=2, label='Cart vel', alpha=0.3)
        vel_line1, = ax_vel.plot([], [], 'o', color='orange', markersize=8)
        vel_line2, = ax_vel.plot([], [], 'o', color='blue', markersize=8)
        vel_vline = ax_vel.axvline(x=0, color='red', linewidth=1.5, alpha=0.7)
        if x_goal is not None:
            ax_vel.axhline(y=x_goal[2], color='orange', linestyle='--', alpha=0.5)
            ax_vel.axhline(y=x_goal[3], color='blue', linestyle='--', alpha=0.5)
        ax_vel.set_xlabel('Time [s]', fontsize=10)
        ax_vel.set_ylabel('Velocity [m/s]', fontsize=10)
        ax_vel.set_title('Velocity vs Time', fontsize=11, fontweight='bold')
        ax_vel.legend(loc='upper right', fontsize=9)
        
        # Force plot (control has one fewer timestep than state)
        time_u = np.arange(len(u)) * dt
        ax_force.plot(time_u, friction_force, '#cc0066', linewidth=2, label='Friction (f)', alpha=0.3)
        ax_force.plot(time_u, control_force, '#009900', linewidth=2, label='Control (u)', alpha=0.3)
        force_line1, = ax_force.plot([], [], 'o', color='#cc0066', markersize=8)
        force_line2, = ax_force.plot([], [], 'o', color='#009900', markersize=8)
        force_vline = ax_force.axvline(x=0, color='red', linewidth=1.5, alpha=0.7)
        ax_force.set_xlabel('Time [s]', fontsize=10)
        ax_force.set_ylabel('Force [N]', fontsize=10)
        ax_force.set_title('Forces vs Time', fontsize=11, fontweight='bold')
        ax_force.legend(loc='upper right', fontsize=9)
        
        # Relative motion plot
        ax_rel.plot(time, relative_pos, 'purple', linewidth=2, label='Rel. Position', alpha=0.3)
        ax_rel.plot(time, relative_vel, 'green', linewidth=2, label='Rel. Velocity', alpha=0.3)
        rel_line1, = ax_rel.plot([], [], 'o', color='purple', markersize=8)
        rel_line2, = ax_rel.plot([], [], 'o', color='green', markersize=8)
        rel_vline = ax_rel.axvline(x=0, color='red', linewidth=1.5, alpha=0.7)
        ax_rel.axhline(y=l, color='red', linestyle='--', alpha=0.5, label=f'Gap limit (±{l})')
        ax_rel.axhline(y=-l, color='red', linestyle='--', alpha=0.5)
        ax_rel.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_rel.set_xlabel('Time [s]', fontsize=10)
        ax_rel.set_ylabel('Relative Motion', fontsize=10)
        ax_rel.set_title('Cargo-Cart Relative Motion', fontsize=11, fontweight='bold')
        ax_rel.legend(loc='upper right', fontsize=9)
        
        # Info panels.
        def create_info_panel(ax_panel, title, color='#00d4ff'):
            """Create one boxed info panel."""
            bbox = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, 
                                 boxstyle="round,pad=0.05", 
                                 transform=ax_panel.transAxes,
                                 facecolor='#f0f0f0', 
                                 edgecolor=color, 
                                 linewidth=2.5, 
                                 alpha=0.9)
            ax_panel.add_patch(bbox)
            ax_panel.text(0.5, 0.92, title, transform=ax_panel.transAxes,
                         fontsize=11, fontweight='bold', color=color,
                         ha='center', va='top')
        
        create_info_panel(ax_info, 'SYSTEM INFO', '#0099cc')
        create_info_panel(ax_state, 'STATE DATA', '#009966')
        
        # Info text elements
        info_texts = {
            'time': ax_info.text(0.5, 0.75, '', transform=ax_info.transAxes, 
                                fontsize=10, ha='center', color='#333333', family='monospace'),
            'step': ax_info.text(0.5, 0.60, '', transform=ax_info.transAxes,
                                fontsize=10, ha='center', color='#333333', family='monospace'),
            'progress': ax_info.text(0.5, 0.45, '', transform=ax_info.transAxes,
                                    fontsize=10, ha='center', color='#333333', family='monospace'),
            'status': ax_info.text(0.5, 0.25, '● RUNNING', transform=ax_info.transAxes,
                                  fontsize=11, ha='center', color='#009900', fontweight='bold'),
            
            'pos_cargo': ax_state.text(0.5, 0.78, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#cc6600', family='monospace'),
            'pos_cart': ax_state.text(0.5, 0.62, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#0066cc', family='monospace'),
            'vel_cargo': ax_state.text(0.5, 0.46, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#cc6600', family='monospace'),
            'vel_cart': ax_state.text(0.5, 0.30, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#0066cc', family='monospace'),
            'rel_pos': ax_state.text(0.5, 0.14, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#990099', family='monospace'),
        }
    else:
        info_texts = None
        pos_line1 = pos_line2 = pos_vline = None
        vel_line1 = vel_line2 = vel_vline = None
        force_line1 = force_line2 = force_vline = None
        rel_line1 = rel_line2 = rel_vline = None

    # Animation callback.
    def update(i):
        t_curr = i * dt
        
        # Get current state
        x1_i = float(x[i, 0])
        x2_i = float(x[i, 1])
        x1_dot_i = float(x[i, 2])
        x2_dot_i = float(x[i, 3])
        
        # Update cart position
        cart.set_x(x2_i - cart_width/2)
        wheel1.set_center((x2_i - cart_width/3, 0))
        wheel2.set_center((x2_i + cart_width/3, 0))
        
        # Update cargo position
        cargo.set_x(x1_i - cargo_width/2)
        
        # Update force arrows
        if i < len(u):
            f_i = float(u[i, 0])
            u_i = float(u[i, 1])
        else:
            f_i = float(u[-1, 0])
            u_i = float(u[-1, 1])
        
        # Remove old arrows and create new ones
        friction_arrow.xy = (x1_i + f_i*force_scale, cart_height + cargo_height/2)
        friction_arrow.set_position((x1_i, cart_height + cargo_height/2))
        
        control_arrow.xy = (x2_i + cart_width/2 + u_i*force_scale*0.5, cart_height/2)
        control_arrow.set_position((x2_i + cart_width/2, cart_height/2))
        
        # Update plot markers (only if not minimal)
        if not minimal_overlay:
            pos_line1.set_data([t_curr], [x1_i])
            pos_line2.set_data([t_curr], [x2_i])
            pos_vline.set_xdata([t_curr])
            
            vel_line1.set_data([t_curr], [x1_dot_i])
            vel_line2.set_data([t_curr], [x2_dot_i])
            vel_vline.set_xdata([t_curr])
            
            if i < len(u):
                force_line1.set_data([t_curr], [f_i])
                force_line2.set_data([t_curr], [u_i])
            force_vline.set_xdata([t_curr])
            
            rel_pos_i = x1_i - x2_i
            rel_vel_i = x1_dot_i - x2_dot_i
            rel_line1.set_data([t_curr], [rel_pos_i])
            rel_line2.set_data([t_curr], [rel_vel_i])
            rel_vline.set_xdata([t_curr])
            
            # Update info texts
            progress = (i / (len(x) - 1)) * 100
            info_texts['time'].set_text(f'Time: {t_curr:6.2f}s')
            info_texts['step'].set_text(f'Step: {i:4d}/{len(x)-1}')
            info_texts['progress'].set_text(f'Progress: {progress:5.1f}%')
            
            info_texts['pos_cargo'].set_text(f'Cargo x1: {x1_i:+.3f} m')
            info_texts['pos_cart'].set_text(f'Cart  x2: {x2_i:+.3f} m')
            info_texts['vel_cargo'].set_text(f'Cargo v1: {x1_dot_i:+.3f} m/s')
            info_texts['vel_cart'].set_text(f'Cart  v2: {x2_dot_i:+.3f} m/s')
            info_texts['rel_pos'].set_text(f'Rel pos: {rel_pos_i:+.3f} m')
            
            # Update status color based on constraint satisfaction
            if abs(rel_pos_i) > l * 0.95:
                info_texts['status'].set_text('● NEAR LIMIT')
                info_texts['status'].set_color('#cc6600')
            else:
                info_texts['status'].set_text('● RUNNING')
                info_texts['status'].set_color('#009900')

            return [cart, cargo, wheel1, wheel2, friction_arrow, control_arrow,
                    pos_line1, pos_line2, pos_vline, vel_line1, vel_line2, vel_vline,
                    force_line1, force_line2, force_vline, rel_line1, rel_line2, rel_vline] + list(info_texts.values())
        else:
            return [cart, cargo, wheel1, wheel2, friction_arrow, control_arrow]

    # Animate
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=len(x), interval=max(20, int(dt*500)),
                                   blit=False, repeat=True)
    
    # Legend and labels for the full figure.
    if not minimal_overlay:
        legend = ax.legend(loc='upper left', framealpha=0.95, facecolor='white', 
                          edgecolor='#0099cc', fontsize=10)
        legend.get_frame().set_linewidth(2)
        
        # Title and axis labels.
        title = ax.set_title("MPCC Cart Transporter Controller", 
                             fontsize=16, fontweight='bold', color='#0066cc', pad=10)
        title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white', alpha=0.5)])
        ax.set_xlabel("Position [m]", fontsize=12, color='#333333', fontweight='bold')
        ax.set_ylabel("Height [m]", fontsize=12, color='#333333', fontweight='bold')
        
        # Small source label in the corner.
        fig.text(0.99, 0.01, 'Robotics Control Visualization', 
                ha='right', va='bottom', fontsize=8, color='#999999', style='italic')
    
    # Create filename with start and end positions
    x1_start, x2_start = float(x[0, 0]), float(x[0, 1])
    x1_end, x2_end = float(x[-1, 0]), float(x[-1, 1])
    output_dir = Path(output_dir or Path(trajectory_file).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = output_dir / f"cart_s{x1_start:.2f}_{x2_start:.2f}_e{x1_end:.2f}_{x2_end:.2f}"
    
    # Save trajectory sequence (vertical stack of frames)
    print("Creating trajectory sequence image...")
    
    # Select frames to display (aim for 7-8 frames)
    total_frames = len(x)
    target_n_frames = 8
    auto_interval = max(1, total_frames // target_n_frames)
    selected_indices = list(range(0, total_frames, auto_interval))
    if (total_frames - 1) not in selected_indices:
        selected_indices.append(total_frames - 1)
    # Limit to approximately 8 frames
    if len(selected_indices) > target_n_frames + 1:
        step = len(selected_indices) // target_n_frames
        selected_indices = selected_indices[::step]
        if (total_frames - 1) not in selected_indices:
            selected_indices.append(total_frames - 1)
    n_selected = len(selected_indices)
    
    # Create vertically stacked figure
    panel_height = 1.5  # Height of each panel
    fig_seq = plt.figure(figsize=(12, panel_height * n_selected), facecolor='white')
    
    for idx, frame_i in enumerate(selected_indices):
        ax_frame = fig_seq.add_subplot(n_selected, 1, idx + 1)
        ax_frame.set_aspect('equal', 'box')
        ax_frame.axis('off')
        
        # Draw track
        ax_frame.axhline(y=0, color='#555555', linewidth=2, zorder=1)
        ax_frame.fill_between([track_min, track_max], [-0.1, -0.1], [0, 0], 
                              color='#dddddd', alpha=0.8, zorder=0)
        
        # Get positions for this frame
        x2_i = float(x[frame_i, 1])
        x1_i = float(x[frame_i, 0])
        
        # Draw cart
        cart_rect = mpatches.FancyBboxPatch(
            (x2_i - cart_width/2, 0), cart_width, cart_height,
            boxstyle="round,pad=0.02",
            facecolor='#4da6ff', edgecolor='#0066cc', linewidth=2,
            alpha=0.9, zorder=5
        )
        ax_frame.add_patch(cart_rect)
        
        # Draw wheels
        ax_frame.add_patch(plt.Circle((x2_i - cart_width/3, 0), wheel_radius, color='#333333', zorder=6))
        ax_frame.add_patch(plt.Circle((x2_i + cart_width/3, 0), wheel_radius, color='#333333', zorder=6))
        
        # Draw cargo
        cargo_rect = mpatches.FancyBboxPatch(
            (x1_i - cargo_width/2, cart_height), cargo_width, cargo_height,
            boxstyle="round,pad=0.02",
            facecolor='#ffaa44', edgecolor='#cc6600', linewidth=2,
            alpha=0.9, zorder=7
        )
        ax_frame.add_patch(cargo_rect)
        
        # Draw target position (ghost)
        if x_goal is not None:
            goal_x1, goal_x2 = x_goal[0], x_goal[1]
            # Ghost cart at target
            cart_ghost = mpatches.FancyBboxPatch(
                (goal_x2 - cart_width/2, 0), cart_width, cart_height,
                boxstyle="round,pad=0.02",
                facecolor='none', edgecolor='#0066cc', linewidth=2,
                linestyle='--', alpha=0.5, zorder=3
            )
            ax_frame.add_patch(cart_ghost)
            # Ghost cargo at target
            cargo_ghost = mpatches.FancyBboxPatch(
                (goal_x1 - cargo_width/2, cart_height), cargo_width, cargo_height,
                boxstyle="round,pad=0.02",
                facecolor='none', edgecolor='#cc6600', linewidth=2,
                linestyle='--', alpha=0.5, zorder=3
            )
            ax_frame.add_patch(cargo_ghost)
        
        # Add time label
        t_curr = frame_i * dt
        ax_frame.text(track_min + 0.2, cart_height + cargo_height + 0.2, 
                     f't = {t_curr:.2f}s', fontsize=10, fontweight='bold', color='#333333')
        
        ax_frame.set_xlim(track_min, track_max)
        ax_frame.set_ylim(-0.3, cart_height + cargo_height + 0.5)
    
    fig_seq.tight_layout()
    fig_seq.savefig(f'{filename_prefix}_sequence.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig_seq.savefig(f'{filename_prefix}_sequence.pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Trajectory sequence saved to '{filename_prefix}_sequence.png' and '{filename_prefix}_sequence.pdf'")
    plt.close(fig_seq)
    
    # Save static trajectory summary plot (only if not minimal)
    if not minimal_overlay:
        print("Creating trajectory summary image...")
        fig_static, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    
        # Position vs time
        axes[0, 0].plot(time, x1, 'orange', linewidth=2, label='Cargo (x1)')
        axes[0, 0].plot(time, x2, 'blue', linewidth=2, label='Cart (x2)')
        if x_goal is not None:
            axes[0, 0].axhline(y=x_goal[0], color='orange', linestyle='--', alpha=0.5, label=f'Goal x1={x_goal[0]}')
            axes[0, 0].axhline(y=x_goal[1], color='blue', linestyle='--', alpha=0.5, label=f'Goal x2={x_goal[1]}')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('Position [m]')
        axes[0, 0].set_title('Position vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity vs time
        axes[0, 1].plot(time, x1_dot, 'orange', linewidth=2, label='Cargo vel')
        axes[0, 1].plot(time, x2_dot, 'blue', linewidth=2, label='Cart vel')
        if x_goal is not None:
            axes[0, 1].axhline(y=x_goal[2], color='orange', linestyle='--', alpha=0.5, label=f'Goal v1={x_goal[2]}')
            axes[0, 1].axhline(y=x_goal[3], color='blue', linestyle='--', alpha=0.5, label=f'Goal v2={x_goal[3]}')
        axes[0, 1].set_xlabel('Time [s]')
        axes[0, 1].set_ylabel('Velocity [m/s]')
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Forces vs time
        time_u = np.arange(len(friction_force)) * dt
        axes[1, 0].plot(time_u, friction_force, '#cc0066', linewidth=2, label='Friction (f)')
        axes[1, 0].plot(time_u, control_force, '#009900', linewidth=2, label='Control (u)')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Force [N]')
        axes[1, 0].set_title('Forces vs Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Relative motion
        axes[1, 1].plot(time, relative_pos, 'purple', linewidth=2, label='Relative Position')
        axes[1, 1].plot(time, relative_vel, 'green', linewidth=2, label='Relative Velocity')
        axes[1, 1].axhline(y=l, color='red', linestyle='--', alpha=0.5, label=f'Gap limit (±{l})')
        axes[1, 1].axhline(y=-l, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Time [s]')
        axes[1, 1].set_ylabel('Relative Motion')
        axes[1, 1].set_title('Cargo-Cart Relative Motion')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fig_static.suptitle('Cart Transporter Trajectory Summary', fontsize=14, fontweight='bold')
        fig_static.tight_layout()
        fig_static.savefig(f'{filename_prefix}_summary.png', dpi=200, bbox_inches='tight', facecolor='white')
        print(f"✓ Trajectory summary saved to '{filename_prefix}_summary.png'")
        plt.close(fig_static)
    
    # Save the animation, falling back to GIF when ffmpeg is unavailable.
    try:
        print("Saving animation as MP4...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='CartTransporter'), bitrate=2400)
        anim.save(f'{filename_prefix}_animation.mp4', writer=writer)
        print(f"✓ Animation saved to '{filename_prefix}_animation.mp4'")
    except Exception as e:
        print(f"Could not save animation as MP4: {e}")
        print("Trying to save as GIF instead...")
        try:
            anim.save(f'{filename_prefix}_animation.gif', writer='pillow', fps=20)
            print(f"✓ Animation saved to '{filename_prefix}_animation.gif'")
        except Exception as e2:
            print(f"Could not save animation as GIF: {e2}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", help="trajectory file (default: newest result)")
    parser.add_argument("--out-dir", help="output directory (default: results/cart_transporter)")
    parser.add_argument("--minimal", action="store_true", help="hide axes, legend, and title")
    parser.add_argument("--interval", type=int, default=10, help="frame interval for the sequence")
    args = parser.parse_args()

    trajectory_file = resolve_trajectory("cart_transporter", args.trajectory)
    output_dir = resolve_output_dir("cart_transporter", args.out_dir)
    
    # Goal state: [x1, x2, x1_dot, x2_dot]
    x_goal = [-1.0, -1.5, 0.0, 0.0]
    
    minimal_mode = args.minimal
    frame_interval = args.interval

    # Use the goal recorded by the solver in the trajectory file, if present.
    try:
        with open(trajectory_file) as _gf:
            _gl = _gf.readlines()
        for _gi, _line in enumerate(_gl):
            if _line.startswith('# Goal State'):
                x_goal = [float(_v) for _v in _gl[_gi + 1].split()]
                break
    except Exception as _ge:
        print(f"(using default goal; could not read goal from file: {_ge})")

    print(f"Loading trajectory from: {trajectory_file}")
    print(f"Goal state: x1={x_goal[0]}, x2={x_goal[1]}, v1={x_goal[2]}, v2={x_goal[3]}")
    if minimal_mode:
        print(f"Minimal mode enabled, frame interval: {frame_interval}")
    
    visualize_solution_from_file(
        trajectory_file,
        l=1.0,  # Gap length constraint
        x_goal=x_goal,
        dt=0.02,  # Time step
        minimal_overlay=minimal_mode,
        frame_interval=frame_interval,
        output_dir=output_dir
    )
