#!/usr/bin/env python3
"""
Push-T trajectory visualizer.

Reads a saved trajectory file and writes both an animation and a static overlay.
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
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as path_effects


def get_t_shape_vertices(l=0.05, dc=2.6429):
    """
    Get the T-shape vertices in body frame (centered at centroid).
    
    The T-shape consists of:
    - Horizontal bar at top: from x=-2l to x=+2l, y from (3-dc)*l to (4-dc)*l
    - Vertical stem: from x=-0.5l to x=+0.5l, y from -dc*l to (3-dc)*l
    
    Args:
        l: Characteristic length (half side)
        dc: Limit surface rotational parameter
    
    Returns:
        numpy array of vertices (8x2)
    """
    t_vertices = np.array([
        [-2*l, (4-dc)*l],      # top-left of horizontal bar
        [2*l, (4-dc)*l],       # top-right of horizontal bar
        [2*l, (3-dc)*l],       # top-right inner corner
        [0.5*l, (3-dc)*l],     # stem top-right
        [0.5*l, -dc*l],        # stem bottom-right
        [-0.5*l, -dc*l],       # stem bottom-left
        [-0.5*l, (3-dc)*l],    # stem top-left
        [-2*l, (3-dc)*l],      # top-left inner corner
    ])
    return t_vertices


def visualize_solution_from_file(trajectory_file, l=0.05, dc=2.6429, x_goal=None, dt=0.05,
                                 minimal_overlay=False, output_dir=None):
    """
    Visualize a Push-T trajectory from a solver output file.
    
    Args:
        trajectory_file: Path to trajectory file
        l: Characteristic length (half side of T)
        dc: Limit surface rotational parameter
        x_goal: Goal position [x, y, theta] (optional)
        dt: Time step for animation speed
        minimal_overlay: If True, overlay image will hide axes, legend, and title
        output_dir: Directory for generated images and animation
    """
    
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
    
    # Parse state trajectory (px, py, theta)
    state_lines = []
    for i in range(state_start, control_start - 2):
        if lines[i].strip() and not lines[i].startswith('#'):
            state_lines.append(lines[i])
    
    x_data = []
    for line in state_lines:
        values = [float(v) for v in line.strip().split()]
        x_data.append(values)
    x = np.array(x_data)  # Shape: (timesteps, 3) - each row is [px, py, theta]
    
    # Parse control trajectory (cx, cy, lam[8], v[7], w[7])
    control_lines = []
    for i in range(control_start, len(lines)):
        if lines[i].strip() and not lines[i].startswith('#'):
            control_lines.append(lines[i])
    
    u_data = []
    for line in control_lines:
        values = [float(v) for v in line.strip().split()]
        u_data.append(values)
    u = np.array(u_data)  # Shape: (timesteps, 24) - [cx, cy, lam[8], v[7], w[7]]
    
    print(f"Loaded trajectory: {x.shape[0]} timesteps, state dim={x.shape[1]}, control dim={u.shape[1]}")
    
    # T-shape vertices in body coordinates.
    local_corners = get_t_shape_vertices(l, dc)

    # Use a plain light theme for saved figures.
    plt.style.use('default')
    
    # Main figure layout.
    if minimal_overlay:
        # Compact figure for minimal overlay mode.
        fig = plt.figure(figsize=(14, 10), facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        # Remove margins but keep a little padding.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        ax_info = ax_state = ax_control = None
    else:
        fig = plt.figure(figsize=(16, 10), facecolor='white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main visualization axis
        ax = fig.add_subplot(gs[:, :2])
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5, color='#cccccc')
        ax.set_facecolor('#f8f8f8')
    
        # Info panels (only if not minimal)
        ax_info = fig.add_subplot(gs[0, 2])
        ax_info.axis('off')
        
        ax_state = fig.add_subplot(gs[1, 2])
        ax_state.axis('off')
        
        ax_control = fig.add_subplot(gs[2, 2])
        ax_control.axis('off')

    # Plot the path behind the moving object.
    ax.plot(x[:,0], x[:,1], '#0088cc', alpha=0.4, linewidth=1.5, linestyle='--', label='Trajectory')
    
    # Initial T-shape patch.
    cx0, cy0, th0 = float(x[0,0]), float(x[0,1]), float(x[0,2])
    c0, s0 = np.cos(th0), np.sin(th0)
    R0 = np.array([[c0, -s0],
                   [s0,  c0]])
    corners0 = (local_corners @ R0.T) + np.array([cx0, cy0])
    poly = mpatches.Polygon(corners0, closed=True, facecolor='#4da6ff', 
                           edgecolor='#0066cc', linewidth=2.5, alpha=0.7, label='T-Block')
    poly.set_path_effects([path_effects.withStroke(linewidth=3, foreground='#99ccff', alpha=0.5)])
    ax.add_patch(poly)

    # Heading arrow.
    span = max(np.ptp(x[:,0]), np.ptp(x[:,1]), 1e-3)
    arrow_len = 0.05 * span  # Shorter arrow
    hd_q = ax.quiver([cx0], [cy0], [np.cos(th0)*arrow_len], [np.sin(th0)*arrow_len],
                     angles='xy', scale_units='xy', scale=1, color='#cc0033', width=0.008,
                     headwidth=4, headlength=5, headaxislength=4)

    # Contact point (pusher) in world coordinates.
    ctrl_pt = u[0, :2] if u.shape[0] > 0 else np.array([0.0, 0.0])
    ctrl_world = (np.array([ctrl_pt]) @ R0.T) + np.array([cx0, cy0])
    ctrl_scatter = ax.scatter(ctrl_world[0,0], ctrl_world[0,1], c='#cc00cc', s=150, zorder=5, 
                              label='Pusher', marker='o', edgecolors='#660066', linewidths=2)

    # Force visualization from the lambda values.
    lam_vals = u[0, 2:10] if u.shape[0] > 0 else np.zeros(8)
    # Net force direction from lambdas.
    # sum_x = lam2 + lam4 + lam6 + lam8 (indices 1,3,5,7)
    # sum_y = lam1 + lam3 + lam5 + lam7 (indices 0,2,4,6)
    sum_x = lam_vals[1] + lam_vals[3] + lam_vals[5] + lam_vals[7]
    sum_y = lam_vals[0] + lam_vals[2] + lam_vals[4] + lam_vals[6]
    force_scale = 0.15  # Scale factor for force arrow (shorter)
    force_body = np.array([sum_x, sum_y]) * force_scale
    force_world = R0 @ force_body
    
    fq = ax.quiver([ctrl_world[0,0]], [ctrl_world[0,1]], 
                   [force_world[0]], [force_world[1]], 
                   angles='xy', scale_units='xy', scale=1,
                   color='#009900', width=0.006, label='Contact Force', alpha=0.8,
                   headwidth=4, headlength=5, headaxislength=4)

    # Goal pose.
    if x_goal is not None:
        cx_goal, cy_goal, th_goal = x_goal[0], x_goal[1], x_goal[2]
        c_goal, s_goal = np.cos(th_goal), np.sin(th_goal)
        R_goal = np.array([[c_goal, -s_goal],
                            [s_goal,  c_goal]])
        corners_goal = (local_corners @ R_goal.T) + np.array([cx_goal, cy_goal])
        poly_goal = mpatches.Polygon(corners_goal, closed=True, facecolor='#ffcc66', edgecolor='#ff9900', 
                                      alpha=0.4, linestyle='--', linewidth=2.5, label='Goal Position')
        poly_goal.set_path_effects([path_effects.withStroke(linewidth=3.5, foreground='#ffdd99', alpha=0.3)])
        ax.add_patch(poly_goal)
        
        # Add goal marker circle
        goal_circle = Circle((cx_goal, cy_goal), radius=0.02, color='#ff9900', alpha=0.7, zorder=3)
        ax.add_patch(goal_circle)

    # Axis limits include the T-shape footprint.
    t_extent = 4 * l  # T-shape extends about 4*l in each direction
    x_min = x[:,0].min() - t_extent - 0.02
    x_max = x[:,0].max() + t_extent + 0.02
    y_min = x[:,1].min() - t_extent - 0.02
    y_max = x[:,1].max() + t_extent + 0.02
    
    # Include the goal pose in the limits when available.
    if x_goal is not None:
        x_min = min(x_min, x_goal[0] - t_extent - 0.02)
        x_max = max(x_max, x_goal[0] + t_extent + 0.02)
        y_min = min(y_min, x_goal[1] - t_extent - 0.02)
        y_max = max(y_max, x_goal[1] + t_extent + 0.02)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Match the minimal overlay aspect ratio to the data.
    if minimal_overlay:
        data_width = x_max - x_min
        data_height = y_max - y_min
        aspect = data_width / data_height if data_height > 0 else 1.0
        fig_height = 10
        fig_width = fig_height * aspect
        fig.set_size_inches(fig_width, fig_height)

    # Info panels, skipped for minimal overlays.
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
        ax_panel.text(0.5, 0.85, title, transform=ax_panel.transAxes,
                     fontsize=11, fontweight='bold', color=color,
                     ha='center', va='top')
    
    if not minimal_overlay:
        create_info_panel(ax_info, 'SYSTEM INFO', '#0099cc')
        create_info_panel(ax_state, 'STATE DATA', '#009966')
        create_info_panel(ax_control, 'CONTROL INPUT', '#cc00cc')
        
        # Text fields updated by the animation callback.
        info_texts = {
            'time': ax_info.text(0.5, 0.65, '', transform=ax_info.transAxes, 
                                fontsize=10, ha='center', color='#333333', family='monospace'),
            'step': ax_info.text(0.5, 0.50, '', transform=ax_info.transAxes,
                                fontsize=10, ha='center', color='#333333', family='monospace'),
            'progress': ax_info.text(0.5, 0.35, '', transform=ax_info.transAxes,
                                    fontsize=10, ha='center', color='#333333', family='monospace'),
            'status': ax_info.text(0.5, 0.15, '● RUNNING', transform=ax_info.transAxes,
                                  fontsize=11, ha='center', color='#009900', fontweight='bold'),
            
            'pos': ax_state.text(0.5, 0.65, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#0066cc', family='monospace'),
            'angle': ax_state.text(0.5, 0.50, '', transform=ax_state.transAxes,
                                  fontsize=9, ha='center', color='#0066cc', family='monospace'),
            'vel': ax_state.text(0.5, 0.35, '', transform=ax_state.transAxes,
                                fontsize=9, ha='center', color='#cc6600', family='monospace'),
            'dist': ax_state.text(0.5, 0.15, '', transform=ax_state.transAxes,
                                 fontsize=9, ha='center', color='#cc0033', family='monospace'),
            
            'contact': ax_control.text(0.5, 0.65, '', transform=ax_control.transAxes,
                                      fontsize=9, ha='center', color='#990099', family='monospace'),
            'forces': ax_control.text(0.5, 0.40, '', transform=ax_control.transAxes,
                                     fontsize=8, ha='center', color='#009900', family='monospace'),
            'force_mag': ax_control.text(0.5, 0.15, '', transform=ax_control.transAxes,
                                        fontsize=9, ha='center', color='#cc8800', family='monospace'),
        }
    else:
        info_texts = None

    # Animation callback.
    def update(i):
        # Pose
        cx_i, cy_i, th_i = float(x[i,0]), float(x[i,1]), float(x[i,2])
        c, s = np.cos(th_i), np.sin(th_i)
        R = np.array([[c, -s],
                       [s,  c]])
        corners = (local_corners @ R.T) + np.array([cx_i, cy_i])
        poly.set_xy(corners)

        # Heading
        hd_q.set_offsets([[cx_i, cy_i]])
        hd_q.set_UVC(np.cos(th_i)*arrow_len, np.sin(th_i)*arrow_len)

        # Control point (pusher position in body frame -> world frame)
        ui = u[i] if i < u.shape[0] else u[-1]
        ctrl_pt = ui[:2]  # cx, cy in body frame
        ctrl_world = ( np.array([ctrl_pt]) @ R.T ) + np.array([cx_i, cy_i])
        ctrl_scatter.set_offsets(ctrl_world)

        # Force calculation from lambdas.
        lam_vals = ui[2:10]  # lam[0..7]
        sum_x = lam_vals[1] + lam_vals[3] + lam_vals[5] + lam_vals[7]
        sum_y = lam_vals[0] + lam_vals[2] + lam_vals[4] + lam_vals[6]
        force_body = np.array([sum_x, sum_y]) * force_scale
        force_world = R @ force_body
        
        fq.set_offsets([[ctrl_world[0,0], ctrl_world[0,1]]])
        fq.set_UVC([force_world[0]], [force_world[1]])

        # Estimate planar speed by finite differences.
        if i > 0:
            dx = x[i,0] - x[i-1,0]
            dy = x[i,1] - x[i-1,1]
            vel = np.sqrt(dx**2 + dy**2) / dt
        else:
            vel = 0.0
        
        # Distance to goal
        if x_goal is not None:
            dist_goal = np.sqrt((cx_i - x_goal[0])**2 + (cy_i - x_goal[1])**2)
            angle_diff = abs(th_i - x_goal[2])
        else:
            dist_goal = 0.0
            angle_diff = 0.0
        
        # Update side-panel text when the panel exists.
        if info_texts is not None:
            progress = (i / (len(x) - 1)) * 100
            info_texts['time'].set_text(f'Time: {i*dt:6.2f}s')
            info_texts['step'].set_text(f'Step: {i:4d}/{len(x)-1}')
            info_texts['progress'].set_text(f'Progress: {progress:5.1f}%')
            
            # Update status at end
            if i >= len(x) - 1:
                info_texts['status'].set_text('● COMPLETE')
                info_texts['status'].set_color('#0066cc')
            
            info_texts['pos'].set_text(f'Pos: ({cx_i:+.4f}, {cy_i:+.4f})')
            info_texts['angle'].set_text(f'θ: {th_i:+.4f} rad ({th_i*180/np.pi:+.1f}°)')
            info_texts['vel'].set_text(f'Vel: {vel:.4f} m/s')
            info_texts['dist'].set_text(f'Goal Dist: {dist_goal:.4f}m, Δθ: {angle_diff:.4f}')
            
            info_texts['contact'].set_text(f'Contact: ({ctrl_pt[0]:+.4f}, {ctrl_pt[1]:+.4f})')
            force_mag = np.sqrt(sum_x**2 + sum_y**2)
            info_texts['forces'].set_text(f'Σx: {sum_x:+.4f}, Σy: {sum_y:+.4f}')
            info_texts['force_mag'].set_text(f'|F|: {force_mag:.4f}')

        if info_texts is not None:
            return [poly, hd_q, ctrl_scatter, fq] + list(info_texts.values())
        else:
            return [poly, hd_q, ctrl_scatter, fq]

    # Animate (interval in ms, 2x slower for better visualization)
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, update, frames=len(x), interval=max(100, int(dt*2000)),
                                   blit=False, repeat=True)
    
    # Tight layout is only needed for the full figure.
    if not minimal_overlay:
        fig.tight_layout()
    
    # Legend and labels for the full figure.
    if not minimal_overlay:
        legend = ax.legend(loc='upper left', framealpha=0.95, facecolor='white', 
                          edgecolor='#0099cc', fontsize=10, ncol=1)
        legend.get_frame().set_linewidth(2)
        
        # Title and axis labels.
        title = ax.set_title("MPCC PushT Controller", 
                             fontsize=16, fontweight='bold', color='#0066cc', pad=20)
        title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white', alpha=0.5)])
        ax.set_xlabel("X Position [m]", fontsize=12, color='#333333', fontweight='bold')
        ax.set_ylabel("Y Position [m]", fontsize=12, color='#333333', fontweight='bold')
        
        # Style the tick labels
        ax.tick_params(colors='#555555', labelsize=9)
        
        # Small source label in the corner.
        fig.text(0.99, 0.01, 'Robotics Control Visualization', 
                ha='right', va='bottom', fontsize=8, color='#999999', style='italic')
    
    # Save individual frames for each timestep
    import os
    
    # Static figure with all timesteps overlaid.
    print("Creating trajectory overlay image...")
    
    # Choose a figure width from the trajectory aspect ratio.
    data_width = x_max - x_min
    data_height = y_max - y_min
    aspect_static = data_width / data_height if data_height > 0 else 1.0
    fig_height_static = 10
    fig_width_static = fig_height_static * aspect_static
    
    fig_static, ax_static = plt.subplots(figsize=(fig_width_static, fig_height_static), facecolor='white')
    ax_static.set_aspect('equal', 'box')
    
    if minimal_overlay:
        ax_static.axis('off')
    else:
        ax_static.grid(True, alpha=0.4, linestyle='--', linewidth=0.5, color='#cccccc')
        ax_static.set_facecolor('#f8f8f8')
    
    # Plot trajectory path
    ax_static.plot(x[:,0], x[:,1], '#0088cc', alpha=0.6, linewidth=2, linestyle='-', label='Trajectory')
    
    # Draw all T-shape states with fading alpha
    n_frames = len(x)
    force_scale_static = 0.03  # Smaller scale for static overlay
    for i in range(n_frames):
        cx_i, cy_i, th_i = float(x[i,0]), float(x[i,1]), float(x[i,2])
        c, s = np.cos(th_i), np.sin(th_i)
        R = np.array([[c, -s], [s, c]])
        corners = (local_corners @ R.T) + np.array([cx_i, cy_i])
        
        # Older states are lighter; newer states are darker.
        alpha = 0.1 + 0.6 * (i / (n_frames - 1)) if n_frames > 1 else 0.7
        
        # Color progresses from blue to red along the trajectory.
        color_ratio = i / (n_frames - 1) if n_frames > 1 else 0
        facecolor = (0.3 + 0.5*color_ratio, 0.5 - 0.3*color_ratio, 0.9 - 0.7*color_ratio, alpha * 0.5)
        edgecolor = (0.1 + 0.6*color_ratio, 0.2 - 0.1*color_ratio, 0.7 - 0.5*color_ratio, alpha)
        
        poly_i = mpatches.Polygon(corners, closed=True, facecolor=facecolor, 
                                   edgecolor=edgecolor, linewidth=1.5)
        ax_static.add_patch(poly_i)
        
        # Draw contact force arrow
        if i < u.shape[0]:
            ui = u[i]
            ctrl_pt = ui[:2]  # contact point in body frame
            ctrl_world = (np.array([ctrl_pt]) @ R.T) + np.array([cx_i, cy_i])
            
            # Forces from lambda values
            lam_vals = ui[2:10]  # lam[0..7]
            # sum_x = lam2 + lam4 + lam6 + lam8 (indices 1,3,5,7)
            # sum_y = lam1 + lam3 + lam5 + lam7 (indices 0,2,4,6)
            sum_x = lam_vals[1] + lam_vals[3] + lam_vals[5] + lam_vals[7]
            sum_y = lam_vals[0] + lam_vals[2] + lam_vals[4] + lam_vals[6]
            force_body = np.array([sum_x, sum_y]) * force_scale_static
            force_world = R @ force_body
            
            force_mag = np.sqrt(sum_x**2 + sum_y**2)
            if force_mag > 0.01:  # Only draw if force is significant
                ax_static.arrow(ctrl_world[0,0], ctrl_world[0,1], 
                               force_world[0], force_world[1],
                               head_width=0.012, head_length=0.006, 
                               fc='#9933ff', ec='#6600cc', alpha=min(alpha+0.3, 1.0), linewidth=2.0,
                               zorder=10)
    
    # Highlight initial and final states
    # Initial state (green dashed outline)
    cx0, cy0, th0 = float(x[0,0]), float(x[0,1]), float(x[0,2])
    c0, s0 = np.cos(th0), np.sin(th0)
    R0_s = np.array([[c0, -s0], [s0, c0]])
    corners0 = (local_corners @ R0_s.T) + np.array([cx0, cy0])
    poly_init = mpatches.Polygon(corners0, closed=True, facecolor='none', 
                                  edgecolor='#00cc00', linewidth=3, alpha=0.9, 
                                  linestyle='--', label='Start')
    ax_static.add_patch(poly_init)
    
    # Final state (red solid)
    cxf, cyf, thf = float(x[-1,0]), float(x[-1,1]), float(x[-1,2])
    cf, sf = np.cos(thf), np.sin(thf)
    Rf = np.array([[cf, -sf], [sf, cf]])
    cornersf = (local_corners @ Rf.T) + np.array([cxf, cyf])
    poly_final = mpatches.Polygon(cornersf, closed=True, facecolor='#ff6666', 
                                   edgecolor='#cc0000', linewidth=3, alpha=0.8, label='Final')
    ax_static.add_patch(poly_final)
    
    # Goal position (orange dashed outline)
    if x_goal is not None:
        cx_goal, cy_goal, th_goal = x_goal[0], x_goal[1], x_goal[2]
        c_goal, s_goal = np.cos(th_goal), np.sin(th_goal)
        R_goal = np.array([[c_goal, -s_goal], [s_goal, c_goal]])
        corners_goal = (local_corners @ R_goal.T) + np.array([cx_goal, cy_goal])
        poly_goal = mpatches.Polygon(corners_goal, closed=True, facecolor='none', 
                                      edgecolor='#ff9900', linewidth=3, alpha=0.9, 
                                      linestyle='--', label='Goal')
        ax_static.add_patch(poly_goal)
    
    # Reuse the animation limits.
    ax_static.set_xlim(x_min, x_max)
    ax_static.set_ylim(y_min, y_max)
    
    # Minimal overlay keeps only the drawing.
    if minimal_overlay:
        fig_static.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    
    if not minimal_overlay:
        ax_static.legend(loc='upper left', fontsize=10)
        ax_static.set_title(f"PushT Trajectory ({n_frames} timesteps)", fontsize=14, fontweight='bold')
        ax_static.set_xlabel("X Position [m]", fontsize=12)
        ax_static.set_ylabel("Y Position [m]", fontsize=12)
    
    # Include start/end poses in the filename.
    x_start, y_start, th_start = float(x[0,0]), float(x[0,1]), float(x[0,2])
    x_end, y_end, th_end = float(x[-1,0]), float(x[-1,1]), float(x[-1,2])
    output_dir = Path(output_dir or Path(trajectory_file).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = output_dir / f"push_t_s{x_start:.2f}_{y_start:.2f}_{th_start:.2f}_e{x_end:.2f}_{y_end:.2f}_{th_end:.2f}"
    
    fig_static.savefig(f'{filename_prefix}_overlay.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig_static.savefig(f'{filename_prefix}_overlay.pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Trajectory overlay saved to '{filename_prefix}_overlay.png' and '{filename_prefix}_overlay.pdf'")
    plt.close(fig_static)
    
    # Save the animation, falling back to GIF when ffmpeg is unavailable.
    try:
        print("Saving animation as MP4...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='PushT'), bitrate=1800)
        anim.save(f'{filename_prefix}_animation.mp4', writer=writer)
        print(f"✓ Animation saved to '{filename_prefix}_animation.mp4'")
    except Exception as e:
        print(f"Could not save animation as MP4: {e}")
        print("Trying to save as GIF instead...")
        try:
            anim.save(f'{filename_prefix}_animation.gif', writer='pillow', fps=15)
            print(f"✓ Animation saved to '{filename_prefix}_animation.gif'")
        except Exception as e2:
            print(f"Could not save animation as GIF: {e2}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", help="trajectory file (default: newest result)")
    parser.add_argument("--out-dir", help="output directory (default: results/push_t)")
    parser.add_argument("--minimal", action="store_true", help="hide axes, legend, and title")
    args = parser.parse_args()

    trajectory_file = resolve_trajectory("push_t", args.trajectory)
    output_dir = resolve_output_dir("push_t", args.out_dir)
    l = 0.05       # Characteristic length
    dc = 2.6429    # Limit surface rotational parameter
    x_goal = [1.0, 0.5, 0.0]  # Default goal position from push_t_penalty.cpp
    dt = 0.05      # Time step
    
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

    # Minimal mode removes axes, legend and title from the overlay.
    minimal_overlay = args.minimal
    
    print("="*70)
    print("PushT Trajectory Visualization")
    print("="*70)
    print(f"Loading trajectory from: {trajectory_file}")
    print(f"Goal position: x={x_goal[0]}, y={x_goal[1]}, theta={x_goal[2]}")
    print(f"T-shape parameters: l={l}, dc={dc}")
    print(f"Minimal overlay: {minimal_overlay}")
    print("="*70)
    
    visualize_solution_from_file(
        trajectory_file,
        l=l,
        dc=dc,
        x_goal=x_goal,
        dt=dt,
        minimal_overlay=minimal_overlay,
        output_dir=output_dir
    )
