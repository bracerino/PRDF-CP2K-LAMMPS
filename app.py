import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import base64
import io
import os
import re
from ase import Atoms
from ase.io import read
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.structure import PartialRadialDistributionFunction
import streamlit.components.v1 as components
st.set_page_config(page_title="CP2K XYZ PRDF Calculator", layout="wide")

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import streamlit as st

import plotly.graph_objects as go
import numpy as np

legend_style = dict(font=dict(size=20))

if 'show_pbc_warning' not in st.session_state:
    st.session_state.show_pbc_warning = False


def blue_divider():
    st.markdown("""
    <hr style="
        height: 4px;
        background-color: #0066cc;
        border: none;
        margin: 20px 0;
    ">
    """, unsafe_allow_html=True)


def parse_cp2k_xyz_with_headers(file_content, manual_lattice=None):
    lines = file_content.strip().split('\n')
    frames = []
    headers = []
    i = 0
    has_lattice_in_file = "Lattice=" in file_content

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        try:
            num_atoms = int(lines[i].strip())
            i += 1

            if i >= len(lines):
                break

            header_line = lines[i].strip() if i < len(lines) else ""
            headers.append(header_line)
            i += 1

            cell_matrix = None
            if has_lattice_in_file and "Lattice=" in header_line:
                lattice_match = re.search(r'Lattice="([^"]*)"', header_line)
                if lattice_match:
                    lattice_str = lattice_match.group(1)
                    lattice_values = [float(x) for x in lattice_str.split()]
                    if len(lattice_values) == 9:
                        cell_matrix = np.array(lattice_values).reshape(3, 3)
            elif manual_lattice is not None:
                cell_matrix = manual_lattice

            symbols = []
            positions = []

            for j in range(num_atoms):
                if i + j >= len(lines):
                    break
                parts = lines[i + j].strip().split()
                if len(parts) >= 4:
                    symbols.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if len(symbols) == num_atoms and len(positions) == num_atoms:
                atoms = Atoms(symbols=symbols, positions=positions)
                if cell_matrix is not None:
                    atoms.set_cell(cell_matrix)
                    atoms.set_pbc([True, True, True])
                    atoms.wrap()
                frames.append(atoms)

            i += num_atoms

        except (ValueError, IndexError):
            i += 1

    return frames, has_lattice_in_file, headers


def parse_lattice_file(file_content):
    lines = file_content.strip().split('\n')
    lattice_data = {}

    start_idx = 0
    if lines[0].strip().startswith('#'):
        start_idx = 1

    for line in lines[start_idx:]:
        if line.strip():
            parts = line.strip().split()
            if len(parts) >= 11:
                try:
                    step = int(parts[0])
                    time = float(parts[1])

                    # Extract lattice vectors (Ax, Ay, Az, Bx, By, Bz, Cx, Cy, Cz)
                    lattice_components = [float(parts[i]) for i in range(2, 11)]
                    lattice_matrix = np.array(lattice_components).reshape(3, 3)

                    lattice_data[step] = {
                        'time': time,
                        'matrix': lattice_matrix
                    }
                except (ValueError, IndexError):
                    continue

    return lattice_data


def create_xyz_with_lattice(frames, lattice_data=None, manual_lattice=None, original_headers=None):
    xyz_content = []

    for i, frame in enumerate(frames):
        num_atoms = len(frame)
        xyz_content.append(str(num_atoms))

        # Determine lattice matrix for this frame
        if lattice_data is not None:
            # Use lattice data from file - find closest timestep
            available_steps = sorted(lattice_data.keys())
            if available_steps:
                # Use the step that matches the frame index or closest
                if i in lattice_data:
                    lattice_matrix = lattice_data[i]['matrix']
                elif available_steps:
                    closest_step = min(available_steps, key=lambda x: abs(x - i))
                    lattice_matrix = lattice_data[closest_step]['matrix']
                else:
                    lattice_matrix = manual_lattice
            else:
                lattice_matrix = manual_lattice
        else:
            lattice_matrix = manual_lattice

        original_header = ""
        if original_headers and i < len(original_headers):
            original_header = original_headers[i]
        else:
            original_header = f'i = {i}, time = {i * 0.5:.3f}, E = 0.0'

        if lattice_matrix is not None:
            lattice_str = ' '.join([f'{val:.6f}' for val in lattice_matrix.flatten()])

            if "Lattice=" in original_header:
                import re
                header = re.sub(r'Lattice="[^"]*"', f'Lattice="{lattice_str}"', original_header)
            else:
                if not original_header.endswith(' '):
                    original_header += ' '
                header = original_header + f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'
        else:
            header = original_header

        xyz_content.append(header)

        symbols = frame.get_chemical_symbols()
        positions = frame.get_positions()

        for symbol, pos in zip(symbols, positions):
            xyz_content.append(f'{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}')

    return '\n'.join(xyz_content)


def wrap_coordinates_with_lattice(frames, lattice_data=None, manual_lattice=None, original_headers=None):
    wrapped_frames = []
    if manual_lattice is not None and lattice_data is None:
        constant_lattice = manual_lattice
        for frame in frames:
            new_frame = frame.copy()
            new_frame.set_cell(constant_lattice)
            new_frame.set_pbc([True, True, True])
            new_frame.wrap()
            wrapped_frames.append(new_frame)

    elif lattice_data is not None:
        for i, frame in enumerate(frames):
            new_frame = frame.copy()

            original_header = ""
            if original_headers and i < len(original_headers):
                original_header = original_headers[i]

            lattice_matrix = None
            if original_header:
                frame_timestep = extract_timestep_from_header(original_header)

                if frame_timestep is not None:
                    if frame_timestep in lattice_data:
                        lattice_matrix = lattice_data[frame_timestep]['matrix']
                    else:
                        available_steps = sorted(lattice_data.keys())
                        if available_steps:
                            closest_step = min(available_steps, key=lambda x: abs(x - frame_timestep))
                            lattice_matrix = lattice_data[closest_step]['matrix']
                else:
                    available_steps = sorted(lattice_data.keys())
                    if available_steps:
                        if i < len(available_steps):
                            step_key = available_steps[i]
                            lattice_matrix = lattice_data[step_key]['matrix']
                        else:
                            lattice_matrix = lattice_data[available_steps[-1]]['matrix']
            else:
                # No header info - use frame index
                available_steps = sorted(lattice_data.keys())
                if available_steps:
                    if i < len(available_steps):
                        step_key = available_steps[i]
                        lattice_matrix = lattice_data[step_key]['matrix']
                    else:
                        lattice_matrix = lattice_data[available_steps[-1]]['matrix']

            if lattice_matrix is not None:
                new_frame.set_cell(lattice_matrix)
                new_frame.set_pbc([True, True, True])
                new_frame.wrap()

            wrapped_frames.append(new_frame)

    else:
        # No lattice parameters - return copies without wrapping
        wrapped_frames = [frame.copy() for frame in frames]

    return wrapped_frames


def extract_timestep_from_header(header_line):
    import re

    # Try to extract step number from "i = X" format
    i_match = re.search(r'i\s*=\s*(\d+)', header_line)
    if i_match:
        return int(i_match.group(1))

    # Try to extract from "step = X" format
    step_match = re.search(r'step\s*=\s*(\d+)', header_line.lower())
    if step_match:
        return int(step_match.group(1))

    # Try to extract from "time = X" format and assume 1 fs timestep
    time_match = re.search(r'time\s*=\s*([0-9.]+)', header_line.lower())
    if time_match:
        time_val = float(time_match.group(1))
        return int(time_val)

    return None


def create_xyz_with_lattice_optimized(frames, lattice_data=None, manual_lattice=None, original_headers=None):
    xyz_lines = []
    constant_lattice_str = None
    if manual_lattice is not None and lattice_data is None:
        constant_lattice_str = ' '.join([f'{val:.6f}' for val in manual_lattice.flatten()])

    for i, frame in enumerate(frames):
        num_atoms = len(frame)
        xyz_lines.append(str(num_atoms))

        original_header = ""
        if original_headers and i < len(original_headers):
            original_header = original_headers[i]
        else:
            original_header = f'i = {i}, time = {i * 0.5:.3f}, E = 0.0'

        lattice_matrix = None
        if lattice_data is not None:
            frame_timestep = extract_timestep_from_header(original_header)

            if frame_timestep is not None:
                if frame_timestep in lattice_data:
                    lattice_matrix = lattice_data[frame_timestep]['matrix']
                else:
                    available_steps = sorted(lattice_data.keys())
                    if available_steps:
                        closest_step = min(available_steps, key=lambda x: abs(x - frame_timestep))
                        lattice_matrix = lattice_data[closest_step]['matrix']

                        if abs(closest_step - frame_timestep) > 0:
                            print(f"Debug: Frame {i} (step {frame_timestep}) mapped to lattice step {closest_step}")
            else:
                available_steps = sorted(lattice_data.keys())
                if available_steps:
                    if i < len(available_steps):
                        step_key = available_steps[i]
                        lattice_matrix = lattice_data[step_key]['matrix']
                    else:
                        lattice_matrix = lattice_data[available_steps[-1]]['matrix']

        elif manual_lattice is not None:
            lattice_matrix = manual_lattice

        if lattice_matrix is not None:
            if constant_lattice_str is not None:
                lattice_str = constant_lattice_str
            else:
                lattice_str = ' '.join([f'{val:.6f}' for val in lattice_matrix.flatten()])
            if "Lattice=" in original_header:

                import re
                header = re.sub(r'Lattice="[^"]*"', f'Lattice="{lattice_str}"', original_header)
            else:
                # Append lattice parameters to original header
                if not original_header.endswith(' '):
                    original_header += ' '
                header = original_header + f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3'
        else:
            header = original_header

        xyz_lines.append(header)

        symbols = frame.get_chemical_symbols()
        positions = frame.get_positions()

        coord_lines = [f'{symbol} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}'
                       for symbol, pos in zip(symbols, positions)]
        xyz_lines.extend(coord_lines)

    return '\n'.join(xyz_lines)


def wrap_coordinates_with_lattice_progress(frames, lattice_data=None, manual_lattice=None, original_headers=None):
    import streamlit as st

    wrapped_frames = []
    total_frames = len(frames)

    # Show progress for large trajectories
    if total_frames > 100:
        progress_bar = st.progress(0)
        status_text = st.empty()

    if manual_lattice is not None and lattice_data is None:
        constant_lattice = manual_lattice
        for i, frame in enumerate(frames):
            new_frame = frame.copy()
            new_frame.set_cell(constant_lattice)
            new_frame.set_pbc([True, True, True])
            new_frame.wrap()
            wrapped_frames.append(new_frame)

            # Update progress for large files
            if total_frames > 100 and i % max(1, total_frames // 20) == 0:
                progress = (i + 1) / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Wrapping coordinates: {i + 1}/{total_frames} frames")

    elif lattice_data is not None:
        for i, frame in enumerate(frames):
            new_frame = frame.copy()

            # Get original header if available
            original_header = ""
            if original_headers and i < len(original_headers):
                original_header = original_headers[i]

            lattice_matrix = None
            if original_header:
                frame_timestep = extract_timestep_from_header(original_header)

                if frame_timestep is not None:
                    if frame_timestep in lattice_data:
                        lattice_matrix = lattice_data[frame_timestep]['matrix']
                    else:
                        available_steps = sorted(lattice_data.keys())
                        if available_steps:
                            closest_step = min(available_steps, key=lambda x: abs(x - frame_timestep))
                            lattice_matrix = lattice_data[closest_step]['matrix']
                else:
                    available_steps = sorted(lattice_data.keys())
                    if available_steps:
                        if i < len(available_steps):
                            step_key = available_steps[i]
                            lattice_matrix = lattice_data[step_key]['matrix']
                        else:
                            lattice_matrix = lattice_data[available_steps[-1]]['matrix']

            if lattice_matrix is not None:
                new_frame.set_cell(lattice_matrix)
                new_frame.set_pbc([True, True, True])
                new_frame.wrap()

            wrapped_frames.append(new_frame)
            if total_frames > 100 and i % max(1, total_frames // 20) == 0:
                progress = (i + 1) / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frames: {i + 1}/{total_frames}")

    else:
        wrapped_frames = [frame.copy() for frame in frames]

    if total_frames > 100:
        progress_bar.empty()
        status_text.empty()

    return wrapped_frames


def display_lattice_file_info(lattice_data):
    """Display information about the uploaded lattice parameter file"""
    if lattice_data:
        st.success(f"📁 **Lattice Parameter File Information:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Steps", len(lattice_data))

        with col2:
            steps = sorted(lattice_data.keys())
            st.write(f"**Step Range:** {steps[0]} - {steps[-1]}")

        with col3:
            times = [data['time'] for data in lattice_data.values()]
            st.write(f"**Time Range:** {min(times):.1f} - {max(times):.1f} fs")

        if lattice_data:
            first_step = min(lattice_data.keys())
            first_matrix = lattice_data[first_step]['matrix']

            with st.expander("Preview First Lattice Matrix"):
                st.write(f"**Step {first_step} (t = {lattice_data[first_step]['time']:.1f} fs):**")
                st.code(f"""
a = [{first_matrix[0, 0]:8.3f} {first_matrix[0, 1]:8.3f} {first_matrix[0, 2]:8.3f}]
b = [{first_matrix[1, 0]:8.3f} {first_matrix[1, 1]:8.3f} {first_matrix[1, 2]:8.3f}]
c = [{first_matrix[2, 0]:8.3f} {first_matrix[2, 1]:8.3f} {first_matrix[2, 2]:8.3f}]
                """)

                cell_volume = np.linalg.det(first_matrix)
                st.write(f"Volume: {cell_volume:.3f} Ų")
    else:
        st.warning("No valid lattice data found in the file")


def create_combined_animated_prdf_plot(all_prdf_dict, all_distance_dict, frame_indices,
                                       smoothing_params, line_style, animation_speed, colors):
    fig_combined = go.Figure()

    all_smoothed_data = {}
    max_frames = 0

    for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):
        valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]
        if not valid_prdf:
            continue

        max_frames = max(max_frames, len(valid_prdf))

        smoothed_prdf = []
        for frame_data in valid_prdf:
            if smoothing_params.get("enabled", False):
                try:
                    smoothed_frame = smooth_prdf_data(
                        all_distance_dict[comb], frame_data,
                        method=smoothing_params["method"],
                        **{k: v for k, v in smoothing_params.items()
                           if k not in ["enabled", "method"]}
                    )
                    smoothed_prdf.append(smoothed_frame)
                except:
                    smoothed_prdf.append(frame_data)
            else:
                smoothed_prdf.append(frame_data)

        all_smoothed_data[comb] = {
            'data': smoothed_prdf,
            'distances': all_distance_dict[comb],
            'color': rgb_to_hex(colors[idx % len(colors)])
        }

    for idx, (comb, plot_info) in enumerate(all_smoothed_data.items()):
        fig_combined.add_trace(go.Scatter(
            x=plot_info['distances'],
            y=plot_info['data'][0],
            mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
            name=f"{comb[0]}-{comb[1]}" + (" (Smoothed)" if smoothing_params.get("enabled", False) else ""),
            line=dict(color=plot_info['color'], width=2),
            marker=dict(size=6) if line_style == "Lines + Markers" else dict()
        ))

    frames = []
    for i in range(max_frames):
        frame_data = []
        for idx, (comb, plot_info) in enumerate(all_smoothed_data.items()):
            if i < len(plot_info['data']):
                frame_data.append(go.Scatter(
                    x=plot_info['distances'],
                    y=plot_info['data'][i],
                    mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
                    name=f"{comb[0]}-{comb[1]}" + (" (Smoothed)" if smoothing_params.get("enabled", False) else ""),
                    line=dict(color=plot_info['color'], width=2),
                    marker=dict(size=6) if line_style == "Lines + Markers" else dict()
                ))

        frame = go.Frame(data=frame_data, name=f"frame_{i}")
        frames.append(frame)

    fig_combined.frames = frames

    updatemenus = [dict(
        type="buttons",
        direction="right",
        x=0.1, y=-0.1,
        buttons=[
            dict(label="▶️ Play", method="animate",
                 args=[None, {"frame": {"duration": int(animation_speed * 1000),
                                        "redraw": True},
                              "fromcurrent": True, "mode": "immediate"}]),
            dict(label="⏹️ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": True},
                                "mode": "immediate", "transition": {"duration": 0}}])
        ]
    )]

    sliders = [dict(
        active=0,
        currentvalue=dict(font=dict(size=16), prefix="Frame: ", visible=True),
        pad=dict(b=10, t=50),
        len=0.9, x=0.1, y=0,
        steps=[dict(
            method="animate",
            args=[[f"frame_{k}"], {"frame": {"duration": 100, "redraw": True},
                                   "mode": "immediate", "transition": {"duration": 0}}],
            label=f"{frame_indices[k]}" if k < len(frame_indices) else f"{k}"
        ) for k in range(max_frames)]
    )]

    all_y_values = []
    for plot_info in all_smoothed_data.values():
        for frame_data in plot_info['data']:
            all_y_values.extend(frame_data)

    max_y = max(all_y_values) * 1.1 if all_y_values else 1.0

    title_text = "Combined PRDF Animation: All Pairs"
    if smoothing_params.get("enabled", False):
        method_name = {
            "moving_average": "Moving Avg",
            "gaussian": "Gaussian",
            "savgol": "Savitzky-Golay",
            "spline": "Spline"
        }.get(smoothing_params["method"], "Smoothed")
        title_text += f" ({method_name})"

    fig_combined.update_layout(
        title={'text': title_text, 'font': dict(size=20, color="black")},
        xaxis_title={'text': "Distance (Å)", 'font': dict(size=20, color="black")},
        yaxis_title={'text': "PRDF Intensity", 'font': dict(size=20, color="black")},
        updatemenus=updatemenus,
        sliders=sliders,
        font=dict(size=20, color="black"),
        xaxis=dict(tickfont=dict(size=20, color="black")),
        yaxis=dict(tickfont=dict(size=20, color="black"), range=[0, max_y]),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=16))
    )

    return fig_combined


def display_trajectory_info(uploaded_file, file_type):
    import re
    try:
        if file_type == "xyz":
            uploaded_file.seek(0)
            file_content = uploaded_file.read().decode('utf-8')

            lines = file_content.strip().split('\n')
            i = 0

            while i < len(lines) and not lines[i].strip():
                i += 1

            if i >= len(lines):
                return

            try:
                # Parse first frame
                num_atoms = int(lines[i].strip())
                i += 1

                first_header_line = lines[i].strip() if i < len(lines) else ""
                i += 1

                symbols = []
                for j in range(num_atoms):
                    if i + j >= len(lines):
                        break
                    parts = lines[i + j].strip().split()
                    if len(parts) >= 4:
                        symbols.append(parts[0])

                i += num_atoms

                # Try to find second frame
                second_header_line = ""
                second_timestep_info = "Not found"
                has_lattice_second = False
                while i < len(lines) and not lines[i].strip():
                    i += 1

                if i < len(lines):
                    try:
                        # Check if this is a valid second frame
                        second_num_atoms = int(lines[i].strip())
                        i += 1
                        if i < len(lines) and second_num_atoms == num_atoms:
                            second_header_line = lines[i].strip()
                            has_lattice_second = "Lattice=" in second_header_line

                            # Extract timestep info from second frame
                            if "step" in second_header_line.lower():
                                timestep_match = re.search(r'step[=\s]*(\d+)', second_header_line.lower())
                                if timestep_match:
                                    second_timestep_info = f"Step {timestep_match.group(1)}"
                            elif "time" in second_header_line.lower():
                                timestep_match = re.search(r'time[=\s]*([0-9.]+)', second_header_line.lower())
                                if timestep_match:
                                    second_timestep_info = f"Time {timestep_match.group(1)}"
                            elif "i =" in second_header_line.lower():
                                i_match = re.search(r'i\s*=\s*(\d+)', second_header_line.lower())
                                if i_match:
                                    second_timestep_info = f"Frame {i_match.group(1)}"
                    except (ValueError, IndexError):
                        pass

                total_frames = sum(1 for line in lines if line.strip().startswith('i ='))

                st.success(f"📁 **CP2K Trajectory Information:**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Frames", total_frames)
                    st.metric("Atoms per Frame", num_atoms)

                with col2:
                    unique_elements = list(set(symbols))
                    element_counts = {elem: symbols.count(elem) for elem in unique_elements}

                    st.write("**Element Composition:**")
                    for element, count in sorted(element_counts.items()):
                        st.write(f"• {element}: {count} atoms")

                with col3:
                    # First frame timestep info
                    first_timestep_info = "Not specified"
                    if "step" in first_header_line.lower():
                        timestep_match = re.search(r'step[=\s]*(\d+)', first_header_line.lower())
                        if timestep_match:
                            first_timestep_info = f"Step {timestep_match.group(1)}"
                    elif "time" in first_header_line.lower():
                        timestep_match = re.search(r'time[=\s]*([0-9.]+)', first_header_line.lower())
                        if timestep_match:
                            first_timestep_info = f"Time {timestep_match.group(1)}"
                    elif "i =" in first_header_line.lower():
                        i_match = re.search(r'i\s*=\s*(\d+)', first_header_line.lower())
                        if i_match:
                            first_timestep_info = f"Frame {i_match.group(1)}"

                    st.write("**Frame Information:**")
                    st.write(f"• **First frame:** {first_timestep_info}")
                    st.write(f"• **Second frame:** {second_timestep_info}")

                    has_lattice = "Lattice=" in file_content
                    lattice_status = "✅ Present" if has_lattice else "❌ Not found"
                    st.write(f"• **Lattice params:** {lattice_status}")

                    if has_lattice:
                        lattice_match = re.search(r'Lattice="([^"]*)"', first_header_line)
                        if lattice_match:
                            lattice_str = lattice_match.group(1)
                            lattice_values = [float(x) for x in lattice_str.split()]
                            if len(lattice_values) == 9:
                                st.write("**First Frame Lattice Matrix (Å):**")
                                lattice_matrix = np.array(lattice_values).reshape(3, 3)
                                for i, row in enumerate(['a', 'b', 'c']):
                                    st.write(
                                        f"• {row}: [{lattice_matrix[i, 0]:6.2f} {lattice_matrix[i, 1]:6.2f} {lattice_matrix[i, 2]:6.2f}]")

                                cell_volume = np.linalg.det(lattice_matrix)
                                st.write(f"• Volume: {cell_volume:.2f} Ų")

                        # Show second frame lattice if different
                        if has_lattice_second and second_header_line != first_header_line:
                            lattice_match_second = re.search(r'Lattice="([^"]*)"', second_header_line)
                            if lattice_match_second:
                                lattice_str_second = lattice_match_second.group(1)
                                if lattice_str_second != lattice_str:
                                    st.write("**⚠️ Variable lattice detected**")
                                    st.write("Second frame has different lattice parameters")

            except (ValueError, IndexError) as e:
                st.warning(f"Could not parse XYZ file structure: {str(e)}")

        elif file_type == "lammps":
            uploaded_file.seek(0)
            file_content_bytes = uploaded_file.read()

            try:
                content = file_content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = file_content_bytes.decode('latin-1')

            lines = content.splitlines()

            # Parse first frame
            i = 0
            first_timestep = None
            second_timestep = None
            num_atoms = None
            element_info = {}

            # Find first timestep
            while i < len(lines):
                line = lines[i].strip()
                if 'ITEM: TIMESTEP' in line:
                    i += 1
                    if i < len(lines):
                        first_timestep = int(lines[i].strip())
                        break
                i += 1

            # Continue searching for second timestep
            while i < len(lines):
                line = lines[i].strip()
                if 'ITEM: TIMESTEP' in line:
                    i += 1
                    if i < len(lines):
                        second_timestep = int(lines[i].strip())
                        break
                i += 1

            # Reset and get atom info
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if 'ITEM: NUMBER OF ATOMS' in line:
                    i += 1
                    if i < len(lines):
                        num_atoms = int(lines[i].strip())
                    i += 1
                    continue

                if 'ITEM: ATOMS' in line and num_atoms is not None:
                    header = line.replace('ITEM: ATOMS', '').strip().split()
                    i += 1

                    element_idx = header.index('element') if 'element' in header else -1
                    type_idx = header.index('type') if 'type' in header else -1

                    elements = []
                    for j in range(min(num_atoms, 100)):
                        if i + j >= len(lines):
                            break

                        values = lines[i + j].strip().split()
                        if len(values) < len(header):
                            continue

                        if element_idx >= 0:
                            elements.append(values[element_idx])
                        elif type_idx >= 0:
                            atom_type = int(values[type_idx])
                            type_map = {1: 'Si', 2: 'O', 3: 'Al', 4: 'Na', 5: 'Ca',
                                        6: 'Mg', 7: 'K', 8: 'Fe', 9: 'H', 10: 'Cu'}
                            elements.append(type_map.get(atom_type, f'Type{atom_type}'))

                    if elements:
                        unique_elements = list(set(elements))
                        sample_counts = {elem: elements.count(elem) for elem in unique_elements}
                        scale_factor = num_atoms / len(elements) if len(elements) > 0 else 1

                        for elem, count in sample_counts.items():
                            element_info[elem] = int(count * scale_factor)

                    break
                i += 1

            total_frames = content.count('ITEM: TIMESTEP')

            st.success(f"📁 **LAMMPS Trajectory Information:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Frames", total_frames)
                if num_atoms:
                    st.metric("Atoms per Frame", num_atoms)

            with col2:
                if element_info:
                    st.write("**Element Composition (estimated):**")
                    for element, count in sorted(element_info.items()):
                        st.write(f"• {element}: {count} atoms")
                else:
                    st.write("**Element info:** Not parsed")

            with col3:
                st.write("**Frame Information:**")
                if first_timestep is not None:
                    st.write(f"• **First frame:** Timestep {first_timestep}")
                else:
                    st.write("• **First frame:** Not found")

                if second_timestep is not None:
                    st.write(f"• **Second frame:** Timestep {second_timestep}")

                    # Calculate time step if both are available
                    if first_timestep is not None:
                        dt = second_timestep - first_timestep
                        st.write(f"• **Δt:** {dt} timesteps")
                else:
                    st.write("• **Second frame:** Not found")

                coord_types = []
                if 'x y z' in content.lower():
                    coord_types.append("Cartesian")
                if 'xu yu zu' in content.lower():
                    coord_types.append("Unwrapped")
                if 'xs ys zs' in content.lower():
                    coord_types.append("Scaled")

                coord_str = ", ".join(coord_types) if coord_types else "Unknown"
                st.write(f"• **Coordinates:** {coord_str}")

    except Exception as e:
        st.warning(f"Could not parse trajectory file information: {str(e)}")

    finally:
        uploaded_file.seek(0)


def parse_lammps_trajectory(file_content):
    frames = []

    if isinstance(file_content, bytes):
        try:
            content = file_content.decode('utf-8')
        except UnicodeDecodeError:
            content = file_content.decode('latin-1')
    else:
        content = file_content

    st.info(f"DEBUG: Content length: {len(content)} characters")
    st.info(f"DEBUG: First 200 chars: {repr(content[:200])}")

    success = False

    try:
        import io
        from ase.io import read as ase_read

        st.info("Attempting standard ASE readers...")

        read_methods = [
            {'format': 'lammps-dump', 'description': 'Standard LAMMPS dump format'},
            {'format': 'lammps-dump-text', 'description': 'LAMMPS dump text format'},
            {'format': None, 'description': 'Automatic format detection'}
        ]

        for method in read_methods:
            if success:
                break

            try:
                if method['format'] in ['lammps-dump', 'lammps-dump-text']:
                    temp_bytes = io.BytesIO()
                    temp_bytes.write(content.encode('utf-8'))
                    temp_bytes.seek(0)
                    frames = ase_read(temp_bytes, index=':', format=method['format'])
                else:
                    raw_bytes = content.encode('utf-8')
                    temp_bytes = io.BytesIO(raw_bytes)
                    frames = ase_read(temp_bytes, index=':')

                if frames and len(frames) > 0:
                    success = True
                    st.success(f"Successfully read using ASE {method['description']}")
                    break

            except Exception as e:
                st.info(f"ASE method {method['description']} failed: {str(e)}")
                continue

    except Exception as e:
        st.info(f"ASE reading failed: {str(e)}")

    if not success:
        try:
            import tempfile
            import uuid
            import os

            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"temp_lammps_{uuid.uuid4().hex}.dump")

            with open(temp_file_path, "w", encoding='utf-8') as f:
                f.write(content)

            for method in read_methods:
                if success:
                    break

                try:
                    if method['format'] is None:
                        frames = ase_read(temp_file_path, index=':')
                    else:
                        frames = ase_read(temp_file_path, index=':', format=method['format'])

                    if frames and len(frames) > 0:
                        success = True
                        st.success(f"Successfully read using ASE {method['description']} from temp file")
                        break

                except Exception as e:
                    st.info(f"Temp file method {method['description']} failed: {str(e)}")

            try:
                os.remove(temp_file_path)
            except:
                pass

        except Exception as e:
            st.info(f"Temp file method failed: {str(e)}")

    if not success:
        st.info("Using custom robust parser...")
        frames = parse_lammps_dump_robust(content)

        if frames and len(frames) > 0:
            success = True
            st.success(f"Successfully read using robust custom parser")

    if not success or not frames:
        raise Exception("Could not parse LAMMPS trajectory file with any available method")

    return frames


def parse_lammps_dump_robust(content):
    from ase import Atoms
    import numpy as np

    frames = []
    lines = content.splitlines()

    st.info(f"DEBUG: Total lines to parse: {len(lines)}")
    if len(lines) > 0:
        st.info(f"DEBUG: First 5 lines: {lines[:5]}")
    else:
        st.error("DEBUG: Content has no lines!")
        return frames

    i = 0
    frame_count = 0

    while i < len(lines):
        if 'ITEM: TIMESTEP' in lines[i]:
            try:
                st.info(f"Found frame {frame_count + 1} at line {i}")

                i += 1
                if i >= len(lines):
                    break
                timestep = int(lines[i].strip())
                i += 1

                while i < len(lines) and 'ITEM: NUMBER OF ATOMS' not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break

                i += 1
                num_atoms = int(lines[i].strip())
                st.info(f"Frame {frame_count + 1}: {num_atoms} atoms at timestep {timestep}")
                i += 1

                while i < len(lines) and 'ITEM: ATOMS' not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break

                header_line = lines[i]
                header = header_line.replace('ITEM: ATOMS', '').strip().split()
                st.info(f"Header columns: {header}")
                i += 1

                try:
                    element_idx = header.index('element') if 'element' in header else -1
                    x_idx = header.index('x') if 'x' in header else -1
                    y_idx = header.index('y') if 'y' in header else -1
                    z_idx = header.index('z') if 'z' in header else -1

                    if x_idx == -1 or y_idx == -1 or z_idx == -1:
                        st.error(f"Could not find coordinate columns in {header}")
                        i += num_atoms
                        continue

                    st.info(f"Using columns - element: {element_idx}, x: {x_idx}, y: {y_idx}, z: {z_idx}")

                except ValueError as e:
                    st.error(f"Error finding columns: {e}")
                    i += num_atoms
                    continue

                positions = []
                symbols = []
                atoms_processed = 0

                for j in range(num_atoms):
                    if i + j >= len(lines):
                        break

                    line = lines[i + j].strip()
                    if not line:
                        continue

                    values = line.split()
                    if len(values) < len(header):
                        continue

                    try:
                        x = float(values[x_idx])
                        y = float(values[y_idx])
                        z = float(values[z_idx])
                        positions.append([x, y, z])

                        if element_idx >= 0:
                            element = values[element_idx]
                            symbols.append(element)
                        else:
                            symbols.append('X')

                        atoms_processed += 1

                    except (ValueError, IndexError) as e:
                        continue

                i += num_atoms

                if atoms_processed > 0:
                    atoms = Atoms(symbols=symbols, positions=positions)
                    frames.append(atoms)
                    frame_count += 1

                    st.success(f"Frame {frame_count}: extracted {atoms_processed}/{num_atoms} atoms")

                    unique_elements = list(set(symbols))
                    element_counts = {elem: symbols.count(elem) for elem in unique_elements}
                    st.info(f"Elements: {element_counts}")

            except Exception as e:
                st.error(f"Error parsing frame {frame_count + 1}: {str(e)}")
                i += 1
        else:
            i += 1

    st.info(f"Custom parser extracted {len(frames)} frames total")
    return frames


def process_trajectory_file_debug(uploaded_file, file_type, frame_sampling, manual_lattice=None):
    if file_type == "xyz":
        uploaded_file.seek(0)
        file_content = uploaded_file.read().decode('utf-8')
        frames, has_lattice_in_file, _ = parse_cp2k_xyz_with_headers(file_content, manual_lattice)

    elif file_type == "lammps":
        st.info(f"Processing LAMMPS trajectory: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size} bytes")

        file_content_bytes = uploaded_file.read()
        st.info(f"Read {len(file_content_bytes)} bytes from file")

        uploaded_file.seek(0)

        with st.status("Reading LAMMPS trajectory file..."):
            try:
                content_sample = file_content_bytes[:2048]
                try:
                    sample_text = content_sample.decode('utf-8')
                except UnicodeDecodeError:
                    sample_text = content_sample.decode('latin-1')

                if "ITEM: TIMESTEP" in sample_text:
                    st.success("Detected standard LAMMPS dump format")
                elif "ITEM: NUMBER OF ATOMS" in sample_text:
                    st.success("Detected LAMMPS dump format with atom counts")
                else:
                    st.warning("Could not detect standard LAMMPS format markers")

                import io
                from ase.io import read as ase_read

                frames = []
                read_methods = [
                    {'format': 'lammps-dump', 'description': 'Standard LAMMPS dump format'},
                    {'format': 'lammps-dump-text', 'description': 'LAMMPS dump text format'},
                    {'format': None, 'description': 'Automatic format detection'}
                ]

                success = False

                for method in read_methods:
                    if success:
                        break

                    try:
                        uploaded_file.seek(0)
                        st.info(f"Trying to read using {method['description']}...")

                        if method['format'] == 'lammps-dump' or method['format'] == 'lammps-dump-text':
                            raw_bytes = uploaded_file.read()
                            try:
                                text_content = raw_bytes.decode('utf-8')
                            except UnicodeDecodeError:
                                text_content = raw_bytes.decode('latin-1')

                            temp_bytes = io.BytesIO()
                            temp_bytes.write(text_content.encode('utf-8'))
                            temp_bytes.seek(0)

                            frames = ase_read(temp_bytes, index=':', format=method['format'])
                        elif method['format'] is None:
                            uploaded_file.seek(0)
                            raw_data = uploaded_file.read()
                            temp_bytes = io.BytesIO(raw_data)
                            frames = ase_read(temp_bytes, index=':')

                        if frames and len(frames) > 0:
                            success = True
                            st.success(f"Successfully read using {method['description']}")
                            break
                    except Exception as e:
                        st.warning(f"Failed with {method['description']}: {str(e)}")

                if not success:
                    st.warning("Direct memory reading failed. Trying with temporary file...")
                    import os
                    import tempfile
                    import uuid

                    temp_dir = tempfile.gettempdir()
                    temp_file_path = os.path.join(temp_dir, f"temp_lammps_{uuid.uuid4().hex}.dump")

                    uploaded_file.seek(0)
                    file_data = uploaded_file.read()

                    with open(temp_file_path, "wb") as f:
                        f.write(file_data)
                    st.info(f"Saved temporary file: {temp_file_path}")

                    for method in read_methods:
                        if success:
                            break

                        try:
                            st.info(f"Trying {method['description']} from temp file...")

                            if method['format'] is None:
                                frames = ase_read(temp_file_path, index=':')
                            else:
                                frames = ase_read(temp_file_path, index=':', format=method['format'])

                            if frames and len(frames) > 0:
                                success = True
                                st.success(f"Successfully read using {method['description']} from temp file")
                                break
                        except Exception as e:
                            st.warning(f"Failed with {method['description']} from temp file: {str(e)}")

                    try:
                        os.remove(temp_file_path)
                        st.info("Temporary file removed")
                    except Exception as clean_err:
                        st.warning(f"Could not remove temporary file: {str(clean_err)}")

                if not success:
                    st.warning("All standard methods failed. Attempting custom parsing...")
                    uploaded_file.seek(0)
                    file_content_bytes = uploaded_file.read()
                    try:
                        text_content = file_content_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        text_content = file_content_bytes.decode('latin-1')

                    frames = parse_lammps_dump_from_string(text_content)

                    if frames and len(frames) > 0:
                        success = True
                        st.success(f"Successfully read using custom parser")

                if not frames or len(frames) == 0:
                    raise Exception("Could not extract any frames from the trajectory file")

            except Exception as e:
                st.error(f"Error reading LAMMPS trajectory file: {str(e)}")
                return None, None

        has_lattice_in_file = False

    if not frames:
        st.error("No valid frames found in the trajectory file")
        return None, None

    total_frames = len(frames)
    selected_frames = frames[::frame_sampling]
    frame_indices = [i * frame_sampling for i in range(len(selected_frames))]

    st.success(
        f"Processed {total_frames} frames → Using {len(selected_frames)} frames (sampling: every {frame_sampling})")

    return selected_frames, frame_indices


def parse_lammps_dump_from_string(content):
    from ase import Atoms
    import numpy as np

    frames = []
    lines = content.splitlines()

    i = 0
    while i < len(lines):
        if 'ITEM: TIMESTEP' in lines[i]:
            try:
                i += 1
                if i >= len(lines):
                    break
                timestep = int(lines[i].strip())
                i += 1

                while i < len(lines) and 'ITEM: NUMBER OF ATOMS' not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break

                i += 1
                num_atoms = int(lines[i].strip())
                i += 1

                while i < len(lines) and 'ITEM: ATOMS' not in lines[i]:
                    i += 1
                if i >= len(lines):
                    break

                header = lines[i].replace('ITEM: ATOMS', '').strip().split()
                i += 1

                positions = np.zeros((num_atoms, 3))
                symbols = []

                for j in range(num_atoms):
                    if i + j >= len(lines):
                        break
                    values = lines[i + j].strip().split()
                    if len(values) < len(header):
                        continue

                    x_idx = header.index('x') if 'x' in header else (header.index('xu') if 'xu' in header else -1)
                    y_idx = header.index('y') if 'y' in header else (header.index('yu') if 'yu' in header else -1)
                    z_idx = header.index('z') if 'z' in header else (header.index('zu') if 'zu' in header else -1)

                    type_idx = header.index('type') if 'type' in header else -1
                    element_idx = header.index('element') if 'element' in header else -1

                    if x_idx >= 0 and y_idx >= 0 and z_idx >= 0:
                        positions[j] = [float(values[x_idx]), float(values[y_idx]), float(values[z_idx])]

                    if element_idx >= 0:
                        symbols.append(values[element_idx])
                    elif type_idx >= 0:
                        type_num = int(values[type_idx])
                        element_map = {1: 'Si', 2: 'O', 3: 'Al', 4: 'Na', 5: 'Ca', 6: 'Mg', 7: 'K', 8: 'Fe',
                                       9: 'H', 10: 'Cu', 11: 'N', 12: 'P', 13: 'S', 14: 'Cl', 15: 'Li'}
                        symbols.append(element_map.get(type_num, f'X{type_num}'))
                    else:
                        symbols.append('X')

                i += num_atoms

                if len(symbols) == num_atoms:
                    atoms = Atoms(symbols=symbols, positions=positions)
                    frames.append(atoms)

            except Exception as e:
                st.warning(f"Error parsing frame: {str(e)}")
                i += 1
        else:
            i += 1

    return frames


def integrate_lammps_support():
    trajectory_type = st.sidebar.radio(
        "Trajectory File Type",
        ["CP2K XYZ", "LAMMPS Trajectory"],
        index=0,
        help="Select the type of trajectory file you want to upload"
    )

    if trajectory_type == "CP2K XYZ":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CP2K XYZ Trajectory File",
            type=["xyz"],
            help="Upload a CP2K XYZ trajectory file with lattice parameters"
        )
        file_type = "xyz"

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload LAMMPS Trajectory File",
            type=["dump", "lammpstrj", "lmp", "txt"],
            help="Upload a LAMMPS trajectory file"
        )
        file_type = "lammps"

    return uploaded_file, file_type, trajectory_type


def process_trajectory_file(uploaded_file, file_type, frame_sampling, manual_lattice=None):
    return process_trajectory_file_debug(uploaded_file, file_type, frame_sampling, manual_lattice)


def create_animated_prdf_plot(distances, valid_prdf, comb, frame_indices, smoothing_params,
                              hex_color, line_style, animation_speed):
    fig = go.Figure()

    smoothed_prdf = []
    original_prdf = valid_prdf.copy()

    for frame_data in valid_prdf:
        if smoothing_params.get("enabled", False):
            try:
                smoothed_frame = smooth_prdf_data(
                    distances, frame_data,
                    method=smoothing_params["method"],
                    **{k: v for k, v in smoothing_params.items()
                       if k not in ["enabled", "method"]}
                )
                smoothed_prdf.append(smoothed_frame)
            except Exception as e:
                smoothed_prdf.append(frame_data)
        else:
            smoothed_prdf.append(frame_data)

    plot_data = smoothed_prdf if smoothing_params.get("enabled", False) else original_prdf

    fig.add_trace(go.Scatter(
        x=distances,
        y=plot_data[0],
        mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
        name=f"{comb[0]}-{comb[1]}" + (" (Smoothed)" if smoothing_params.get("enabled", False) else ""),
        line=dict(color=hex_color, width=2),
        marker=dict(size=8) if line_style == "Lines + Markers" else dict()
    ))

    frames = []
    for i, frame_data in enumerate(plot_data):
        frame = go.Frame(
            data=[go.Scatter(
                x=distances,
                y=frame_data,
                mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
                line=dict(color=hex_color, width=2),
                marker=dict(size=8) if line_style == "Lines + Markers" else dict()
            )],
            name=f"frame_{i}"
        )
        frames.append(frame)

    fig.frames = frames

    updatemenus = [dict(
        type="buttons",
        direction="right",
        x=0.1, y=-0.1,
        buttons=[
            dict(label="▶️ Play", method="animate",
                 args=[None, {"frame": {"duration": int(animation_speed * 1000),
                                        "redraw": True},
                              "fromcurrent": True, "mode": "immediate"}]),
            dict(label="⏹️ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": True},
                                "mode": "immediate", "transition": {"duration": 0}}])
        ]
    )]

    sliders = [dict(
        active=0,
        currentvalue=dict(font=dict(size=16), prefix="Frame: ", visible=True),
        pad=dict(b=10, t=50),
        len=0.9, x=0.1, y=0,
        steps=[dict(
            method="animate",
            args=[[f"frame_{k}"], {"frame": {"duration": 100, "redraw": True},
                                   "mode": "immediate", "transition": {"duration": 0}}],
            label=f"{frame_indices[k]}"
        ) for k in range(len(plot_data))]
    )]

    all_y_values = [y for data in plot_data for y in data]
    max_y = max(all_y_values) * 1.1 if all_y_values else 1.0

    title_text = f"PRDF: {comb[0]}-{comb[1]} Animation"
    if smoothing_params.get("enabled", False):
        method_name = {
            "moving_average": "Moving Avg",
            "gaussian": "Gaussian",
            "savgol": "Savitzky-Golay",
            "spline": "Spline"
        }.get(smoothing_params["method"], "Smoothed")
        title_text += f" ({method_name})"

    fig.update_layout(
        title={'text': title_text, 'font': dict(size=20, color="black")},
        xaxis_title={'text': "Distance (Å)", 'font': dict(size=20, color="black")},
        yaxis_title={'text': "PRDF Intensity", 'font': dict(size=20, color="black")},
        updatemenus=updatemenus,
        sliders=sliders,
        font=dict(size=20, color="black"),
        xaxis=dict(tickfont=dict(size=20, color="black")),
        yaxis=dict(tickfont=dict(size=20, color="black"), range=[0, max_y])
    )

    return fig


def create_animated_global_rdf_plot(global_bins, global_rdf_list, frame_indices,
                                    smoothing_params, hex_color_global, line_style, animation_speed):
    fig_global = go.Figure()

    smoothed_global_rdf = []

    for global_dict in global_rdf_list:
        frame_values = [global_dict.get(b, 0) for b in global_bins]

        if smoothing_params.get("enabled", False):
            try:
                smoothed_frame = smooth_prdf_data(
                    global_bins, frame_values,
                    method=smoothing_params["method"],
                    **{k: v for k, v in smoothing_params.items()
                       if k not in ["enabled", "method"]}
                )
                smoothed_global_rdf.append(smoothed_frame)
            except:
                smoothed_global_rdf.append(frame_values)
        else:
            smoothed_global_rdf.append(frame_values)

    initial_values = smoothed_global_rdf[0]

    fig_global.add_trace(go.Scatter(
        x=global_bins, y=initial_values,
        mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
        name="Global RDF" + (" (Smoothed)" if smoothing_params.get("enabled", False) else ""),
        line=dict(color=hex_color_global, width=2),
        marker=dict(size=8) if line_style == "Lines + Markers" else dict()
    ))

    frames = []
    for i, frame_values in enumerate(smoothed_global_rdf):
        frame = go.Frame(
            data=[go.Scatter(
                x=global_bins, y=frame_values,
                mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
                line=dict(color=hex_color_global, width=2),
                marker=dict(size=8) if line_style == "Lines + Markers" else dict()
            )],
            name=f"frame_{i}"
        )
        frames.append(frame)

    fig_global.frames = frames

    updatemenus = [dict(
        type="buttons", direction="right", x=0.1, y=-0.1,
        buttons=[
            dict(label="▶️ Play", method="animate",
                 args=[None,
                       {"frame": {"duration": int(animation_speed * 1000), "redraw": True},
                        "fromcurrent": True, "mode": "immediate"}]),
            dict(label="⏹️ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": True},
                                "mode": "immediate", "transition": {"duration": 0}}])
        ]
    )]

    sliders = [dict(
        active=0,
        currentvalue=dict(font=dict(size=16), prefix="Frame: ", visible=True),
        pad=dict(b=10, t=50), len=0.9, x=0.1, y=0,
        steps=[dict(
            method="animate",
            args=[[f"frame_{k}"], {"frame": {"duration": 100, "redraw": True},
                                   "mode": "immediate", "transition": {"duration": 0}}],
            label=f"{frame_indices[k]}"
        ) for k in range(len(smoothed_global_rdf))]
    )]

    all_values = [val for frame_vals in smoothed_global_rdf for val in frame_vals]
    max_y = max(all_values) * 1.1 if all_values else 1.0

    title_text = "Global RDF Animation"
    if smoothing_params.get("enabled", False):
        method_name = {
            "moving_average": "Moving Avg",
            "gaussian": "Gaussian",
            "savgol": "Savitzky-Golay",
            "spline": "Spline"
        }.get(smoothing_params["method"], "Smoothed")
        title_text += f" ({method_name})"

    fig_global.update_layout(
        title={'text': title_text, 'font': dict(size=20, color="black")},
        xaxis_title={'text': "Distance (Å)", 'font': dict(size=20, color="black")},
        yaxis_title={'text': "Total RDF Intensity", 'font': dict(size=20, color="black")},
        updatemenus=updatemenus, sliders=sliders,
        font=dict(size=20, color="black"),
        xaxis=dict(tickfont=dict(size=20, color="black")),
        yaxis=dict(tickfont=dict(size=20, color="black"), range=[0, max_y])
    )

    return fig_global


def display_animation_smoothing_info(smoothing_params):
    if smoothing_params.get("enabled", False):
        method_names = {
            "moving_average": "Moving Average",
            "gaussian": "Gaussian Filter",
            "savgol": "Savitzky-Golay",
            "spline": "Spline Interpolation"
        }

        method_name = method_names.get(smoothing_params["method"], "Unknown")

        st.info(f"🎬 Animation smoothing: **{method_name}** applied to each individual frame")

        if smoothing_params["method"] == "moving_average":
            st.caption(f"Window size: {smoothing_params.get('window_size', 'N/A')} points")
        elif smoothing_params["method"] == "gaussian":
            st.caption(f"Gaussian sigma: {smoothing_params.get('sigma', 'N/A'):.1f}")
        elif smoothing_params["method"] == "savgol":
            st.caption(
                f"Window: {smoothing_params.get('window_length', 'N/A')}, Polynomial order: {smoothing_params.get('polyorder', 'N/A')}")
    else:
        st.caption("🎬 Animation showing original (unsmoothed) PRDF data")


def smooth_prdf_data(distances, prdf_values, method="moving_average", **kwargs):
    prdf_array = np.array(prdf_values)

    if method == "moving_average":
        window_size = kwargs.get('window_size', 5)
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        smoothed = np.convolve(prdf_array, np.ones(window_size) / window_size, mode='same')
        return smoothed

    elif method == "gaussian":
        sigma = kwargs.get('sigma', 1.0)
        smoothed = ndimage.gaussian_filter1d(prdf_array, sigma=sigma)
        return smoothed

    elif method == "savgol":
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 3)

        if window_length % 2 == 0:
            window_length += 1
        if window_length <= polyorder:
            window_length = polyorder + 2
        if window_length > len(prdf_array):
            window_length = len(prdf_array) if len(prdf_array) % 2 == 1 else len(prdf_array) - 1

        smoothed = savgol_filter(prdf_array, window_length, polyorder)
        return smoothed

    elif method == "spline":
        smoothing_factor = kwargs.get('smoothing_factor', None)
        spline_degree = kwargs.get('spline_degree', 3)

        spline = UnivariateSpline(distances, prdf_array, s=smoothing_factor, k=spline_degree)
        smoothed = spline(distances)
        return smoothed

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def add_smoothing_controls():
    st.sidebar.subheader("📈 PRDF Smoothing Options")

    enable_smoothing = st.sidebar.checkbox("Enable PRDF Smoothing",
                                           help="Apply smoothing to reduce noise in PRDF curves")

    smoothing_params = {"enabled": enable_smoothing}

    if enable_smoothing:
        col1, col2 = st.sidebar.columns(2)

        with col1:
            smoothing_method = st.selectbox(
                "Smoothing Method",
                ["moving_average", "gaussian", "savgol", "spline"],
                format_func=lambda x: {
                    "moving_average": "Moving Average",
                    "gaussian": "Gaussian Filter",
                    "savgol": "Savitzky-Golay",
                    "spline": "Spline Interpolation"
                }[x],
                help="Choose the smoothing algorithm",
                index=1
            )

            smoothing_params["method"] = smoothing_method

        with col2:
            smoothing_intensity = st.slider(
                "Smoothing Intensity",
                min_value=1,
                max_value=10,
                value=4,
                help="Higher values = more smoothing"
            )

        if smoothing_method == "moving_average":
            # Convert intensity to window size (1-10 -> 3-21)
            window_size = 2 * smoothing_intensity + 1
            smoothing_params["window_size"] = window_size
            st.sidebar.info(f"Moving average window size: {window_size} points")

        elif smoothing_method == "gaussian":
            # Convert intensity to sigma (1-10 -> 0.5-5.0)
            sigma = smoothing_intensity * 0.5
            smoothing_params["sigma"] = sigma
            st.sidebar.info(f"Gaussian filter sigma: {sigma:.1f}")

        elif smoothing_method == "savgol":
            # Convert intensity to window length and set polyorder
            window_length = 2 * smoothing_intensity + 1
            polyorder = min(3, window_length - 2)  # Ensure polyorder < window_length
            smoothing_params["window_length"] = window_length
            smoothing_params["polyorder"] = polyorder
            st.sidebar.info(f"Savitzky-Golay: window={window_length}, polynomial order={polyorder}")

        elif smoothing_method == "spline":
            if smoothing_intensity <= 2:
                smoothing_factor = None
            else:
                smoothing_factor = smoothing_intensity * 10

            spline_degree = st.sidebar.selectbox("Spline Degree", [1, 2, 3, 4, 5], index=2)
            smoothing_params["smoothing_factor"] = smoothing_factor
            smoothing_params["spline_degree"] = spline_degree

            if smoothing_factor is None:
                st.sidebar.info("Automatic smoothing factor (cross-validation)")
            else:
                st.sidebar.info(f"Spline smoothing factor: {smoothing_factor}")

    return smoothing_params


def create_smoothed_plot(distances, prdf_data, comb, smoothing_params, hex_color,
                         title_str, line_style, multi_structures=False, prdf_std=None):
    fig = go.Figure()
    show_original = smoothing_params.get("enabled", False)

    if show_original:
        fig.add_trace(go.Scatter(
            x=distances,
            y=prdf_data,
            mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
            name=f"{comb[0]}-{comb[1]} (Original)",
            line=dict(color=hex_color, width=1, dash='dot'),
            marker=dict(size=4) if line_style == "Lines + Markers" else dict(),
            opacity=0.5
        ))
    if smoothing_params.get("enabled", False):
        try:
            smoothed_data = smooth_prdf_data(
                distances, prdf_data,
                method=smoothing_params["method"],
                **{k: v for k, v in smoothing_params.items()
                   if k not in ["enabled", "method"]}
            )

            # Main smoothed trace
            fig.add_trace(go.Scatter(
                x=distances,
                y=smoothed_data,
                mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
                name=f"{comb[0]}-{comb[1]} (Smoothed)",
                line=dict(color=hex_color, width=2),
                marker=dict(size=6) if line_style == "Lines + Markers" else dict()
            ))

            # smoothed error bands if available
            if multi_structures and prdf_std is not None:
                try:
                    smoothed_std = smooth_prdf_data(
                        distances, prdf_std,
                        method=smoothing_params["method"],
                        **{k: v for k, v in smoothing_params.items()
                           if k not in ["enabled", "method"]}
                    )

                    fig.add_trace(go.Scatter(
                        x=distances, y=smoothed_data + smoothed_std,
                        mode='lines', line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=distances, y=np.maximum(0, smoothed_data - smoothed_std),
                        mode='lines', line=dict(width=0),
                        fillcolor='rgba(100,100,100,0.2)', fill='tonexty', showlegend=False
                    ))
                except:
                    pass

        except Exception as e:
            st.warning(f"Smoothing failed for {comb[0]}-{comb[1]}: {str(e)}. Showing original data.")
            fig.add_trace(go.Scatter(
                x=distances,
                y=prdf_data,
                mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
                name=f"{comb[0]}-{comb[1]}",
                line=dict(color=hex_color, width=2),
                marker=dict(size=6) if line_style == "Lines + Markers" else dict()
            ))
    else:
        # No smoothing
        fig.add_trace(go.Scatter(
            x=distances,
            y=prdf_data,
            mode='lines+markers' if line_style == "Lines + Markers" else 'lines',
            name=f"{comb[0]}-{comb[1]}",
            line=dict(color=hex_color, width=2),
            marker=dict(size=6) if line_style == "Lines + Markers" else dict()
        ))

        if multi_structures and prdf_std is not None:
            fig.add_trace(go.Scatter(
                x=distances, y=prdf_data + prdf_std,
                mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=distances, y=np.maximum(0, prdf_data - prdf_std),
                mode='lines', line=dict(width=0),
                fillcolor='rgba(100,100,100,0.2)', fill='tonexty', showlegend=False
            ))

    return fig



components.html(
    """
    <head>
        <meta name="description" content="XRDlicious submodule:  Calculate PRDF from XYZ or LAMMPS trajectories">
    </head>
    """,
    height=0,
)

st.markdown(
    "#### XRDlicious submodule:  PRDF Calculator from CP2K XYZ / LAMMPS Trajectories ")
col1_header, col2_header = st.columns([1.25, 1])

with col2_header:
    st.info(
        "🌀 Developed by [IMPLANT team](https://implant.fs.cvut.cz/). 📺 [Quick tutorial HERE.](https://www.youtube.com/watch?v=7ZgQ0fnR8dQ&ab_channel=Implantgroup)"
    )
with col1_header:
    st.info("Visit the main [XRDlicious](http://xrdlicious.com) page")
blue_divider()


if 'calc_rdf' not in st.session_state:
    st.session_state.calc_rdf = False
if 'do_calculation' not in st.session_state:
    st.session_state.do_calculation = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}
if 'display_mode' not in st.session_state:
    st.session_state.display_mode = "Average PRDF across frames"
if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 0.5
if 'line_style' not in st.session_state:
    st.session_state.line_style = "Lines + Markers"
if 'lattice_params' not in st.session_state:
    st.session_state.lattice_params = None
if 'use_manual_lattice' not in st.session_state:
    st.session_state.use_manual_lattice = False


def trigger_calculation():
    st.session_state.calc_rdf = True
    st.session_state.do_calculation = True


def update_display_mode():
    st.session_state.display_mode = st.session_state.display_mode_radio


def parse_cp2k_xyz(file_content, manual_lattice=None):
    lines = file_content.strip().split('\n')
    frames = []
    i = 0
    has_lattice_in_file = "Lattice=" in file_content

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        try:
            num_atoms = int(lines[i].strip())
            i += 1

            if i >= len(lines):
                break

            header_line = lines[i].strip()
            i += 1

            cell_matrix = None
            if has_lattice_in_file and "Lattice=" in header_line:
                lattice_match = re.search(r'Lattice="([^"]*)"', header_line)
                if lattice_match:
                    lattice_str = lattice_match.group(1)
                    lattice_values = [float(x) for x in lattice_str.split()]
                    if len(lattice_values) == 9:
                        cell_matrix = np.array(lattice_values).reshape(3, 3)
            elif manual_lattice is not None:
                cell_matrix = manual_lattice

            symbols = []
            positions = []

            for j in range(num_atoms):
                if i + j >= len(lines):
                    break
                parts = lines[i + j].strip().split()
                if len(parts) >= 4:
                    symbols.append(parts[0])
                    positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if len(symbols) == num_atoms and len(positions) == num_atoms:
                atoms = Atoms(symbols=symbols, positions=positions)
                if cell_matrix is not None:
                    atoms.set_cell(cell_matrix)
                    atoms.set_pbc([True, True, True])
                    atoms.wrap()
                frames.append(atoms)

            i += num_atoms

        except (ValueError, IndexError):
            i += 1

    return frames, has_lattice_in_file


uploaded_file, file_type, traj_file = integrate_lammps_support()

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    display_trajectory_info(uploaded_file, file_type)
    blue_divider()
    file_content = uploaded_file.read().decode('utf-8')
    has_lattice = "Lattice=" in file_content
    if file_type != "lammps":
        st.subheader("📥 Append Lattice Parameters to the XYZ Trajectory (For OVITO Visualization)")
        st.write("Download your trajectory with lattice parameters and wrapped coordinates")

        # Create expandable section for download options
        with st.expander("🔧 Configure Enhanced XYZ Export", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                wrap_coords_pre = st.checkbox(
                    "Wrap coordinates into unit cell",
                    value=True,
                    key="wrap_coords_pre",
                    help="Apply periodic boundary conditions and wrap atoms into the unit cell"
                )

                frame_sampling_pre = st.slider(
                    "Frame sampling for export",
                    min_value=1,
                    max_value=100,
                    value=1,
                    key="frame_sampling_pre",
                    help="Export every Nth frame (1 = all frames)"
                )

            with col2:
                if not has_lattice and traj_file == "CP2K XYZ":
                    lattice_source_pre = st.radio(
                        "Lattice parameter source:",
                        ["Manual definition", "From lattice file", "Both (file priority)"],
                        key="lattice_source_pre",
                        help="Choose which lattice parameters to use"
                    )
                else:
                    lattice_source_pre = "From trajectory file"
                    st.info("✅ Lattice parameters will be taken from the trajectory file")

            # Manual lattice input for pre-calculation download
            if not has_lattice and traj_file == "CP2K XYZ" and lattice_source_pre in ["Manual definition",
                                                                                      "Both (file priority)"]:
                st.write("**Quick Lattice Definition:**")

                lattice_type = st.selectbox(
                    "Lattice type:",
                    ["Custom", "Cubic", "Orthorhombic"],
                    key="lattice_type_pre"
                )

                if lattice_type == "Cubic":
                    a_pre = st.number_input("Lattice parameter a (Å):", value=10.0, key="a_pre")
                    lattice_matrix_pre = np.array([[a_pre, 0, 0], [0, a_pre, 0], [0, 0, a_pre]])

                elif lattice_type == "Orthorhombic":
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        a_pre = st.number_input("a (Å):", value=10.0, key="a_ortho_pre")
                    with col_b:
                        b_pre = st.number_input("b (Å):", value=10.0, key="b_ortho_pre")
                    with col_c:
                        c_pre = st.number_input("c (Å):", value=10.0, key="c_ortho_pre")
                    lattice_matrix_pre = np.array([[a_pre, 0, 0], [0, b_pre, 0], [0, 0, c_pre]])

                else:  # Custom
                    st.write("Enter 3x3 lattice matrix:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        a1_pre = st.number_input("a₁", value=10.0, key="a1_pre")
                        a2_pre = st.number_input("a₂", value=0.0, key="a2_pre")
                        a3_pre = st.number_input("a₃", value=0.0, key="a3_pre")
                    with col2:
                        b1_pre = st.number_input("b₁", value=0.0, key="b1_pre")
                        b2_pre = st.number_input("b₂", value=10.0, key="b2_pre")
                        b3_pre = st.number_input("b₃", value=0.0, key="b3_pre")
                    with col3:
                        c1_pre = st.number_input("c₁", value=0.0, key="c1_pre")
                        c2_pre = st.number_input("c₂", value=0.0, key="c2_pre")
                        c3_pre = st.number_input("c₃", value=10.0, key="c3_pre")

                    lattice_matrix_pre = np.array([[a1_pre, a2_pre, a3_pre],
                                                   [b1_pre, b2_pre, b3_pre],
                                                   [c1_pre, c2_pre, c3_pre]])
            else:
                lattice_matrix_pre = None

            # Lattice file upload for pre-calculation
            if not has_lattice and traj_file == "CP2K XYZ" and lattice_source_pre in ["From lattice file",
                                                                                      "Both (file priority)"]:
                lattice_file_pre = st.file_uploader(
                    "Upload lattice parameter file:",
                    type=["txt", "dat", "log", "cell"],
                    key="lattice_file_pre",
                    help="Upload CP2K lattice parameter file"
                )

                if lattice_file_pre:
                    try:
                        lattice_content_pre = lattice_file_pre.read().decode('utf-8')
                        lattice_data_pre = parse_lattice_file(lattice_content_pre)

                        if lattice_data_pre:
                            st.success(f"✅ Loaded {len(lattice_data_pre)} lattice parameter sets")

                            # Display lattice info directly without nested expander
                            st.write("**Lattice Parameter File Preview:**")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Steps", len(lattice_data_pre))
                            with col2:
                                steps = sorted(lattice_data_pre.keys())
                                st.write(f"**Step Range:** {steps[0]} - {steps[-1]}")
                            with col3:
                                times = [data['time'] for data in lattice_data_pre.values()]
                                st.write(f"**Time Range:** {min(times):.1f} - {max(times):.1f} fs")

                            # Show first lattice matrix as example (without expander)
                            if st.checkbox("Show first lattice matrix preview", key="show_lattice_preview"):
                                first_step = min(lattice_data_pre.keys())
                                first_matrix = lattice_data_pre[first_step]['matrix']

                                st.write(f"**Step {first_step} (t = {lattice_data_pre[first_step]['time']:.1f} fs):**")
                                st.code(f"""
            a = [{first_matrix[0, 0]:8.3f} {first_matrix[0, 1]:8.3f} {first_matrix[0, 2]:8.3f}]
            b = [{first_matrix[1, 0]:8.3f} {first_matrix[1, 1]:8.3f} {first_matrix[1, 2]:8.3f}]
            c = [{first_matrix[2, 0]:8.3f} {first_matrix[2, 1]:8.3f} {first_matrix[2, 2]:8.3f}]
                                """)

                                cell_volume = np.linalg.det(first_matrix)
                                st.write(f"Volume: {cell_volume:.3f} Ų")

                        else:
                            st.error("Could not parse lattice file")
                            lattice_data_pre = None
                    except Exception as e:
                        st.error(f"Error reading lattice file: {str(e)}")
                        lattice_data_pre = None
                else:
                    lattice_data_pre = None
            else:
                lattice_data_pre = None
            if 'original_headers' not in locals():
                original_headers = None
            # Download button
            if st.button("🚀 Generate Enhanced XYZ File", type="primary", key="generate_enhanced_pre"):
                with st.spinner("Processing trajectory file..."):
                    try:
                        # Parse the original trajectory
                        if file_type == "xyz":
                            uploaded_file.seek(0)
                            file_content = uploaded_file.read().decode('utf-8')
                            original_frames, _, original_headers = parse_cp2k_xyz_with_headers(file_content, None)
                        elif file_type == "lammps":
                            # You'll need to add LAMMPS processing here if needed
                            st.error("LAMMPS trajectory export not yet implemented")
                            st.stop()

                        if not original_frames:
                            st.error("Could not parse trajectory file")
                            st.stop()

                        # Apply frame sampling
                        sampled_frames = original_frames[::frame_sampling_pre]
                        st.info(
                            f"Using {len(sampled_frames)} frames out of {len(original_frames)} (sampling rate: {frame_sampling_pre})")

                        # Determine lattice parameters to use
                        lattice_data_for_export = None
                        manual_lattice_for_export = None

                        if has_lattice:
                            pass
                        elif lattice_source_pre == "Manual definition" and 'lattice_matrix_pre' in locals() and lattice_matrix_pre is not None:
                            manual_lattice_for_export = lattice_matrix_pre
                        elif lattice_source_pre == "From lattice file" and 'lattice_data_pre' in locals() and lattice_data_pre is not None:
                            lattice_data_for_export = lattice_data_pre
                        elif lattice_source_pre == "Both (file priority)":
                            if 'lattice_data_pre' in locals() and lattice_data_pre is not None:
                                lattice_data_for_export = lattice_data_pre
                            elif 'lattice_matrix_pre' in locals() and lattice_matrix_pre is not None:
                                manual_lattice_for_export = lattice_matrix_pre

                        # Wrap coordinates if requested
                        # Wrap coordinates if requested
                        if wrap_coords_pre and (
                                lattice_data_for_export is not None or manual_lattice_for_export is not None or has_lattice):
                            # Use progress version for large files
                            if len(sampled_frames) > 100:
                                processed_frames = wrap_coordinates_with_lattice_progress(
                                    sampled_frames,
                                    lattice_data_for_export,
                                    manual_lattice_for_export,
                                    original_headers[::frame_sampling_pre] if original_headers else None
                                    # Pass sampled headers
                                )
                            else:
                                processed_frames = wrap_coordinates_with_lattice(
                                    sampled_frames,
                                    lattice_data_for_export,
                                    manual_lattice_for_export,
                                    original_headers[::frame_sampling_pre] if original_headers else None
                                    # Pass sampled headers
                                )
                            coord_status = "wrapped"
                        else:
                            processed_frames = sampled_frames
                            coord_status = "original"
                        st.info("Creating XYZ content...")
                        # Create XYZ content (use optimized version)
                        xyz_content = create_xyz_with_lattice_optimized(
                            processed_frames,
                            lattice_data_for_export,
                            manual_lattice_for_export,
                            original_headers[::frame_sampling_pre] if original_headers else None  # Pass sampled headers
                        )
                        # Generate download
                        b64 = base64.b64encode(xyz_content.encode()).decode()
                        original_name = uploaded_file.name.split('.')[0]
                        filename = f"enhanced_{original_name}_frames{len(processed_frames)}_{coord_status}.xyz"

                        href = f'<a href="data:file/xyz;base64,{b64}" download="{filename}" style="text-decoration: none; background-color: #0066cc; color: white; padding: 10px 20px; border-radius: 5px; display: inline-block; margin: 10px 0;">📥 Download {filename}</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Show processing summary
                        st.success("✅ Enhanced XYZ file generated successfully!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Frames processed", len(processed_frames))
                        with col2:
                            st.metric("Coordinates", coord_status.title())
                        with col3:
                            lattice_source_display = "File" if has_lattice else (
                                "Manual" if manual_lattice_for_export is not None else (
                                    "External file" if lattice_data_for_export else "None"
                                )
                            )
                            st.metric("Lattice source", lattice_source_display)

                        # File preview
                        #with st.expander("📋 Preview Enhanced XYZ File (first 20 lines)"):
                        #preview_lines = xyz_content.split('\n')[:20]
                        #st.code('\n'.join(preview_lines))
                        #if len(xyz_content.split('\n')) > 20:
                        #    st.caption(f"... and {len(xyz_content.split('\n')) - 20} more lines")

                    except Exception as e:
                        st.error(f"Error generating enhanced XYZ file: {str(e)}")
                        st.exception(e)  # This will show the full traceback for debugging

        blue_divider()  # Separator before PRDF calculation section

    if not has_lattice and traj_file == "CP2K XYZ":
        st.warning(
            "⚠️ No lattice parameters detected in the XYZ file. Please append them to your trajectory, or **specify"
            " constant lattice parameters that will be used below**. You can also continue without the lattice parameters, "
            "but this could potentially result in calculation errors.")

        use_manual_lattice = st.checkbox("Define lattice parameters manually",
                                         help="Enable this to set custom lattice vectors for PBC")
        st.session_state.use_manual_lattice = use_manual_lattice

        if use_manual_lattice:
            st.subheader("Lattice Parameters")
            lattice_input_mode = st.radio("Input Mode",
                                          ["Lattice vectors (3x3 matrix)", "Cell parameters (a,b,c,α,β,γ)"],
                                          horizontal=True)

            if lattice_input_mode == "Lattice vectors (3x3 matrix)":
                st.write("Enter lattice vectors as 3x3 matrix (Å):")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**a vector:**")
                    a1 = st.number_input("a₁", value=10.0, format="%.3f", key="a1")
                    a2 = st.number_input("a₂", value=0.0, format="%.3f", key="a2")
                    a3 = st.number_input("a₃", value=0.0, format="%.3f", key="a3")

                with col2:
                    st.write("**b vector:**")
                    b1 = st.number_input("b₁", value=0.0, format="%.3f", key="b1")
                    b2 = st.number_input("b₂", value=10.0, format="%.3f", key="b2")
                    b3 = st.number_input("b₃", value=0.0, format="%.3f", key="b3")

                with col3:
                    st.write("**c vector:**")
                    c1 = st.number_input("c₁", value=0.0, format="%.3f", key="c1")
                    c2 = st.number_input("c₂", value=0.0, format="%.3f", key="c2")
                    c3 = st.number_input("c₃", value=10.0, format="%.3f", key="c3")

                lattice_matrix = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])

            else:
                st.write("Enter cell parameters:")
                col1, col2 = st.columns(2)

                with col1:
                    a = st.number_input("a (Å)", value=10.0, format="%.3f")
                    b = st.number_input("b (Å)", value=10.0, format="%.3f")
                    c = st.number_input("c (Å)", value=10.0, format="%.3f")

                with col2:
                    alpha = st.number_input("α (degrees)", value=90.0, format="%.1f")
                    beta = st.number_input("β (degrees)", value=90.0, format="%.1f")
                    gamma = st.number_input("γ (degrees)", value=90.0, format="%.1f")

                alpha_rad = np.radians(alpha)
                beta_rad = np.radians(beta)
                gamma_rad = np.radians(gamma)

                cos_alpha = np.cos(alpha_rad)
                cos_beta = np.cos(beta_rad)
                cos_gamma = np.cos(gamma_rad)
                sin_gamma = np.sin(gamma_rad)

                lattice_matrix = np.array([
                    [a, 0, 0],
                    [b * cos_gamma, b * sin_gamma, 0],
                    [c * cos_beta, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                     c * np.sqrt(
                         1 - cos_alpha ** 2 - cos_beta ** 2 - cos_gamma ** 2 + 2 * cos_alpha * cos_beta * cos_gamma) / sin_gamma]
                ])

            st.session_state.lattice_params = lattice_matrix

            with st.expander("Preview Lattice Matrix"):
                st.write("Lattice matrix (Å):")
                st.code(f"""
a = [{lattice_matrix[0, 0]:8.3f} {lattice_matrix[0, 1]:8.3f} {lattice_matrix[0, 2]:8.3f}]
b = [{lattice_matrix[1, 0]:8.3f} {lattice_matrix[1, 1]:8.3f} {lattice_matrix[1, 2]:8.3f}]
c = [{lattice_matrix[2, 0]:8.3f} {lattice_matrix[2, 1]:8.3f} {lattice_matrix[2, 2]:8.3f}]
                """)

                cell_volume = np.linalg.det(lattice_matrix)
                st.write(f"Cell volume: {cell_volume:.3f} Ų")
        else:
            st.session_state.lattice_params = None
    else:
        if traj_file == "CP2K XYZ":
            st.success("✅ Lattice parameters found in XYZ file")
            st.session_state.use_manual_lattice = False
            st.session_state.lattice_params = None

    col1, col2 = st.columns(2)

    with col1:
        st.sidebar.subheader("📈 Choose Average PRDF or Single PRDFs for Animation")
        # plot_display_mode = st.sidebar.radio(
        #    "Plot Display Mode",
        #    ["Separate plots for each pair", "Combined plot with all pairs"],
        #    index=0,
        #    horizontal=True
        # )
    display_mode = st.sidebar.radio(
        "Display Mode",
        ["Average PRDF across frames", "Individual frame PRDFs"],
        index=0 if st.session_state.display_mode == "Average PRDF across frames" else 1,
        key="display_mode_radio",
        on_change=update_display_mode
    )
    # if display_mode == "Average PRDF across frames":
    plot_display_mode = st.sidebar.radio(
        "Plot Display Mode",
        ["Separate plots for each pair", "Combined plot with all pairs"],
        index=0,
        horizontal=True
    )

    with col2:
        line_style = st.sidebar.radio(
            "Line Style",
            ["Lines + Markers", "Lines Only"],
            index=1,
            key="line_style_radio",
            horizontal=True
        )
        st.session_state.line_style = line_style
        smoothing_params = add_smoothing_controls()

    frame_sampling = st.slider("Frame Sampling Rate", min_value=1, max_value=500, value=50,
                               help="Select every Nth frame from the trajectory")

    col1, col2 = st.columns(2)
    with col1:
        cutoff = st.number_input("Cutoff (Å)", min_value=1.0, max_value=50.0, value=7.0, step=1.0)
    with col2:
        bin_size = st.number_input("Bin Size (Å)", min_value=0.001, max_value=5.000, value=0.100, step=0.005)

    if st.button("Calculate PRDF", type="primary"):
        if file_type == "xyz" and not has_lattice and not st.session_state.use_manual_lattice:
            st.warning("⚠️ No lattice parameters defined...")
            st.session_state.show_pbc_warning = True  # SET FLAG
        else:
            trigger_calculation()

    if st.session_state.get('show_pbc_warning', False):
        if st.button("Continue without PBC", key="continue_no_pbc"):
            st.session_state.show_pbc_warning = False  # CLEAR FLAG
            trigger_calculation()

    if st.session_state.calc_rdf and uploaded_file:

        if st.session_state.do_calculation:
            with st.spinner("Processing CP2K trajectory file..."):
                try:
                    manual_lattice = st.session_state.lattice_params if st.session_state.use_manual_lattice else None
                    selected_frames, frame_indices = process_trajectory_file(uploaded_file, file_type, frame_sampling,
                                                                             manual_lattice)

                    frame_indices = [i * frame_sampling for i in range(len(selected_frames))]

                    st.info(f"Analyzing {len(selected_frames)} frames with sampling rate of {frame_sampling}")

                    species_list = list(set([atom.symbol for frame in selected_frames for atom in frame]))
                    species_combinations = list(combinations(species_list, 2)) + [(s, s) for s in species_list]

                    all_prdf_dict = defaultdict(list)
                    all_distance_dict = {}
                    global_rdf_list = []

                    progress_bar = st.progress(0)

                    for i, frame in enumerate(selected_frames):
                        try:
                            # Check if frame has proper cell/lattice parameters
                            if frame.cell is None or np.all(frame.cell.lengths() == 0):
                                # Create a large vacuum box for structures without PBC
                                vacuum_size = 500.0  # Large enough to avoid interactions
                                frame.set_cell([vacuum_size, vacuum_size, vacuum_size])
                                frame.set_pbc([False, False, False])  # No periodic boundary conditions

                                # Center atoms in the box
                                positions = frame.get_positions()
                                center = positions.mean(axis=0)
                                box_center = np.array([vacuum_size / 2, vacuum_size / 2, vacuum_size / 2])
                                frame.translate(box_center - center)

                            mg_structure = AseAtomsAdaptor.get_structure(frame)

                            prdf_featurizer = PartialRadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
                            prdf_featurizer.fit([mg_structure])
                            prdf_data = prdf_featurizer.featurize(mg_structure)
                            feature_labels = prdf_featurizer.feature_labels()

                            prdf_dict = defaultdict(list)
                            distance_dict = {}
                            global_dict = {}

                            for j, label in enumerate(feature_labels):
                                parts = label.split(" PRDF r=")
                                element_pair = tuple(parts[0].split("-"))
                                distance_range = parts[1].split("-")
                                bin_center = (float(distance_range[0]) + float(distance_range[1])) / 2
                                prdf_dict[element_pair].append(prdf_data[j])

                                if element_pair not in distance_dict:
                                    distance_dict[element_pair] = []
                                distance_dict[element_pair].append(bin_center)
                                global_dict[bin_center] = global_dict.get(bin_center, 0) + prdf_data[j]

                            for pair, values in prdf_dict.items():
                                if pair not in all_distance_dict:
                                    all_distance_dict[pair] = distance_dict[pair]
                                if isinstance(values, float):
                                    values = [values]
                                all_prdf_dict[pair].append(values)

                            global_rdf_list.append(global_dict)

                        except Exception as e:
                            st.warning(f"Error processing frame {i}: {str(e)}")

                        progress_bar.progress((i + 1) / len(selected_frames))

                    st.session_state.processed_data = {
                        "all_prdf_dict": all_prdf_dict,
                        "all_distance_dict": all_distance_dict,
                        "global_rdf_list": global_rdf_list,
                        "frame_indices": frame_indices,
                        "multi_structures": len(selected_frames) > 1
                    }

                    st.session_state.do_calculation = False

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.stop()

        all_prdf_dict = st.session_state.processed_data["all_prdf_dict"]
        all_distance_dict = st.session_state.processed_data["all_distance_dict"]
        global_rdf_list = st.session_state.processed_data["global_rdf_list"]
        frame_indices = st.session_state.processed_data["frame_indices"]
        multi_structures = st.session_state.processed_data["multi_structures"]

        colors = plt.cm.tab10.colors


        def rgb_to_hex(color):
            return '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))


        font_dict = dict(size=20, color="black")

        st.divider()
        st.subheader("PRDF Results")

        if display_mode == "Individual frame PRDFs":
            st.sidebar.slider("Animation Speed (seconds per frame)", min_value=0.05, max_value=2.0,
                              value=st.session_state.animation_speed, step=0.05, key="speed_slider")
            st.session_state.animation_speed = st.session_state.speed_slider

            display_animation_smoothing_info(smoothing_params)

            if plot_display_mode == "Combined plot with all pairs":
                fig_combined_anim = create_combined_animated_prdf_plot(
                    all_prdf_dict, all_distance_dict, frame_indices,
                    smoothing_params, st.session_state.line_style,
                    st.session_state.animation_speed, colors
                )
                st.plotly_chart(fig_combined_anim, use_container_width=True, key="combined_anim_prdf")


            else:
                for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):
                    hex_color = rgb_to_hex(colors[idx % len(colors)])
                    valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                    if not valid_prdf:
                        continue

                    fig = create_animated_prdf_plot(
                        all_distance_dict[comb], valid_prdf, comb, frame_indices,
                        smoothing_params, hex_color, st.session_state.line_style,
                        st.session_state.animation_speed
                    )

                    st.plotly_chart(fig, use_container_width=True, key=f"anim_prdf_{comb[0]}_{comb[1]}_{idx}")

        elif plot_display_mode == "Combined plot with all pairs":
            fig_combined = go.Figure()

            for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):
                hex_color = rgb_to_hex(colors[idx % len(colors)])
                valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                if valid_prdf:
                    prdf_array = np.vstack(valid_prdf)
                    prdf_data = np.mean(prdf_array, axis=0) if multi_structures else prdf_array[0]

                    if smoothing_params.get("enabled", False):
                        try:
                            prdf_data = smooth_prdf_data(
                                all_distance_dict[comb], prdf_data,
                                method=smoothing_params["method"],
                                **{k: v for k, v in smoothing_params.items()
                                   if k not in ["enabled", "method"]}
                            )
                        except:
                            pass

                    fig_combined.add_trace(go.Scatter(
                        x=all_distance_dict[comb],
                        y=prdf_data,
                        mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                        name=f"{comb[0]}-{comb[1]}",
                        line=dict(color=hex_color, width=2),
                        marker=dict(size=6) if st.session_state.line_style == "Lines + Markers" else dict()
                    ))

            title_str = "Combined Averaged PRDF: All Pairs" if multi_structures else "Combined PRDF: All Pairs"

            fig_combined.update_layout(
                title={'text': title_str, 'font': font_dict},
                xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                hovermode='x',
                font=font_dict,
                xaxis=dict(tickfont=font_dict),
                yaxis=dict(tickfont=font_dict, range=[0, None]),
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=20))
            )

            st.plotly_chart(fig_combined, use_container_width=True, key="combined_prdf_plot")

        else:
            for idx, (comb, prdf_list) in enumerate(all_prdf_dict.items()):
                hex_color = rgb_to_hex(colors[idx % len(colors)])
                valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                if not valid_prdf:
                    continue

                prdf_array = np.vstack(valid_prdf)
                prdf_data = np.mean(prdf_array, axis=0) if multi_structures else prdf_array[0]
                prdf_std = np.std(prdf_array, axis=0) if multi_structures and len(valid_prdf) > 1 else None

                title_str = f"Averaged PRDF: {comb[0]}-{comb[1]}" if multi_structures else f"PRDF: {comb[0]}-{comb[1]}"

                fig = create_smoothed_plot(
                    all_distance_dict[comb], prdf_data, comb, smoothing_params,
                    hex_color, title_str, st.session_state.line_style,
                    multi_structures, prdf_std
                )

                fig.update_layout(
                    title={'text': title_str, 'font': font_dict},
                    xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                    yaxis_title={'text': "PRDF Intensity", 'font': font_dict},
                    font=font_dict,
                    legend=legend_style,
                    xaxis=dict(tickfont=font_dict),
                    yaxis=dict(tickfont=font_dict, range=[0, None])
                )

                st.plotly_chart(fig, use_container_width=True, key=f"static_prdf_{comb[0]}_{comb[1]}_{idx}")

        st.subheader("Total RDF")

        global_bins_set = set()
        for gd in global_rdf_list:
            global_bins_set.update(gd.keys())
        global_bins = sorted(list(global_bins_set))

        hex_color_global = rgb_to_hex(colors[len(all_prdf_dict) % len(colors)])

        if display_mode == "Individual frame PRDFs":
            fig_global = create_animated_global_rdf_plot(
                global_bins, global_rdf_list, frame_indices, smoothing_params,
                hex_color_global, st.session_state.line_style, st.session_state.animation_speed
            )

            st.plotly_chart(fig_global, use_container_width=True, key="anim_global_rdf")

        else:
            global_rdf_avg = []
            global_rdf_std = []

            for b in global_bins:
                vals = [gd.get(b, 0) for gd in global_rdf_list]
                global_rdf_avg.append(np.mean(vals))
                global_rdf_std.append(np.std(vals))

            if smoothing_params.get("enabled", False):
                try:
                    global_rdf_avg = smooth_prdf_data(
                        global_bins, global_rdf_avg,
                        method=smoothing_params["method"],
                        **{k: v for k, v in smoothing_params.items()
                           if k not in ["enabled", "method"]}
                    )
                    global_rdf_std = smooth_prdf_data(
                        global_bins, global_rdf_std,
                        method=smoothing_params["method"],
                        **{k: v for k, v in smoothing_params.items()
                           if k not in ["enabled", "method"]}
                    )
                except:
                    pass

            fig_global = go.Figure()

            if smoothing_params.get("enabled", False):
                original_avg = []
                for b in global_bins:
                    vals = [gd.get(b, 0) for gd in global_rdf_list]
                    original_avg.append(np.mean(vals))

                fig_global.add_trace(go.Scatter(
                    x=global_bins, y=original_avg,
                    mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                    name="Global RDF (Original)",
                    line=dict(color=hex_color_global, width=1, dash='dot'),
                    marker=dict(size=4) if st.session_state.line_style == "Lines + Markers" else dict(),
                    opacity=0.5
                ))

                fig_global.add_trace(go.Scatter(
                    x=global_bins, y=global_rdf_avg,
                    mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                    name="Global RDF (Smoothed)",
                    line=dict(color=hex_color_global, width=2),
                    marker=dict(size=8) if st.session_state.line_style == "Lines + Markers" else dict()
                ))

                title_global = "Smoothed Global RDF" if not multi_structures else "Smoothed Averaged Global RDF"
            else:
                fig_global.add_trace(go.Scatter(
                    x=global_bins, y=global_rdf_avg,
                    mode='lines+markers' if st.session_state.line_style == "Lines + Markers" else 'lines',
                    name="Global RDF",
                    line=dict(color=hex_color_global, width=2),
                    marker=dict(size=8) if st.session_state.line_style == "Lines + Markers" else dict()
                ))

                title_global = "Global RDF" if not multi_structures else "Averaged Global RDF"

            if multi_structures and len(global_rdf_list) > 1:
                fig_global.add_trace(go.Scatter(
                    x=global_bins, y=[a + s for a, s in zip(global_rdf_avg, global_rdf_std)],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_global.add_trace(go.Scatter(
                    x=global_bins, y=[max(0, a - s) for a, s in zip(global_rdf_avg, global_rdf_std)],
                    mode='lines', line=dict(width=0),
                    fillcolor='rgba(100,100,100,0.2)', fill='tonexty', showlegend=False
                ))

            fig_global.update_layout(
                title={'text': title_global, 'font': font_dict},
                xaxis_title={'text': "Distance (Å)", 'font': font_dict},
                yaxis_title={'text': "Total RDF Intensity", 'font': font_dict},
                font=font_dict,
                legend=legend_style,
                xaxis=dict(tickfont=font_dict),
                yaxis=dict(tickfont=font_dict, range=[0, None])
            )

            st.plotly_chart(fig_global, use_container_width=True, key="static_global_rdf")

        st.subheader("Download Data")

        if st.button("Prepare Downloads"):
            is_individual_mode = display_mode == "Individual frame PRDFs"

            for comb, prdf_list in all_prdf_dict.items():
                valid_prdf = [np.array(p) for p in prdf_list if isinstance(p, list)]

                if valid_prdf:
                    df = pd.DataFrame()
                    df["Distance (Å)"] = all_distance_dict[comb]

                    if is_individual_mode:
                        for i, frame_data in enumerate(valid_prdf):
                            df[f"Frame_{frame_indices[i]}"] = frame_data
                        filename = f"{comb[0]}_{comb[1]}_prdf_frames.csv"
                        link_text = f"Download {comb[0]}-{comb[1]} PRDF data for all frames"
                    elif multi_structures:
                        prdf_array = np.vstack(valid_prdf)
                        df["Average"] = np.mean(prdf_array, axis=0)
                        if len(valid_prdf) > 1:
                            df["StdDev"] = np.std(prdf_array, axis=0)
                        filename = f"{comb[0]}_{comb[1]}_prdf_average.csv"
                        link_text = f"Download {comb[0]}-{comb[1]} Average PRDF data"
                    else:
                        df["PRDF"] = valid_prdf[0]
                        filename = f"{comb[0]}_{comb[1]}_prdf.csv"
                        link_text = f"Download {comb[0]}-{comb[1]} PRDF data"

                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
                    st.markdown(href, unsafe_allow_html=True)

            global_df = pd.DataFrame()
            global_df["Distance (Å)"] = global_bins

            if is_individual_mode:
                for i, gd in enumerate(global_rdf_list):
                    global_df[f"Frame_{frame_indices[i]}"] = [gd.get(b, 0) for b in global_bins]
                global_filename = "global_rdf_frames.csv"
                global_link_text = "Download Total RDF data for all frames"
            elif multi_structures:
                global_avgs = [np.mean([gd.get(b, 0) for gd in global_rdf_list]) for b in global_bins]
                global_stds = [np.std([gd.get(b, 0) for gd in global_rdf_list]) for b in global_bins]
                global_df["Average"] = global_avgs
                global_df["StdDev"] = global_stds
                global_filename = "global_rdf_average.csv"
                global_link_text = "Download Average Total RDF data"
            else:
                global_df["RDF"] = [global_rdf_list[0].get(b, 0) for b in global_bins]
                global_filename = "global_rdf.csv"
                global_link_text = "Download Total RDF data"

            global_csv = global_df.to_csv(index=False)
            global_b64 = base64.b64encode(global_csv.encode()).decode()
            global_href = f'<a href="data:file/csv;base64,{global_b64}" download="{global_filename}">{global_link_text}</a>'
            st.markdown(global_href, unsafe_allow_html=True)

else:
    st.info(
        "Please upload (in sidebar) a CP2K (.xyz) or LAMMPS (.lammpstrj, .dump, .txt) trajectory file to begin analysis")
    st.markdown("""
    **Supported file formats:**

    **Option 1: XYZ with lattice parameters (recommended)**
    - Standard XYZ format with lattice parameters in comment line
    - Format: `Lattice="a11 a12 a13 a21 a22 a23 a31 a32 a33"`
    - Atomic coordinates in Cartesian format
    - Periodic boundary conditions applied automatically

    **Option 2: Standard XYZ format**
    - Basic XYZ format without lattice information
    - You can define lattice parameters manually or from the CP2K .cell file in the interface and download the modified trajectory
    - Coordinates will be wrapped into the defined unit cell
    - If no lattice parameters are provided, PRDF calculated without PBC

    **Option 3: LAMMPS trajectory**
    - Standard dump trajectory files (.lammpstrj, .dump)

    **Example XYZ formats:**
    ```
    # With lattice parameters
    64
    Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3
    Si  0.000  0.000  0.000
    O   1.500  1.500  1.500
    ...

    # Standard XYZ (manual lattice input available)
    64
    Frame 1
    Si  0.000  0.000  0.000
    O   1.500  1.500  1.500
    ...
    ```
    """)

    # st.info(
    #    "💡 **Tip**: For best results with PRDF calculations, ensure your structure has proper periodic boundary conditions defined.")
