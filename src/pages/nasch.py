from typing import Union, Tuple

import pandas as pd
import streamlit as st
import altair as alt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

rcParams["font.family"] = "monospace"


NASCH_explanation_string = """
The Nagel-Schereckenberg (NaSch) Model, is an extension of the TASEP model. \n
In the NaSch Model, each particle $x_i$ has a velocity $v_i$, which is updated in each step.
The velocity has a maximum value, $v_{max}$ which can be adjusted. 
Following the model with $L$ slots as in the TASEP model, 
for each step in the NaSch model each particle goes through the following steps:\n
1. Update velocity step: $v_i \\rightarrow \\min(v_i + 1, v_{max}, x_{i+1} - x_i - 1)$ 
 Here, we make sure that the new velocity will not cause a crash by introducing the $x_{i+1} - x_i - 1$ term.
2. Randomly brake with probability $p$: $v_i \\rightarrow \\max(v_i - 1,0)$ 
3. Move particle: $x_{i} \\rightarrow x_i + v_i$ 
4. If the boundary is nonperiodic, we add a new particle with a random velocity.
 $x_i = 0,  v_i \\in [0, v_{max}]$ with probability $alpha$, as in the normal TASEP.
"""


def get_spatiotemporal_plot(
    df: pd.DataFrame, flux: np.ndarray
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    fig = plt.Figure(figsize=(7, 8))
    gs = GridSpec(nrows=3, ncols=1, figure=fig)
    ax_traj = fig.add_subplot(gs[:-1])
    ax_flux = fig.add_subplot(gs[-1])

    ax_flux.plot(flux, alpha=0.7, lw=1)

    ax_flux.set_xlabel("Time")
    ax_flux.set_ylabel(r"# particles moved ($J_{empirical}/L$) ")

    ax_traj.set_title(rf"Simulation with $L={df['position'].max()}$")
    ax_traj.scatter(df["position"], df["time"], marker=">", c="k", s=4)

    ax_traj.set_xlabel("Position")
    ax_traj.set_ylabel("Time")

    fig.tight_layout()
    return fig, (ax_flux, ax_traj)


@st.cache(allow_output_mutation=True)
def simulate_nasch(
    n_steps: int,
    n_slots: int,
    init_random: bool,
    init_density: float,
    const_vmax: int,
    const_stepprob: float,
    periodic_boundary: bool = False,
    entry_prob: Union[float, None] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if not periodic_boundary and entry_prob is None:
        raise ValueError(
            "entry_prob must be given when simulating open boundaries!!"
        )

    # Initialize the array for storing all positions over time
    position_time_array = np.ones(shape=(n_steps, n_slots), dtype=int) * -1
    flux_array = np.zeros(shape=n_steps)
    # Populate the array at time 0
    if init_random:  # Randomly populate
        position_time_array[0] = (
            np.random.binomial(n=1, p=init_density, size=n_slots) * -1
        )
    else:
        position_time_array[0][: int(n_slots * init_density)] = 1

    for ti in range(1, n_steps):
        curr_vs = np.copy(position_time_array[ti - 1])
        next_vs = get_new_speeds(
            curr_vs, const_stepprob, const_vmax, periodic_boundary
        )
        if (
            not periodic_boundary
            and next_vs[0] < 0
            and np.random.uniform() < entry_prob
        ):
            #  add a new car going at a random speed between 0 and vmax
            next_vs[0] = np.random.choice(np.arange(const_vmax))
        position_time_array[ti] = next_vs
        # flux is approximated here to be equal to avg velocities over the whole road
        flux_array[ti] = np.sum(next_vs[next_vs > 0]) / n_slots

    (timepoints, positions) = np.nonzero(position_time_array + 1)
    speeds = position_time_array[timepoints, positions]
    return (
        pd.DataFrame(
            {"time": timepoints, "position": positions, "speed": speeds}
        ),
        flux_array,
    )


def get_new_speeds(
    curr_vs: np.ndarray, stepprob: float, vmax: int, periodic: bool
) -> np.ndarray:
    n_slots = len(curr_vs)
    next_vs = np.ones_like(curr_vs) * -1
    for xi in range(n_slots):
        if curr_vs[xi] > -1:  # if -1, that means no particle
            # find distance to next car
            distance = 1
            while curr_vs[(xi + distance) % n_slots] < 0:
                distance += 1
            vi = min(curr_vs[xi] + 1, distance - 1, vmax)
            #  Random braking by one speed unit
            if np.random.uniform() > stepprob:
                vi = max(0, vi - 1)
            if periodic:
                next_vs[
                    (xi + vi) % n_slots
                ] = vi  # make the step periodically, so loop around
            else:
                if xi + vi <= n_slots - 1:
                    next_vs[xi + vi] = vi
    return next_vs


def write():
    st.title("Nagel-Schereckenberg (NaSch) Model Simulation Page")
    st.write(NASCH_explanation_string)
    # Set parameters
    st.sidebar.subheader("Set Parameters")

    bool_periodic_boundary = st.sidebar.checkbox(
        "Periodic Boundary Condition?", True
    )
    bool_initrandom = st.sidebar.checkbox("Random initialization?", True)
    const_nSteps = st.sidebar.slider(
        "Number of steps to simulate", 10, 2000, 100, 10
    )
    const_L = st.sidebar.slider("Number of slots", 10, 50, 20)
    const_vmax = st.sidebar.slider("Maximum speed", 1, 10, 5)
    const_density = st.sidebar.slider("Initial Density", 0.0, 0.95, 0.5, 0.05)
    const_stepProb = st.sidebar.slider("Step Probability", 0.05, 1.0, 0.5, 0.05)
    const_entryProb = st.sidebar.slider(
        "Entry Probability (alpha)", 0.05, 0.95, 0.5, 0.05
    )
    if st.button("Run simulation?"):
        df, flux = simulate_nasch(
            n_steps=const_nSteps,
            n_slots=const_L,
            init_random=bool_initrandom,
            init_density=const_density,
            const_vmax=const_vmax,
            const_stepprob=const_stepProb,
            periodic_boundary=bool_periodic_boundary,
            entry_prob=const_entryProb,
        )
        df_plot = df.copy()
        df_plot["img"] = ["car"] * len(df_plot)
        df_plot["car"] = [1] * len(df_plot)

        slider = alt.binding_range(min=0, max=const_nSteps - 1, step=1)
        select_time = alt.selection_single(
            name="time", fields=["time"], bind=slider, init={"time": 1}
        )
        c = (
            alt.Chart(df_plot)
            .properties(height=100)
            .mark_text(size=30, baseline="middle")
            .encode(
                alt.X(
                    "position",
                    type="quantitative",
                    scale=alt.Scale(domain=[0, const_L]),
                ),
                alt.Y(field="car", type="ordinal"),
                alt.Text("emoji:N"),
            )
            .transform_calculate(emoji="{'car':'🚗'}[datum.img]")
            .add_selection(select_time)
            .transform_filter(select_time)
        )

        st.write(
            "Now, try and play with the slider representing the timepoint, to see the cars move in time!"
        )
        st.altair_chart(c, use_container_width=True)

        fig, axes = get_spatiotemporal_plot(df, flux)
        st.write("Here is a summary figure as well!")
        st.pyplot(fig)
