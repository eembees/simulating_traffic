import streamlit as st
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

rcParams["font.family"] = "monospace"

TASEP_explanation_string = """
The Totally Asymmetric Exclusion Process (TASEP) 
models the flow of traffic on a highway, 
given that cars move in one direction only, 
therefore the 'Total' asymmetry.
\n
TASEP with an Open Boundary Condition (OBC) models traffic as a discrete set of $L$ slots 
where cars move forward with probability $p$ (if not blocked by the next car), 
onto the first slot with probability $\\alpha$, 
and out of the last slot with probability $\\beta$.
\n
In the image below, we see that $L=10$, for visualization purposes.
\n
TASEP with an Periodic Boundary Condition (OBC) models traffic as a discrete set of $L$ slots 
where cars move forward with probability $p$, 
and from the last slot to the first slot also with probability $p$, not $\\alpha$ and $\\beta$.
"""

TASEP_PBC_string = """
For this case, we know a couple of things. 
\n
The first thing, is that the average flux, $J$, should equal the following, for low densities:
\n
$J = p \\rho(1-\\rho)$\n
This is because the flux at any point is equal to the hopping rate, times the probability that a site is filled, 
and the probability that the next site is empty $(1-\\rho)$.
\n
In the figure below, we calculate the empirical flux as well. 
If this doesn't fit with the data, there must be something afoot - usually there is an issue with the number of slots.
"""
CAR_URL = "figs/car.png"


# CAR_URL  ="https://vega.github.io/vega-datasets/data/ffox.png"


def plot_periodic(
        fluxes, const_L: int, const_stepProb: float, const_density: float, df: pd.DataFrame,
):
    J, Jmean, Jstd, J_theoretical = fluxes
    fig = plt.Figure(figsize=(7, 8))
    gs = GridSpec(nrows=3, ncols=1, figure=fig)
    ax_traj = fig.add_subplot(gs[:-1])
    ax_flux = fig.add_subplot(gs[-1])

    ax_flux.plot(J, alpha=0.7, lw=1)
    ax_flux.axhline(
        Jmean,
        ls="-.",
        alpha=0.5,
        color="xkcd:pastel red",
        label=rf"SS at $\operatorname{{E}}[J] = {Jmean:.2f}$",
    )
    ax_flux.axhspan(
        Jmean - Jstd / 2, Jmean + Jstd / 2, alpha=0.2, color="xkcd:pastel orange",
    )

    ax_flux.axhline(
        J_theoretical * const_L,
        ls="--",
        c="g",
        label=rf"$J_{{theoretical}} \cdot L = {J_theoretical * const_L:.2f} $",
    )

    ax_flux.set_xlabel("Time")
    ax_flux.set_ylabel(r"# particles moved ($J_{empirical}/L$) ")
    ax_flux.legend(loc="upper right")

    ax_traj.set_title(
        rf"Simulation with $L={const_L}, p\Delta t = {const_stepProb}, \rho={const_density:.2f}$"
    )

    # ax_traj.scatter(position, timepoints, marker=">", c="k", s=4)
    ax_traj.scatter(df["position"], df["time"], marker=">", c="k", s=4)

    ax_traj.set_xlabel("Position")
    ax_traj.set_ylabel("Time")

    fig.tight_layout()
    return fig, (ax_flux, ax_traj)


def calc_flux_empirical(position_time_array):
    # Calculate flux between timesteps
    fluxmat = np.diff(position_time_array, axis=0)
    fluxmat[fluxmat < 0] = 0
    J = fluxmat.sum(axis=1)

    # if bool_initrandom:  # Randomly populated means steady state more or less at once
    #     Jmean = J.mean()
    #     Jstd = J.std()
    # else:  # only use last 50% to guess SS
    Jmean = J[-int(len(J) // 2):].mean()
    Jstd = J[-int(len(J) // 2):].std()

    return J, Jmean, Jstd


@st.cache(allow_output_mutation=True)
def simulate_periodic(
        const_nSteps: int,
        const_L: int,
        const_density: float,
        const_stepProb: float,
        bool_initrandom: bool,
):
    # Initialize the array for storing all positions over time

    position_time_array = np.zeros(
        shape=(const_nSteps, const_L)
    )  # now we access it as [time, position]

    # Populate the array at time 0
    if bool_initrandom:  # Randomly populate
        position_time_array[0] = np.random.binomial(n=1, p=const_density, size=const_L)
    else:
        position_time_array[0][: int(const_L * const_density)] = 1

    for i in range(1, const_nSteps):
        N_curr = np.copy(position_time_array[i - 1])
        move_inds = np.random.choice(np.arange(const_L), size=const_L, replace=True)

        for j in move_inds:

            if (
                    N_curr[j] == 1
                    and N_curr[(j + 1) % (const_L - 1)] == 0
                    and np.random.uniform() > const_stepProb
            ):
                N_curr[j] = 0
                N_curr[(j + 1) % (const_L - 1)] = 1

        position_time_array[i] = N_curr

    (timepoints, positions) = np.nonzero(position_time_array)
    fluxes_empirical = calc_flux_empirical(position_time_array)

    return pd.DataFrame({"time": timepoints, "position": positions}), fluxes_empirical


def write():
    st.title("TASEP Simulation Page")
    # Introduction and analytical
    st.write(TASEP_explanation_string)
    st.image("figs/TASEP_OBC.png", use_column_width=True, format="PNG")

    # Simulation part

    # Setting parameters
    const_nSteps = st.sidebar.slider("Number of steps to simulate", 10, 2000, 100, 10)
    const_L = st.sidebar.slider("Number of slots", 5, 50, 10)
    const_density = st.sidebar.slider("Initial Density", 0.01, 0.99, 0.5, 0.05)
    const_stepProb = st.sidebar.slider("Step Probability", 0.01, 0.99, 0.5, 0.05)
    bool_periodic_boundary = st.sidebar.checkbox("Periodic Boundary Condition?", True)
    bool_initrandom = st.sidebar.checkbox("Random initialization?", True)

    if st.button("Run simulation?"):
        if bool_periodic_boundary:
            st.subheader("Simulation with Periodic Boundary Condition.")
            st.write()
            df, (J, Jmean, Jstd) = simulate_periodic(
                const_L=const_L,
                const_stepProb=const_stepProb,
                const_nSteps=const_nSteps,
                const_density=const_density,
                bool_initrandom=bool_initrandom,
            )
            J_theoretical = const_stepProb * const_density * (1 - const_density)
            fluxes = J, Jmean, Jstd, J_theoretical
            df_plot = df.copy()
            df_plot["img"] = [CAR_URL] * len(df_plot)
            df_plot["car"] = [1] * len(df_plot)

            slider = alt.binding_range(min=0, max=const_nSteps - 1, step=1)
            select_time = alt.selection_single(
                name="time", fields=["time"], bind=slider, init={"time": 1}
            )

            c = (
                alt.Chart(df_plot)
                    .properties(height=100)
                    # .mark_image(width=15, height=15) # TODO: get the image to work to get a car!
                    .mark_circle(size=200)
                    .encode(
                    x=alt.X(
                        "position",
                        type="quantitative",
                        scale=alt.Scale(domain=[0, const_L]),
                    ),
                    y=alt.Y(field="car", type="ordinal"),
                    url="img",
                )
                    .add_selection(select_time)
                    .transform_filter(select_time)
            )
            st.write(
                "Now, try and play with the slider representing the timepoint, to see the cars move in time!"
            )
            st.altair_chart(c, use_container_width=True)

            fig, axes = plot_periodic(
                fluxes=fluxes,
                const_L=const_L,
                const_stepProb=const_stepProb,
                const_density=const_density,
                df=df,
            )

            st.write("Here is a summary figure as well!")
            st.pyplot(fig)

            st.write(f"Here we have a an average flux of $J_{{emp}}={Jmean:.2f}\\pm{Jstd:.3f}$, compared to $J_{{theo}}={J_theoretical:.2f}$")
        else:
            st.write("Nothing to see here...")
