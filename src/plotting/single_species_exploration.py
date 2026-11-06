"""Linked dynamics and a browser-side landscape slider for the teaching notebook."""

import copy

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from src.workflows.single_species_exploration import model_display_name


SPECIES_COLOR = "#0072B2"
TWO_SPECIES_COLORS = ("#0072B2", "#D55E00")


def resident_symbol(extinct):
    return "x" if extinct else "circle"


def resident_label(result, index):
    status = " · extinct" if result.extinct[index] else ""
    return (f"Species 1 · x = {result.population[index]:.4g}"
            f" · u = {result.trait[index]:.4f}{status}")


def fitness_limits(result, limits=None):
    if limits is not None:
        if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
            raise ValueError("Fitness limits must be a finite (minimum, maximum) pair.")
        # A focused window may crop distant mutant tails, but must never hide
        # the resident circle or its extinction cross.
        lower = min(limits[0], float(np.min(result.resident_fitness)))
        upper = max(limits[1], float(np.max(result.resident_fitness)))
        margin = max(0.01, (upper - lower) * 0.03)
        return lower - margin, upper + margin
    lower = min(0.0, float(np.min(result.mutant_fitness)),
                float(np.min(result.resident_fitness)))
    upper = max(0.0, float(np.max(result.mutant_fitness)),
                float(np.max(result.resident_fitness)))
    margin = max(0.01, (upper - lower) * 0.07)
    return lower - margin, upper + margin


def landscape_time_slider(result, limits=None, reference=None):
    """Update curve, legend, symbol, hover values, and time cursors together.

    A native Plotly update slider uses prepared arrays; it never calls Python
    while scrubbing. Explicit restyles also refresh legend symbols and names.
    """
    figure = make_subplots(
        rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, None]],
        column_widths=[0.4, 0.6], horizontal_spacing=0.12,
        subplot_titles=("Population dynamics", "Adaptive fitness landscape", "Trait dynamics"),
    )
    figure.add_trace(go.Scatter(x=result.time, y=result.population, mode="lines",
                               line={"color": SPECIES_COLOR}, showlegend=False,
                               hovertemplate="t = %{x:.4g}<br>x = %{y:.5g}<extra>Species 1</extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=result.time, y=result.trait, mode="lines",
                               line={"color": SPECIES_COLOR}, showlegend=False), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=result.mutant_traits, y=result.mutant_fitness[0], mode="lines",
        line={"color": "#333333", "width": 2}, name="Mutant fitness G(v)",
        hovertemplate="v = %{x:.4f}<br>G(v) = %{y:.5g}<extra></extra>",
    ), row=1, col=2)
    figure.add_trace(go.Scatter(
        x=[result.trait[0]], y=[result.resident_fitness[0]], mode="markers",
        marker={"color": SPECIES_COLOR, "size": 13,
                "symbol": resident_symbol(result.extinct[0]),
                "line": {"width": 1, "color": SPECIES_COLOR}},
        name=resident_label(result, 0),
    ), row=1, col=2)
    if reference is not None:
        for row, values, label in (
            (1, reference.population, "x"), (2, reference.trait, "u"),
        ):
            figure.add_trace(go.Scatter(
                x=reference.time, y=values, mode="lines",
                line={"color": SPECIES_COLOR, "dash": "dash"}, opacity=0.65,
                name="Evolution disabled (comparison)", showlegend=row == 1,
                hovertemplate=f"t = %{{x:.4g}}<br>{label} = %{{y:.5g}}"
                              "<extra>Evolution disabled</extra>",
            ), row=row, col=1)
    figure.add_hline(y=0, line_color="gray", line_dash="dash", row=1, col=2)
    figure.add_vline(x=float(result.time[0]), line_color="#D55E00", row=1, col=1)
    figure.add_vline(x=float(result.time[0]), line_color="#D55E00", row=2, col=1)
    base_shapes = [shape.to_plotly_json() for shape in figure.layout.shapes]
    steps = []
    for index, time in enumerate(result.time):
        shapes = copy.deepcopy(base_shapes)
        for shape in shapes[1:]:
            shape["x0"] = shape["x1"] = float(time)
        status = "extinct" if result.extinct[index] else "living"
        hover = (f"Species 1<br>t = {time:g}<br>x = {result.population[index]:.6g}"
                 f"<br>u = {result.trait[index]:.6g}<br>G(u) = {result.resident_fitness[index]:.6g}"
                 f"<br>{status}<extra></extra>")
        steps.append({
            "method": "update", "label": f"{time:g}",
            "args": [
                {"x": [result.mutant_traits.tolist(), [float(result.trait[index])]],
                 "y": [result.mutant_fitness[index].tolist(), [float(result.resident_fitness[index])]],
                 "name": ["Mutant fitness G(v)", resident_label(result, index)],
                 "marker.symbol": ["circle", resident_symbol(result.extinct[index])],
                 "hovertemplate": ["v = %{x:.4f}<br>G(v) = %{y:.5g}<extra></extra>", hover]},
                {"shapes": shapes,
                 "title.text": f"{model_display_name(result.settings.model_name)} · effective resource support E = {result.settings.E:g} · t = {time:g}"},
                [2, 3],
            ],
        })
    figure.data[3].hovertemplate = steps[0]["args"][0]["hovertemplate"][1]
    figure.update_xaxes(title_text="Time", range=[0, result.settings.duration], row=1, col=1)
    figure.update_xaxes(title_text="Time", range=[0, result.settings.duration], row=2, col=1)
    figure.update_yaxes(title_text="Population x", row=1, col=1)
    figure.update_yaxes(title_text="Trait u", row=2, col=1)
    figure.update_xaxes(title_text="Trait v",
                       range=[result.mutant_traits[0], result.mutant_traits[-1]], row=1, col=2)
    figure.update_yaxes(title_text="Fitness G(v)", range=fitness_limits(result, limits), row=1, col=2)
    figure.update_layout(
        template="plotly_white", height=720,
        title={"text": steps[0]["args"][1]["title.text"], "y": 0.98, "x": 0.03},
        margin={"t": 150, "b": 120},
        legend={"orientation": "h", "y": 1.18, "x": 0.0},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Time: "},
                  "pad": {"t": 55}, "steps": steps}],
    )
    return figure


def two_species_landscape_time_slider(result, limits=None, reference=None):
    """Plot both interacting trajectories and their shared fitness landscape."""
    figure = make_subplots(
        rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, None]],
        column_widths=[0.4, 0.6], horizontal_spacing=0.12,
        subplot_titles=("Population dynamics", "Shared adaptive fitness landscape",
                        "Trait dynamics"),
    )
    for species, color in enumerate(TWO_SPECIES_COLORS):
        figure.add_trace(go.Scatter(
            x=result.time, y=result.population[species], mode="lines",
            line={"color": color}, name=f"Species {species + 1} population",
            hovertemplate="t = %{x:.4g}<br>x = %{y:.5g}<extra></extra>",
        ), row=1, col=1)
    for species, color in enumerate(TWO_SPECIES_COLORS):
        figure.add_trace(go.Scatter(
            x=result.time, y=result.trait[species], mode="lines",
            line={"color": color}, name=f"Species {species + 1} trait",
            hovertemplate="t = %{x:.4g}<br>u = %{y:.5g}<extra></extra>",
        ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=result.mutant_traits, y=result.mutant_fitness[0], mode="lines",
        line={"color": "#333333", "width": 2}, name="Mutant fitness G(v)",
        hovertemplate="v = %{x:.4f}<br>G(v) = %{y:.5g}<extra></extra>",
    ), row=1, col=2)
    for species, color in enumerate(TWO_SPECIES_COLORS):
        figure.add_trace(go.Scatter(
            x=[result.trait[species, 0]],
            y=[result.resident_fitness[species, 0]], mode="markers",
            marker={"color": color, "size": 13,
                    "symbol": "x" if result.extinct[species, 0] else "circle",
                    "line": {"width": 1, "color": color}},
            name=f"Species {species + 1} resident",
        ), row=1, col=2)

    if reference is not None:
        for species, color in enumerate(TWO_SPECIES_COLORS):
            for row, values, label in (
                (1, reference.population[species], "x"),
                (2, reference.trait[species], "u"),
            ):
                figure.add_trace(go.Scatter(
                    x=reference.time, y=values, mode="lines",
                    line={"color": color, "dash": "dash"}, opacity=0.65,
                    name=f"Species {species + 1} · evolution disabled",
                    showlegend=row == 1,
                    hovertemplate=f"t = %{{x:.4g}}<br>{label} = %{{y:.5g}}"
                                  f"<extra>Species {species + 1}, evolution disabled</extra>",
                ), row=row, col=1)

    figure.add_hline(y=0, line_color="gray", line_dash="dash", row=1, col=2)
    figure.add_vline(x=float(result.time[0]), line_color="#777777", row=1, col=1)
    figure.add_vline(x=float(result.time[0]), line_color="#777777", row=2, col=1)
    base_shapes = [shape.to_plotly_json() for shape in figure.layout.shapes]

    def resident_name(species, index):
        status = " · extinct" if result.extinct[species, index] else ""
        return (f"Species {species + 1} · x = {result.population[species, index]:.4g}"
                f" · u = {result.trait[species, index]:.4f}{status}")

    def resident_hover(species, index):
        status = "extinct" if result.extinct[species, index] else "living"
        return (f"Species {species + 1}<br>t = {result.time[index]:g}"
                f"<br>x = {result.population[species, index]:.6g}"
                f"<br>u = {result.trait[species, index]:.6g}"
                f"<br>G(u) = {result.resident_fitness[species, index]:.6g}"
                f"<br>{status}<extra></extra>")

    steps = []
    for index, time in enumerate(result.time):
        shapes = copy.deepcopy(base_shapes)
        for shape in shapes[1:]:
            shape["x0"] = shape["x1"] = float(time)
        steps.append({
            "method": "update", "label": f"{time:g}",
            "args": [
                {
                    "x": [result.mutant_traits.tolist(),
                          [float(result.trait[0, index])],
                          [float(result.trait[1, index])]],
                    "y": [result.mutant_fitness[index].tolist(),
                          [float(result.resident_fitness[0, index])],
                          [float(result.resident_fitness[1, index])]],
                    "name": ["Mutant fitness G(v)", resident_name(0, index),
                             resident_name(1, index)],
                    "marker.symbol": ["circle", *("x" if result.extinct[s, index]
                                                    else "circle" for s in range(2))],
                    "hovertemplate": [
                        "v = %{x:.4f}<br>G(v) = %{y:.5g}<extra></extra>",
                        resident_hover(0, index), resident_hover(1, index),
                    ],
                },
                {"shapes": shapes,
                 "title.text": f"{model_display_name(result.settings.model_name)} · "
                               f"E = {result.settings.E:g} · time since introduction = {time:g}"},
                [4, 5, 6],
            ],
        })
    for species in range(2):
        figure.data[5 + species].name = resident_name(species, 0)
        figure.data[5 + species].hovertemplate = resident_hover(species, 0)
    figure.update_xaxes(title_text="Time since second species introduction",
                        range=[0, result.settings.duration], row=1, col=1)
    figure.update_xaxes(title_text="Time since second species introduction",
                        range=[0, result.settings.duration], row=2, col=1)
    figure.update_yaxes(title_text="Population x", row=1, col=1)
    figure.update_yaxes(title_text="Trait u", row=2, col=1)
    figure.update_xaxes(title_text="Mutant trait v",
                        range=[result.mutant_traits[0], result.mutant_traits[-1]], row=1, col=2)
    figure.update_yaxes(title_text="Fitness G(v)", range=fitness_limits(result, limits),
                        row=1, col=2)
    figure.update_layout(
        template="plotly_white", height=720,
        title={"text": steps[0]["args"][1]["title.text"], "y": 0.98, "x": 0.03},
        margin={"t": 150, "b": 120},
        legend={"orientation": "h", "y": 1.18, "x": 0.0},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Time: "},
                  "pad": {"t": 55}, "steps": steps}],
    )
    return figure


def plot_leader_analysis(analysis, trajectory=None):
    """One interactive view of payoffs, responses, and candidate game points."""
    from src.workflows.single_species_leader_exploration import fixed_trait_population

    figure = make_subplots(
        rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]],
        horizontal_spacing=0.12, vertical_spacing=0.18,
        subplot_titles=("Leader payoff across resource support",
                        "Species and leader best responses", "Population x versus effective resource support E"),
    )

    def separate_branches(x, y, branches):
        # The global optimum can jump between a living and extinct policy.
        # Connecting those values would draw policies that are not responses.
        separated_x, separated_y = [], []
        for index, (x_value, y_value) in enumerate(zip(x, y)):
            if index and branches[index] != branches[index - 1]:
                separated_x.append(np.nan)
                separated_y.append(np.nan)
            separated_x.append(x_value)
            separated_y.append(y_value)
        return separated_x, separated_y

    figure.add_trace(go.Scatter(
        x=analysis.E, y=analysis.payoff, mode="lines+markers", name="Evo-aware payoff",
        line={"color": SPECIES_COLOR},
        marker={"size": 6, "symbol": np.where(analysis.stationary, "circle", "circle-open")},
        customdata=analysis.stationary,
        hovertemplate="E = %{x:.5g}<br>Q = %{y:.5g}<br>Stationary: %{customdata}<extra></extra>",
    ), row=1, col=1)
    for index, curve in enumerate(analysis.nash_payoff_curves):
        support, payoff = separate_branches(
            curve.E, curve.payoff, curve.population > analysis.biology.extinction_threshold,
        )
        figure.add_trace(go.Scatter(
            x=support, y=payoff, mode="lines",
            name=f"Eco-aware: Nash trait u = {curve.point.u:.4g}",
            line={"color": ("#777777", "#999933", "#882255")[index % 3], "dash": "dash"},
        ), row=1, col=1)
        figure.add_trace(go.Scatter(
            x=curve.E, y=curve.population, mode="lines", showlegend=False,
            name=f"Population at Nash trait u = {curve.point.u:.4g}",
            line={"color": ("#777777", "#999933", "#882255")[index % 3], "dash": "dash"},
        ), row=2, col=2)
    living_traits = np.where(analysis.population > analysis.biology.extinction_threshold,
                             analysis.trait, np.nan)
    figure.add_trace(go.Scatter(
        x=analysis.E, y=living_traits, mode="lines+markers", name="Species response",
        line={"color": SPECIES_COLOR}, connectgaps=False,
    ), row=2, col=1)
    leader_branches = [float(fixed_trait_population(E, u, analysis.biology)) > analysis.biology.extinction_threshold
                       for E, u in zip(analysis.leader_E, analysis.leader_traits)]
    leader_support, leader_traits = separate_branches(analysis.leader_E, analysis.leader_traits, leader_branches)
    figure.add_trace(go.Scatter(
        x=leader_support, y=leader_traits, mode="lines", name="Leader best response",
        line={"color": "#CC79A7"},
    ), row=2, col=1)
    figure.add_trace(go.Scatter(
        x=analysis.E, y=analysis.population, mode="lines+markers", showlegend=False,
        name="Population x", line={"color": SPECIES_COLOR}, connectgaps=False,
        marker={"size": 6, "symbol": np.where(
            analysis.population > analysis.biology.extinction_threshold, "circle", "x",
        )},
        hovertemplate="E = %{x:.5g}<br>x = %{y:.5g}<extra></extra>",
    ), row=2, col=2)

    for name, points, symbol, color in (
        ("Candidate Stackelberg", [analysis.stackelberg], "star", "#D55E00"),
        ("Candidate Nash", analysis.nash_points, "diamond", "#009E73"),
    ):
        for index, point in enumerate(points):
            hover = (f"{name}<br>E = {point.E:.6g}<br>x = {point.x:.6g}"
                     f"<br>u = {point.u:.6g}<br>Q = {point.payoff:.6g}"
                     f"<br>Stationary response: {point.stationary}<extra></extra>")
            marker = {"symbol": symbol, "size": 13, "color": color}
            figure.add_trace(go.Scatter(
                x=[point.E], y=[point.payoff], mode="markers", name=name,
                legendgroup=name, showlegend=index == 0, marker=marker, hovertemplate=hover,
            ), row=1, col=1)
            if point.x > analysis.biology.extinction_threshold:
                figure.add_trace(go.Scatter(
                    x=[point.E], y=[point.u], mode="markers", showlegend=False,
                    legendgroup=name, marker=marker, hovertemplate=hover,
                ), row=2, col=1)
            figure.add_trace(go.Scatter(
                x=[point.E], y=[point.x], mode="markers", showlegend=False,
                legendgroup=name, marker=marker, hovertemplate=hover,
            ), row=2, col=2)

    for row, col in ((1, 1), (2, 1), (2, 2)):
        figure.update_xaxes(title_text="Effective resource support E",
                           range=[analysis.scan.E_min, analysis.scan.E_max], row=row, col=col)
    figure.update_yaxes(title_text="Leader payoff Q", row=1, col=1)
    figure.update_yaxes(title_text="Trait u", row=2, col=1)
    figure.update_yaxes(title_text="Population x", row=2, col=2)
    figure.update_layout(
        template="plotly_white", height=850,
        title={"text": f"{model_display_name(analysis.biology.model_name)} · leader analysis", "y": 0.99},
        legend={"orientation": "h", "y": 1.18, "x": 0},
        margin={"t": 175, "b": 75},
    )
    return figure


def species_color(index):
    return qualitative.Plotly[index % len(qualitative.Plotly)]


def _fitness_range(result, limits=None):
    if limits is not None:
        if (len(limits) != 2 or not np.all(np.isfinite(limits))
                or limits[0] >= limits[1]):
            raise ValueError("Fitness limits must be a finite increasing pair.")
        low, high = float(limits[0]), float(limits[1])
    else:
        low = min(0.0, float(np.min(result.mutant_fitness)),
                  float(np.min(result.resident_fitness)))
        high = max(0.0, float(np.max(result.mutant_fitness)),
                   float(np.max(result.resident_fitness)))
    low = min(low, float(np.min(result.resident_fitness)))
    high = max(high, float(np.max(result.resident_fitness)))
    margin = max(0.01, (high - low) * 0.05)
    return low - margin, high + margin


def multi_population_landscape_time_slider(result, limits=None, reference=None):
    """Plot all population trajectories with their shared fitness landscape."""
    count = result.population.shape[0]
    figure = make_subplots(
        rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, None]],
        column_widths=[0.4, 0.6], horizontal_spacing=0.12,
        subplot_titles=("Population dynamics", "Shared adaptive fitness landscape",
                        "Trait dynamics"),
    )
    for species in range(count):
        color = species_color(species)
        figure.add_trace(go.Scatter(
            x=result.time, y=result.population[species], mode="lines",
            line={"color": color}, name=f"Population {species + 1}",
            hovertemplate="t = %{x:.4g}<br>x = %{y:.5g}<extra></extra>",
        ), row=1, col=1)
    for species in range(count):
        color = species_color(species)
        figure.add_trace(go.Scatter(
            x=result.time, y=result.trait[species], mode="lines",
            line={"color": color}, name=f"Trait {species + 1}",
            hovertemplate="t = %{x:.4g}<br>u = %{y:.5g}<extra></extra>",
        ), row=2, col=1)

    landscape_index = 2 * count
    marker_start = landscape_index + 1
    figure.add_trace(go.Scatter(
        x=result.mutant_traits, y=result.mutant_fitness[0], mode="lines",
        line={"color": "#333333", "width": 2}, name="Mutant fitness G(v)",
        hovertemplate="v = %{x:.4f}<br>G(v) = %{y:.5g}<extra></extra>",
    ), row=1, col=2)
    for species in range(count):
        color = species_color(species)
        figure.add_trace(go.Scatter(
            x=[result.trait[species, 0]],
            y=[result.resident_fitness[species, 0]], mode="markers",
            marker={"color": color, "size": 12,
                    "symbol": "x" if result.extinct[species, 0] else "circle",
                    "line": {"width": 1, "color": color}},
            name=f"Population {species + 1} resident",
        ), row=1, col=2)

    if reference is not None:
        for species in range(count):
            color = species_color(species)
            for row, values, label in (
                (1, reference.population[species], "x"),
                (2, reference.trait[species], "u"),
            ):
                figure.add_trace(go.Scatter(
                    x=reference.time, y=values, mode="lines",
                    line={"color": color, "dash": "dash"}, opacity=0.6,
                    name=f"Population {species + 1} · evolution off",
                    showlegend=row == 1,
                    hovertemplate=f"t = %{{x:.4g}}<br>{label} = %{{y:.5g}}"
                                  f"<extra>Population {species + 1}, evolution off</extra>",
                ), row=row, col=1)

    figure.add_hline(y=0, line_color="gray", line_dash="dash", row=1, col=2)
    figure.add_vline(x=float(result.time[0]), line_color="#777777", row=1, col=1)
    figure.add_vline(x=float(result.time[0]), line_color="#777777", row=2, col=1)
    base_shapes = [shape.to_plotly_json() for shape in figure.layout.shapes]

    def resident_name(species, index):
        status = " · extinct" if result.extinct[species, index] else ""
        return (f"Population {species + 1} · x={result.population[species, index]:.4g}"
                f" · u={result.trait[species, index]:.4f}{status}")

    def resident_hover(species, index):
        status = "extinct" if result.extinct[species, index] else "living"
        return (f"Population {species + 1}<br>t = {result.time[index]:g}"
                f"<br>x = {result.population[species, index]:.6g}"
                f"<br>u = {result.trait[species, index]:.6g}"
                f"<br>G(u) = {result.resident_fitness[species, index]:.6g}"
                f"<br>{status}<extra></extra>")

    traces_to_update = list(range(landscape_index, marker_start + count))
    steps = []
    for index, time in enumerate(result.time):
        shapes = copy.deepcopy(base_shapes)
        for shape in shapes[1:]:
            shape["x0"] = shape["x1"] = float(time)
        x_values = [result.mutant_traits.tolist()]
        y_values = [result.mutant_fitness[index].tolist()]
        names = ["Mutant fitness G(v)"]
        symbols = ["circle"]
        hover = ["v = %{x:.4f}<br>G(v) = %{y:.5g}<extra></extra>"]
        for species in range(count):
            x_values.append([float(result.trait[species, index])])
            y_values.append([float(result.resident_fitness[species, index])])
            names.append(resident_name(species, index))
            symbols.append("x" if result.extinct[species, index] else "circle")
            hover.append(resident_hover(species, index))
        steps.append({
            "method": "update", "label": f"{time:g}",
            "args": [
                {"x": x_values, "y": y_values, "name": names,
                 "marker.symbol": symbols, "hovertemplate": hover},
                {"shapes": shapes,
                 "title.text": f"{model_display_name(result.settings.model_name)} · "
                               f"{count} population(s) · E={result.settings.E:g} · t={time:g}"},
                traces_to_update,
            ],
        })
    for species in range(count):
        marker = figure.data[marker_start + species]
        marker.name = resident_name(species, 0)
        marker.hovertemplate = resident_hover(species, 0)

    figure.update_xaxes(title_text="Time since the latest population was added",
                        range=[0, result.settings.duration], row=1, col=1)
    figure.update_xaxes(title_text="Time since the latest population was added",
                        range=[0, result.settings.duration], row=2, col=1)
    figure.update_yaxes(title_text="Population x", row=1, col=1)
    figure.update_yaxes(title_text="Trait u", row=2, col=1)
    figure.update_xaxes(title_text="Mutant trait v",
                        range=[result.mutant_traits[0], result.mutant_traits[-1]], row=1, col=2)
    figure.update_yaxes(title_text="Fitness G(v)", range=_fitness_range(result, limits),
                        row=1, col=2)
    figure.update_layout(
        template="plotly_white", height=760,
        title={"text": steps[0]["args"][1]["title.text"], "y": 0.98, "x": 0.03},
        margin={"t": 160, "b": 120},
        legend={"orientation": "h", "y": 1.2, "x": 0.0},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Time: "},
                  "pad": {"t": 55}, "steps": steps}],
    )
    return figure
