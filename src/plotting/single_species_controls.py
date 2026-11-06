"""Reusable notebook controls for sequential population introductions."""

from dataclasses import asdict, replace

import ipywidgets as widgets
import numpy as np
from IPython.display import HTML, Markdown

from src.workflows.single_species_exploration import (
    model_display_name, settings_for_scenario, simulate_exploration,
)
from src.workflows.single_species_leader_exploration import (
    LeaderParameters, LeaderScanSettings, calculate_leader_analysis,
)
from src.plotting.single_species_exploration import landscape_time_slider, plot_leader_analysis
from src.workflows.single_species_exploration import simulate_multi_population_exploration
from src.plotting.single_species_exploration import multi_population_landscape_time_slider


def clear_population_addition_results(stages, start=2):
    """Clear stored results and outputs from ``start`` onward, keeping controls."""
    for count, stage in stages.items():
        if count >= start:
            stage["experiment"].clear()
            stage["output"].outputs = ()


def population_addition_controls(
    count, initial_population=10.0, initial_trait=10.0, *, stages,
    get_previous_result, get_fitness_limits, style=None, layout=None,
):
    """Build a sequential introduction panel without relying on notebook globals.

    ``stages`` stores panels by population count. Callbacks read the latest
    preceding result and display bounds when Run is clicked. The returned
    dictionary includes ``widget`` for display, editable controls, ``run``,
    and an ``experiment`` dictionary containing the last result or error.
    A preceding single-population result may contain one-dimensional arrays;
    community results contain one row per population. Both are supported.
    Rebuilding or rerunning a panel clears results for all later stages.
    """
    clear_population_addition_results(stages, count)
    style = {} if style is None else style
    layout = widgets.Layout(width="400px") if layout is None else layout
    population_control = widgets.FloatText(
        value=initial_population, description=f"Population {count} initial abundance",
        style=style, layout=layout,
    )
    trait_control = widgets.FloatText(
        value=initial_trait, description=f"Population {count} initial trait",
        style=style, layout=layout,
    )
    button = widgets.Button(
        description=f"Run {count}-population simulation", button_style="primary",
        icon="play", layout=widgets.Layout(width="280px"),
    )
    output = widgets.Output()
    record = {}

    def run(_=None):
        button.disabled = True
        record.clear()
        output.outputs = ()
        clear_population_addition_results(stages, count + 1)
        try:
            previous_result = get_previous_result()
            if previous_result is None:
                output.append_stdout(f"Run a successful {count - 1}-population simulation first.\n")
                return
            if (not np.isfinite(population_control.value) or population_control.value < 0
                    or not np.isfinite(trait_control.value) or trait_control.value < 0):
                raise ValueError("The new population's abundance and trait must be finite and nonnegative.")
            settings = previous_result.settings
            populations = np.atleast_2d(previous_result.population)[:, -1].astype(float).tolist()
            traits = np.atleast_2d(previous_result.trait)[:, -1].astype(float).tolist()
            populations.append(population_control.value)
            traits.append(trait_control.value)
            limits = get_fitness_limits()
            if limits is not None and (
                not np.all(np.isfinite(limits)) or limits[0] >= limits[1]
            ):
                raise ValueError("Minimum fitness must be finite and less than maximum fitness.")
            result = simulate_multi_population_exploration(settings, populations, traits)
            figure = multi_population_landscape_time_slider(result, limits)
            record.update(result=result, reference=None, interactive=figure,
                          initial_populations=populations, initial_traits=traits)
            ending = []
            for species in range(count):
                state = "extinct" if result.extinct[species, -1] else "above the extinction threshold"
                ending.append(
                    f"Population {species + 1}: x={result.population[species, -1]:.5g}, "
                    f"u={result.trait[species, -1]:.5g} ({state})"
                )
            output.append_display_data(Markdown(
                f"**{model_display_name(settings.model_name)} · E={settings.E:g} · "
                f"{count} populations · evolution {'on' if settings.evo_dyn else 'off'}**  \n"
                f"Finished in {result.elapsed_seconds:.2f} s. "
                f"Populations 1–{count - 1} start at the previous run's endpoint. "
                f"population {count} starts at (x={populations[-1]:.5g}, u={traits[-1]:.5g}). "
                f"{' · '.join(ending)}. The duration does not certify equilibrium."
            ))
            output.append_display_data(HTML(figure.to_html(
                full_html=False, include_plotlyjs=True,
            )))
        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
            record.clear()
            record["error"] = str(error)
            output.append_stdout(f"{count}-population experiment could not be completed: {error}\n")
        finally:
            button.disabled = False

    button.on_click(run)
    stage = dict(experiment=record, output=output, population=population_control,
                 trait=trait_control, button=button, run=run)
    stage["widget"] = widgets.VBox([population_control, trait_control, button, output])
    stages[count] = stage
    return stage


def single_species_experiment_controls(initial_settings, starting_scenario, *, stages, leader_panels):
    """Build the population experiment UI with shared dependent-result registries.

    The returned controls and callbacks are live objects. Running the experiment
    clears every registered leader panel and population-addition result.
    Construction does not start a simulation or display the widget.
    """
    style = {"description_width": "200px"}
    field_layout = widgets.Layout(width="400px")
    controls = {}

    def number(name, label, integer=False):
        field = widgets.IntText if integer else widgets.FloatText
        controls[name] = field(value=getattr(initial_settings, name), description=label,
                               style=style, layout=field_layout)
        return controls[name]

    def toggle(name, label):
        controls[name] = widgets.Checkbox(value=getattr(initial_settings, name),
                                         description=label, indent=False,
                                         style=style, layout=field_layout)
        return controls[name]

    controls["model_name"] = widgets.Dropdown(
        options=[("Model 1 - cost in growth", "model1"),
                 ("Model 2 - cost in carrying capacity", "model3")],
        value=initial_settings.model_name, description="Model variant", style=style,
        layout=widgets.Layout(width="460px"),
    )
    basic = widgets.VBox([
        controls["model_name"], number("E", "Effective resource support E"),
        number("initial_population", "Initial abundance x"),
        number("initial_trait", "Initial trait u"), toggle("evo_dyn", "Enable evolution"),
    ])
    demography = widgets.VBox([
        number("r_max", "r_max"), number("K", "K"), number("g", "g (trait cost)"),
        number("d", "d (mortality)"), number("sigma", "sigma (evolution rate)"),
        number("alpha", "alpha"), number("beta", "beta"),
    ])
    competition = widgets.VBox([
        toggle("fd", "Trait-dependent competition"),
        number("beta_bully", "beta_bully"), number("sigma_a_squared", "sigma_a_squared"),
    ])
    sampling = widgets.VBox([
        number("duration", "Duration"), number("frame_count", "Time samples", integer=True),
        number("trait_min", "Minimum mutant trait"), number("trait_max", "Maximum mutant trait"),
        number("trait_points", "Mutant-trait samples", integer=True),
        number("extinction_threshold", "Extinction threshold"),
    ])
    compare_control = widgets.Checkbox(value=True, description="Compare with evolution disabled",
                                       indent=False, layout=widgets.Layout(width="400px"))
    full_range = widgets.Checkbox(value=True, description="Full fitness range", indent=False)
    y_min = widgets.FloatText(value=-0.2, description="Minimum fitness", style=style)
    y_max = widgets.FloatText(value=0.2, description="Maximum fitness", style=style)
    display_options = widgets.VBox([compare_control, full_range, y_min, y_max])
    sections = widgets.Accordion(children=[basic, demography, competition, sampling, display_options])
    for index, title in enumerate([
        "Model and starting state", "Demographic parameters", "Competition",
        "Simulation and sampling", "Comparison and fitness display",
    ]):
        sections.set_title(index, title)
    sections.selected_index = 0
    scenario_control = widgets.Dropdown(
        options=[("Model 1 - only density dependence", "model1_only_density_dependence"),
                 ("Model 1 - density and frequency dependence", "model1_density_and_frequency_dependence"),
                 ("Model 2 - density and frequency dependence", "model2_density_and_frequency_dependence")],
        value=starting_scenario, layout=widgets.Layout(width="380px"),
    )
    load_button = widgets.Button(description="Load example")
    run_button = widgets.Button(description="Run simulation", button_style="primary",
                                 icon="play")
    output_area = widgets.Output()
    experiment = {}  # Holds the parameters and results of the last successful run.

    def load_example(_):
        loaded = settings_for_scenario(scenario_control.value)
        for name, value in asdict(loaded).items():
            controls[name].value = value
        full_range.value = loaded.model_name != "model3"


    def run_experiment(_=None):
        run_button.disabled = True
        experiment.clear()
        for panel in leader_panels.values():
            panel["experiment"].clear()
            panel["output"].outputs = ()
        clear_population_addition_results(stages)
        # Own the output explicitly: replace the previous result on every run.
        # Displaying a Figure through multiple renderers can create duplicate views.
        output_area.outputs = ()
        try:
            settings = replace(initial_settings, **{key: control.value
                                                   for key, control in controls.items()})
            limits = None if full_range.value else (y_min.value, y_max.value)
            if limits is not None and (not np.all(np.isfinite(limits)) or limits[0] >= limits[1]):
                raise ValueError("Minimum fitness must be finite and less than maximum fitness.")
            result = simulate_exploration(settings)
            reference = (simulate_exploration(replace(settings, evo_dyn=False))
                         if compare_control.value and settings.evo_dyn else None)
            interactive = landscape_time_slider(result, limits, reference=reference)
            experiment.update(result=result, reference=reference,
                              interactive=interactive, fitness_limits=limits)
            status = "extinct" if result.extinct[-1] else "above the extinction threshold"
            elapsed = result.elapsed_seconds + (reference.elapsed_seconds if reference else 0)
            output_area.append_display_data(Markdown(
                f"**{model_display_name(settings.model_name)} · E={settings.E:g} · evolution "
                f"{'on' if settings.evo_dyn else 'off'}**  \n"
                f"Finished in {elapsed:.2f} s; final x={result.population[-1]:.5g}, "
                f"u={result.trait[-1]:.5g} ({status}). "
                f"{len(result.time)} slider positions. "
                "The time limit does not certify equilibrium."
            ))
            # A single HTML output avoids MIME/HTML renderer duplication and also
            # works offline. The figure itself is retained above for export.
            output_area.append_display_data(HTML(interactive.to_html(
                full_html=False, include_plotlyjs=True,
            )))
        except (ValueError, RuntimeError, OverflowError, FloatingPointError) as error:
            experiment.clear()
            experiment["error"] = str(error)
            output_area.append_stdout(f"Experiment could not be completed: {error}\n")
        finally:
            run_button.disabled = False


    load_button.on_click(load_example)
    run_button.on_click(run_experiment)

    def get_fitness_limits():
        return None if full_range.value else (y_min.value, y_max.value)

    return dict(
        widget=widgets.VBox([
            widgets.HBox([scenario_control, load_button]), sections, run_button, output_area,
        ]),
        controls=controls, experiment=experiment, run=run_experiment,
        load_example=load_example, scenario=scenario_control, output=output_area,
        compare=compare_control, full_range=full_range, y_min=y_min, y_max=y_max,
        get_fitness_limits=get_fitness_limits, style=style, layout=field_layout,
    )


def leader_analysis_controls(initial_settings, *, experiment):
    """Build leader analysis controls that read the latest population experiment.

    Construction does not run the scan. The returned dictionary exposes its
    widget, coefficients, scan settings, progress, callback, and saved results.
    """
    leader_experiment = {}
    leader_output = widgets.Output()
    leader_button = widgets.Button(description="Run leader analysis", button_style="primary")
    leader_progress = widgets.IntProgress(value=0, min=0, max=200, description="Support sweep")
    leader_style = {"description_width": "210px"}
    leader_layout = widgets.Layout(width="410px")

    def leader_field(value, label, integer=False):
        field = widgets.IntText if integer else widgets.FloatText
        return field(value=value, description=label, style=leader_style, layout=leader_layout)

    payoff_defaults = LeaderParameters()
    leader_payoff_controls = {
        name: leader_field(value, name)
        for name, value in asdict(payoff_defaults).items()
    }
    scan_defaults = LeaderScanSettings()
    scan_labels = {
        "E_min": "Minimum resource support E", "E_max": "Maximum resource support E",
        "E_points": "Resource-support samples", "trait_min": "Minimum trait for leader BR",
        "trait_max": "Maximum trait for leader BR", "trait_points": "Leader BR trait samples",
        "convergence_tolerance": "Stationary-response tolerance",
        "nash_tolerance": "Nash residual tolerance",
    }
    leader_scan_controls = {
        name: leader_field(value, scan_labels[name], integer=name in ("E_points", "trait_points"))
        for name, value in asdict(scan_defaults).items()
    }
    leader_duration = leader_field(initial_settings.duration, "Species-response duration")
    leader_sections = widgets.Accordion(children=[
        widgets.VBox(list(leader_payoff_controls.values())),
        widgets.VBox([leader_duration, *leader_scan_controls.values()]),
    ])
    leader_sections.set_title(0, "Leader payoff coefficients")
    leader_sections.set_title(1, "Response sweep and candidate checks")
    leader_sections.selected_index = 0

    def run_leader_analysis(_=None):
        leader_output.outputs = ()
        leader_experiment.clear()
        if "result" not in experiment:
            leader_output.append_stdout("Run a successful population experiment first.\n")
            return
        leader_button.disabled = True
        try:
            source = experiment["result"]
            biology = replace(source.settings, duration=leader_duration.value, evo_dyn=True)
            economics = LeaderParameters(**{key: control.value
                                            for key, control in leader_payoff_controls.items()})
            scan = LeaderScanSettings(**{key: control.value
                                        for key, control in leader_scan_controls.items()})
            leader_progress.max = max(3, scan.E_points)
            leader_progress.value = 0
            def update_progress(done, total):
                leader_progress.max = total
                leader_progress.value = done
            analysis = calculate_leader_analysis(biology, economics, scan, progress=update_progress)
            figure = plot_leader_analysis(analysis)
            leader_experiment.update(source=source, analysis=analysis, figure=figure)
            lines = [
                f"**{model_display_name(biology.model_name)} · leader analysis**  ",
                f"Computed in {analysis.elapsed_seconds:.2f} s using a response duration of {biology.duration:g}.  ",
                f"Stationary sampled responses: {int(analysis.stationary.sum())}/{len(analysis.E)}. "
                "Open payoff circles have not reached equilibrium, run a longer simulation to ensure convergence.  ",
                "", "| Candidate | E | x | u | Q | Stationary response | BR residual |",
                "|---|---:|---:|---:|---:|:---:|---:|",
            ]
            for name, point in [("Stackelberg (sampled)", analysis.stackelberg),
                                *(("Nash", point) for point in analysis.nash_points)]:
                residual = "—" if point.residual is None else f"{point.residual:.3g}"
                lines.append(f"| {name} | {point.E:.5g} | {point.x:.5g} | {point.u:.5g} "
                             f"| {point.payoff:.5g} | {'yes' if point.stationary else 'no'} | {residual} |")
            if not analysis.nash_points:
                lines.extend(["", "No living Nash candidate was resolved, so no fixed-trait payoff comparison is drawn. "
                              "The best-response curves may have no intersection. Refine the grid or extend "
                              "the response duration to distinguish this from an unresolved numerical intersection."])
            if analysis.failures:
                lines.extend(["", "Some evaluations failed and were excluded:"])
                lines.extend(f"- {key}: {message}" for key, message in analysis.failures.items())
            leader_output.append_display_data(Markdown("\n".join(lines)))
            leader_output.append_display_data(HTML(figure.to_html(full_html=False, include_plotlyjs=True)))
        except (ValueError, RuntimeError, ArithmeticError) as error:
            leader_experiment.clear()
            leader_output.append_stdout(f"Leader analysis could not be completed: {error}\n")
        finally:
            leader_button.disabled = False

    leader_button.on_click(run_leader_analysis)

    return dict(
        widget=widgets.VBox([
            leader_sections, widgets.HBox([leader_button, leader_progress]), leader_output,
        ]),
        experiment=leader_experiment, output=leader_output, run=run_leader_analysis,
        button=leader_button, progress=leader_progress, payoff_controls=leader_payoff_controls,
        scan_controls=leader_scan_controls, duration=leader_duration,
    )
