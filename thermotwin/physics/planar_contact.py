"""Conservative two-dimensional transient reference for a contacted plate.

Coordinates are ``0 <= x <= width`` and ``0 <= z <= thickness``.  The
constant-property continuum problem is

``density * specific_heat * T_t = k * (T_xx + T_zz) + source``.

Both sides are insulated.  A prescribed inward top flux obeys ``k T_z = q``
at ``z=thickness``.  At the bottom, ``k T_z = h (T - T_TE)`` on the known
contact interval, and ``T_z=0`` elsewhere.  Positive contact heat leaves the
plate.  The out-of-plane depth only converts fluxes to powers; it cancels
from the temperature solution.  No thermoelectric constitutive terms or
unknown source parameters are hidden in this deliberately bounded model.

The independent reference uses cell-centred finite volumes, a sparse
conductance matrix, and implicit theta time integration.  The Robin face
includes the half-cell conduction resistance.  Exact interval/face overlap
handles a contact edge that does not coincide with a mesh edge.  Integrating
the boundary powers with the same time quadrature closes the global energy
balance to linear-solver roundoff.  NumPy and SciPy are optional dependencies
of this module; importing the base thermotwin package does not import them.
"""

from dataclasses import dataclass
import math
import time as _time
from typing import Callable, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve


ScalarSchedule = Union[float, Callable[[float], float]]
SpatialSource = Union[float, Callable[[np.ndarray, np.ndarray, float], np.ndarray]]


@dataclass(frozen=True)
class PlanarContactConfig:
    """Known geometry and constant material properties, in SI units.

    Defaults describe a synthetic steel-like plate, not measured hardware.
    Contact conductance is supplied separately to the model as the one
    candidate inverse parameter, in W/(m² K).
    """

    width: float = 0.04
    thickness: float = 0.004
    depth: float = 0.01
    thermal_conductivity: float = 15.0
    density: float = 7800.0
    specific_heat: float = 500.0
    contact_start: float = 0.008
    contact_end: float = 0.024

    def __post_init__(self) -> None:
        for name in (
            "width", "thickness", "depth", "thermal_conductivity", "density",
            "specific_heat",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (
            math.isfinite(self.contact_start)
            and math.isfinite(self.contact_end)
            and 0.0 <= self.contact_start < self.contact_end <= self.width
        ):
            raise ValueError("contact interval must have positive width inside the plate")

    @property
    def diffusivity(self) -> float:
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def volume(self) -> float:
        return self.width * self.thickness * self.depth


@dataclass(frozen=True)
class PlanarPowerBalance:
    """Instantaneous powers in W; storage is positive for warming."""

    top_heat_input: float
    contact_heat_output: float
    volumetric_heat_input: float
    storage_rate: float

    @property
    def residual(self) -> float:
        return self.storage_rate - (
            self.top_heat_input - self.contact_heat_output
            + self.volumetric_heat_input
        )


def _axis_indices(coordinates: np.ndarray, values: np.ndarray):
    """Linear interpolation indices, including half-cell extrapolation."""
    if len(coordinates) == 1:
        zeros = np.zeros(values.shape, dtype=int)
        return zeros, zeros, np.zeros(values.shape)
    upper = np.clip(np.searchsorted(coordinates, values), 1, len(coordinates) - 1)
    lower = upper - 1
    weight = (values - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
    return lower, upper, weight


@dataclass(frozen=True)
class PlanarContactTrajectory:
    """Saved cell temperatures and conservative cumulative boundary heats.

    Temperature shape is ``(time, z, x)``.  Cumulative heat arrays are in J.
    Sensor sampling interpolates linearly in time and bilinearly in space.
    At the top surface it uses the imposed flux and half-cell conduction
    distance.  Other boundaries use cell-centre extrapolation, so interior
    sensors are preferred near a mixed boundary corner.  The energy
    diagnostic does not use this interpolation.
    """

    config: PlanarContactConfig
    times: np.ndarray
    x: np.ndarray
    z: np.ndarray
    temperature: np.ndarray
    stored_energy: np.ndarray
    cumulative_top_heat: np.ndarray
    cumulative_contact_heat: np.ndarray
    cumulative_source_heat: np.ndarray
    top_heat_flux: np.ndarray

    @property
    def energy_balance_residual(self) -> np.ndarray:
        return (
            self.stored_energy - self.stored_energy[0]
            - self.cumulative_top_heat + self.cumulative_contact_heat
            - self.cumulative_source_heat
        )

    def sample(self, x, z, time) -> np.ndarray:
        """Sample temperatures at broadcast-compatible physical coordinates."""
        xp, zp, tp = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(z, dtype=float),
            np.asarray(time, dtype=float),
        )
        if not all(np.isfinite(value).all() for value in (xp, zp, tp)):
            raise ValueError("sample coordinates must be finite")
        if (
            (xp < 0.0).any() or (xp > self.config.width).any()
            or (zp < 0.0).any() or (zp > self.config.thickness).any()
            or (tp < self.times[0]).any() or (tp > self.times[-1]).any()
        ):
            raise ValueError("sample coordinates must lie inside the saved domain")
        xl, xu, xw = _axis_indices(self.x, xp)
        zl, zu, zw = _axis_indices(self.z, zp)
        tl, tu, tw = _axis_indices(self.times, tp)

        def spatial(ti):
            lower = (
                (1.0 - xw) * self.temperature[ti, zl, xl]
                + xw * self.temperature[ti, zl, xu]
            )
            upper = (
                (1.0 - xw) * self.temperature[ti, zu, xl]
                + xw * self.temperature[ti, zu, xu]
            )
            interior = (1.0 - zw) * lower + zw * upper
            half_cell = self.config.thickness / (2.0 * len(self.z))
            top_face = (
                (1.0 - xw) * self.temperature[ti, -1, xl]
                + xw * self.temperature[ti, -1, xu]
                + self.top_heat_flux[ti] * half_cell / self.config.thermal_conductivity
            )
            # The declared initial field wins at t=0, including for an
            # intentionally incompatible step boundary used in verification.
            top_face = np.where(ti == 0, interior, top_face)
            return np.where(zp == self.config.thickness, top_face, interior)

        return (1.0 - tw) * spatial(tl) + tw * spatial(tu)


class PlanarContactModel:
    """Cell-centred finite-volume model, independent of any PINN code.

    ``contact_conductance=0`` is allowed for the insulated verification limit;
    an inverse contact problem must use a strictly positive candidate value.
    Each cell has a volume and thermal capacity.  Internal faces use equal
    and opposite heat powers, including those adjacent to the contact edge.
    """

    def __init__(
        self, config: PlanarContactConfig = PlanarContactConfig(), *,
        nx: int = 40, nz: int = 8, contact_conductance: float = 1000.0,
    ) -> None:
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
               for n in (nx, nz)):
            raise ValueError("nx and nz must be positive integers")
        if not math.isfinite(contact_conductance) or contact_conductance < 0.0:
            raise ValueError("contact_conductance must be finite and nonnegative")
        self.config = config
        self.nx, self.nz = int(nx), int(nz)
        self.contact_conductance = float(contact_conductance)
        self.dx, self.dz = config.width / nx, config.thickness / nz
        self.x = (np.arange(nx) + 0.5) * self.dx
        self.z = (np.arange(nz) + 0.5) * self.dz
        self.x_grid, self.z_grid = np.meshgrid(self.x, self.z)
        self.cell_volumes = np.full((nz, nx), self.dx * self.dz * config.depth)
        self.heat_capacity = self.cell_volumes * config.density * config.specific_heat
        self.top_surface_areas = np.full(nx, self.dx * config.depth)
        edges = np.linspace(0.0, config.width, nx + 1)
        overlaps = np.clip(
            np.minimum(edges[1:], config.contact_end)
            - np.maximum(edges[:-1], config.contact_start), 0.0, self.dx,
        )
        self.contact_surface_areas = overlaps * config.depth
        if contact_conductance == 0.0:
            self.bottom_conductances = np.zeros(nx)
        else:
            self.bottom_conductances = self.contact_surface_areas / (
                1.0 / contact_conductance
                + self.dz / (2.0 * config.thermal_conductivity)
            )
        self._horizontal_conductance = (
            config.thermal_conductivity * self.dz * config.depth / self.dx
        )
        self._vertical_conductance = (
            config.thermal_conductivity * self.dx * config.depth / self.dz
        )
        indices = np.arange(nx * nz).reshape(nz, nx)
        rows, columns, data = [], [], []
        diagonal = np.zeros(nx * nz)
        for left, right, conductance in (
            (indices[:, :-1].ravel(), indices[:, 1:].ravel(),
             self._horizontal_conductance),
            (indices[:-1, :].ravel(), indices[1:, :].ravel(),
             self._vertical_conductance),
        ):
            rows.extend(left)
            columns.extend(right)
            data.extend(np.full(len(left), conductance))
            rows.extend(right)
            columns.extend(left)
            data.extend(np.full(len(left), conductance))
            np.add.at(diagonal, left, -conductance)
            np.add.at(diagonal, right, -conductance)
        diagonal[:nx] -= self.bottom_conductances
        rows.extend(np.arange(nx * nz))
        columns.extend(np.arange(nx * nz))
        data.extend(diagonal)
        self.conductance_matrix = sparse.coo_matrix(
            (data, (rows, columns)), shape=(nx * nz, nx * nz),
        ).tocsr()
        self._capacity_matrix = sparse.diags(self.heat_capacity.ravel(), format="csc")

    def _temperature_array(self, temperature) -> np.ndarray:
        if callable(temperature):
            temperature = temperature(self.x_grid, self.z_grid)
        result = np.broadcast_to(np.asarray(temperature, dtype=float), (self.nz, self.nx))
        if not np.isfinite(result).all():
            raise ValueError("temperature must be finite")
        return result

    @staticmethod
    def _schedule_value(schedule: ScalarSchedule, time: float, name: str) -> float:
        value = float(schedule(time) if callable(schedule) else schedule)
        if not math.isfinite(value):
            raise ValueError(f"{name} must return finite scalars")
        return value

    def _forcing(
        self, time: float, top_heat_flux: ScalarSchedule,
        te_temperature: ScalarSchedule, volumetric_source: SpatialSource,
    ):
        top = self._schedule_value(top_heat_flux, time, "top_heat_flux")
        te = self._schedule_value(te_temperature, time, "te_temperature")
        source = (
            volumetric_source(self.x_grid, self.z_grid, time)
            if callable(volumetric_source) else volumetric_source
        )
        source = np.broadcast_to(np.asarray(source, dtype=float), (self.nz, self.nx))
        if not np.isfinite(source).all():
            raise ValueError("volumetric_source must return finite cell values")
        source_power = source * self.cell_volumes
        forcing = source_power.copy()
        forcing[-1, :] += top * self.top_surface_areas
        forcing[0, :] += te * self.bottom_conductances
        return forcing.ravel(), top, te, float(source_power.sum())

    def interface_heat_powers(self, temperature):
        """Return internal face powers in +x and +z directions, in W.

        Arrays have shapes ``(nz,nx-1)`` and ``(nz-1,nx)`` respectively.
        The same face power is removed from one cell and added to its neighbour.
        """
        values = self._temperature_array(temperature)
        return (
            self._horizontal_conductance * (values[:, :-1] - values[:, 1:]),
            self._vertical_conductance * (values[:-1, :] - values[1:, :]),
        )

    def rhs(
        self, time: float, temperature, *, top_heat_flux: ScalarSchedule = 0.0,
        te_temperature: ScalarSchedule = 300.0, volumetric_source: SpatialSource = 0.0,
    ) -> np.ndarray:
        """Return all cell temperature rates in K/s."""
        values = self._temperature_array(temperature)
        forcing, _, _, _ = self._forcing(time, top_heat_flux, te_temperature, volumetric_source)
        return (
            (self.conductance_matrix @ values.ravel() + forcing)
            / self.heat_capacity.ravel()
        ).reshape(self.nz, self.nx)

    def power_balance(
        self, time: float, temperature, *, top_heat_flux: ScalarSchedule = 0.0,
        te_temperature: ScalarSchedule = 300.0, volumetric_source: SpatialSource = 0.0,
    ) -> PlanarPowerBalance:
        """Evaluate storage and boundary powers without sensor interpolation."""
        values = self._temperature_array(temperature)
        forcing, top, te, source = self._forcing(
            time, top_heat_flux, te_temperature, volumetric_source,
        )
        return PlanarPowerBalance(
            top * float(self.top_surface_areas.sum()),
            float(np.dot(self.bottom_conductances, values[0, :] - te)),
            source,
            float((self.conductance_matrix @ values.ravel() + forcing).sum()),
        )

    def steady_state(
        self, *, top_heat_flux: float = 0.0, te_temperature: float = 300.0,
        volumetric_source: SpatialSource = 0.0,
    ) -> np.ndarray:
        """Solve the linear steady state, requiring positive contact conductance."""
        if self.contact_conductance == 0.0:
            raise ValueError("steady state requires positive contact conductance")
        forcing, _, _, _ = self._forcing(0.0, top_heat_flux, te_temperature, volumetric_source)
        return spsolve(-self.conductance_matrix, forcing).reshape(self.nz, self.nx)

    def solve(
        self, times, *, initial_temperature=300.0, top_heat_flux: ScalarSchedule = 0.0,
        te_temperature: ScalarSchedule = 300.0, volumetric_source: SpatialSource = 0.0,
        max_step: float = 0.25, theta: float = 0.5, breakpoints=(),
    ) -> PlanarContactTrajectory:
        """Integrate from time zero at requested output times, in seconds.

        ``theta=.5`` is Crank-Nicolson (second order for smooth schedules),
        and ``theta=1`` is backward Euler (first order, strongly damping).
        Supply every schedule discontinuity in ``breakpoints``.  Integration
        steps end there and evaluate the forcing from inside each interval,
        so a jump never receives an artificial half-step of pre/post forcing.
        Outputs and breakpoints can refine the internal step partition.
        """
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all()
            or times[0] != 0.0 or (np.diff(times) <= 0.0).any()
        ):
            raise ValueError("times must start at zero and strictly increase")
        if not math.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step must be finite and positive")
        if not math.isfinite(theta) or not 0.5 <= theta <= 1.0:
            raise ValueError("theta must lie in [0.5, 1]")
        breakpoints = np.asarray(breakpoints, dtype=float)
        if breakpoints.ndim != 1 or not np.isfinite(breakpoints).all():
            raise ValueError("breakpoints must be a finite sequence")
        if (breakpoints < 0.0).any() or (breakpoints > times[-1]).any():
            raise ValueError("breakpoints must lie inside the time window")
        stops = np.unique(np.concatenate((times, breakpoints)))
        values = self._temperature_array(initial_temperature).copy().ravel()
        temperature = np.empty((len(times), self.nz, self.nx))
        temperature[0] = values.reshape(self.nz, self.nx)
        stored = np.empty(len(times))
        stored[0] = np.dot(self.heat_capacity.ravel(), values)
        heats = np.zeros((len(times), 3))
        cumulative = np.zeros(3)
        solvers = {}
        output_index = 1
        for left, right in zip(stops[:-1], stops[1:]):
            count = max(1, int(math.ceil((right - left) / max_step)))
            dt = (right - left) / count
            # A small rounding key reuses the factorization for equal steps
            # whose binary representations differ after output partitioning.
            key = round(dt, 14)
            if key not in solvers:
                solvers[key] = factorized(
                    self._capacity_matrix - theta * dt * self.conductance_matrix
                )
            solve_step = solvers[key]
            for step in range(count):
                t0, t1 = left + step * dt, left + (step + 1) * dt
                f0, top0, te0, src0 = self._forcing(
                    float(np.nextafter(t0, t1)), top_heat_flux, te_temperature,
                    volumetric_source,
                )
                f1, top1, te1, src1 = self._forcing(
                    float(np.nextafter(t1, t0)), top_heat_flux, te_temperature,
                    volumetric_source,
                )
                next_values = solve_step(
                    self.heat_capacity.ravel() * values
                    + (1.0 - theta) * dt * (self.conductance_matrix @ values)
                    + dt * ((1.0 - theta) * f0 + theta * f1)
                )
                if not np.isfinite(next_values).all():
                    raise FloatingPointError("planar temperature integration became nonfinite")
                contact0 = np.dot(self.bottom_conductances, values[:self.nx] - te0)
                contact1 = np.dot(self.bottom_conductances, next_values[:self.nx] - te1)
                cumulative += dt * (
                    (1.0 - theta) * np.array((
                        top0 * self.top_surface_areas.sum(), contact0, src0,
                    ))
                    + theta * np.array((
                        top1 * self.top_surface_areas.sum(), contact1, src1,
                    ))
                )
                values = next_values
            if output_index < len(times) and right == times[output_index]:
                temperature[output_index] = values.reshape(self.nz, self.nx)
                stored[output_index] = np.dot(self.heat_capacity.ravel(), values)
                heats[output_index] = cumulative
                output_index += 1
        return PlanarContactTrajectory(
            self.config, times.copy(), self.x.copy(), self.z.copy(), temperature,
            stored, heats[:, 0], heats[:, 1], heats[:, 2],
            np.asarray([
                self._schedule_value(top_heat_flux, float(time), "top_heat_flux")
                for time in times
            ]),
        )


def verify_planar_contact_reference(
    config: PlanarContactConfig = PlanarContactConfig(), *,
    conductances=(800.0, 1000.0),
    meshes=((80, 20), (160, 40), (320, 80)),
    reference_max_step: float = 0.0625,
    inference_max_step: float = 0.125,
    fine_max_step: float = 0.03125,
    duration: float = 120.0,
    observation_interval: float = 10.0,
    sensor_tolerance: float = 0.002,
    field_rms_tolerance: float = 0.005,
    field_max_tolerance: float = 0.01,
) -> dict:
    """Return JSON-safe disposable mesh/time/energy verification evidence.

    This does not create observations, noise, optimization runs, reports or
    named scientific cohorts.  It checks two prescribed smooth traces: a
    10 s ramp to a 295 K bath and 500 W/m² top flux, and a 15 s ramp to a
    293 K bath and 750 W/m² top flux.  Sensors are at x/width=.1,.5,.9 and
    z/thickness=.75, sampled at the observation cadence.  Fields are
    compared at those times plus explicit early outputs at 1, 2 and 5 s on
    a common 40-by-20 interior grid.  Field RMS is the unweighted RMS over
    that fixed space-time point set.  Mesh convergence holds the small
    time step fixed.  A separate time comparison holds the middle mesh fixed.  The
    default middle mesh/time step is the canonical truth, with the coarse
    mesh/time step used by the scalar finite-volume inverse baseline.

    Run the caller with BLAS/OpenMP thread counts set to one when sharing
    CPU resources.  Default verification normally takes seconds on one CPU.
    """
    started = _time.perf_counter()
    if len(meshes) != 3:
        raise ValueError("verification requires exactly three meshes")
    if (
        not math.isfinite(duration) or duration <= 0.0
        or not math.isfinite(observation_interval) or observation_interval <= 0.0
        or any(not math.isfinite(float(h)) or h <= 0.0 for h in conductances)
        or any(not math.isfinite(v) or v <= 0.0 for v in (
            reference_max_step, inference_max_step, fine_max_step,
            sensor_tolerance, field_rms_tolerance, field_max_tolerance,
        ))
    ):
        raise ValueError("verification values must be finite and positive")
    sensor_output_times = np.unique(np.append(
        np.arange(0.0, duration, observation_interval), duration,
    ))
    early_times = np.array([1.0, 2.0, 5.0])
    times = np.unique(np.concatenate((
        sensor_output_times, early_times[early_times <= duration],
    )))
    sensor_x = config.width * np.array([0.1, 0.5, 0.9])[None, :]
    sensor_z = 0.75 * config.thickness
    sensor_times = sensor_output_times[:, None]
    field_x, field_z = np.meshgrid(
        (np.arange(40) + 0.5) * config.width / 40.0,
        (np.arange(20) + 0.5) * config.thickness / 20.0,
    )
    field_times = times[:, None, None]
    traces = (
        ("ramp10_bath295_flux500", 10.0, 5.0, 500.0),
        ("ramp15_bath293_flux750", 15.0, 7.0, 750.0),
    )
    cases = []

    def comparison(candidate, reference):
        sensor_error = (
            candidate.sample(sensor_x, sensor_z, sensor_times)
            - reference.sample(sensor_x, sensor_z, sensor_times)
        )
        field_error = (
            candidate.sample(field_x, field_z, field_times)
            - reference.sample(field_x, field_z, field_times)
        )
        metrics = {
            "sensor_max_abs_k": float(np.max(np.abs(sensor_error))),
            "field_rms_k": float(np.sqrt(np.mean(field_error**2))),
            "field_max_abs_k": float(np.max(np.abs(field_error))),
        }
        metrics["passes"] = bool(
            metrics["sensor_max_abs_k"] <= sensor_tolerance
            and metrics["field_rms_k"] <= field_rms_tolerance
            and metrics["field_max_abs_k"] <= field_max_tolerance
        )
        return metrics

    for trace_name, ramp_time, bath_drop, top_flux in traces:
        for h in conductances:
            # Bind each trace's values into the schedule: no mutable loop
            # closure survives into a returned object or subsequent run.
            def te(t):
                return 300.0 + bath_drop * math.expm1(-t / ramp_time)

            def flux(t):
                return -top_flux * math.expm1(-t / ramp_time)

            def run(mesh, step):
                return PlanarContactModel(
                    config, nx=mesh[0], nz=mesh[1], contact_conductance=h,
                ).solve(
                    times, initial_temperature=300.0, te_temperature=te,
                    top_heat_flux=flux, max_step=step,
                )

            mesh_results = [run(mesh, fine_max_step) for mesh in meshes]
            truth = run(meshes[1], reference_max_step)
            inference = run(meshes[0], inference_max_step)
            all_results = mesh_results + [truth, inference]
            energy_abs = max(float(np.abs(r.energy_balance_residual).max()) for r in all_results)
            energy_scaled = max(
                float(np.abs(r.energy_balance_residual).max()) / max(
                    1.0,
                    float(np.abs(r.cumulative_top_heat).max()),
                    float(np.abs(r.cumulative_contact_heat).max()),
                ) for r in all_results
            )
            mesh_differences = [
                comparison(mesh_results[i], mesh_results[i + 1]) for i in range(2)
            ]
            truth_difference = comparison(truth, mesh_results[-1])
            inference_difference = comparison(inference, truth)
            time_difference = comparison(truth, mesh_results[1])
            cases.append({
                "trace": trace_name,
                "contact_conductance_w_m2k": float(h),
                "mesh_successive_differences": mesh_differences,
                "reference_vs_finest": truth_difference,
                "inference_vs_reference": inference_difference,
                "reference_time_halving": time_difference,
                "energy_residual_max_abs_j": energy_abs,
                "energy_residual_scaled": energy_scaled,
                "passes": bool(
                    truth_difference["passes"] and inference_difference["passes"]
                    and time_difference["passes"] and energy_scaled <= 1e-8
                ),
            })
    return {
        "verification_kind": "disposable_reference_solver_verification",
        "passes": bool(cases and all(case["passes"] for case in cases)),
        "meshes_nx_nz": [list(mesh) for mesh in meshes],
        "mesh_comparison_max_step_s": float(fine_max_step),
        "reference_max_step_s": float(reference_max_step),
        "inference_max_step_s": float(inference_max_step),
        "sensor_x_m": sensor_x[0].tolist(),
        "sensor_z_m": float(sensor_z),
        "sample_times_s": times.tolist(),
        "sensor_sample_times_s": sensor_output_times.tolist(),
        "field_sample_times_s": times.tolist(),
        "field_grid_nx_nz": [40, 20],
        "field_rms_weighting": "unweighted_fixed_space_time_point_set",
        "criteria": {
            "sensor_max_abs_k": float(sensor_tolerance),
            "field_rms_k": float(field_rms_tolerance),
            "field_max_abs_k": float(field_max_tolerance),
            "energy_residual_scaled": 1e-8,
        },
        "cases": cases,
        "runtime_seconds": float(_time.perf_counter() - started),
    }
