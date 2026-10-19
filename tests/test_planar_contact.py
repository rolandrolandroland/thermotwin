"""Independent conservation, analytic-limit and convergence checks for 2D FVM."""

from dataclasses import replace
import unittest

try:
    import numpy as np

    from thermotwin.physics.planar_contact import (
        PlanarContactConfig,
        PlanarContactModel,
    )
except ModuleNotFoundError:
    np = None


@unittest.skipIf(np is None, "optional NumPy/SciPy dependencies are not installed")
class PlanarContactTests(unittest.TestCase):
    def test_geometry_tracks_volumes_and_partial_contact_face_areas(self):
        model = PlanarContactModel(nx=7, nz=3)
        self.assertAlmostEqual(model.cell_volumes.sum(), model.config.volume)
        self.assertAlmostEqual(
            model.contact_surface_areas.sum(),
            (model.config.contact_end - model.config.contact_start) * model.config.depth,
        )
        self.assertEqual(np.count_nonzero(model.contact_surface_areas), 4)
        self.assertTrue((model.contact_surface_areas <= model.top_surface_areas).all())
        self.assertAlmostEqual(
            model.heat_capacity.sum(),
            model.config.volume * model.config.density * model.config.specific_heat,
        )

    def test_internal_face_powers_cancel_cell_by_cell_and_close_boundary_balance(self):
        model = PlanarContactModel(nx=9, nz=5, contact_conductance=800.0)
        temperature = 300.0 + 7.0 * model.x_grid / model.config.width - 3.0 * (
            model.z_grid / model.config.thickness
        ) ** 2
        horizontal, vertical = model.interface_heat_powers(temperature)
        conductive = np.zeros_like(temperature)
        conductive[:, :-1] -= horizontal
        conductive[:, 1:] += horizontal
        conductive[:-1, :] -= vertical
        conductive[1:, :] += vertical
        reconstructed = conductive.copy()
        reconstructed[0, :] -= model.bottom_conductances * (temperature[0, :] - 292.0)
        reconstructed[-1, :] += 600.0 * model.top_surface_areas
        reconstructed += 130.0 * model.cell_volumes
        rhs = model.rhs(
            0.0, temperature, te_temperature=292.0, top_heat_flux=600.0,
            volumetric_source=130.0,
        )
        np.testing.assert_allclose(rhs * model.heat_capacity, reconstructed, atol=3e-12)
        self.assertAlmostEqual(conductive.sum(), 0.0, places=12)
        balance = model.power_balance(
            0.0, temperature, te_temperature=292.0, top_heat_flux=600.0,
            volumetric_source=130.0,
        )
        self.assertGreater(balance.contact_heat_output, 0.0)
        self.assertAlmostEqual(balance.residual, 0.0, places=11)

    def test_full_contact_steady_solution_is_exact_one_dimensional_extrusion(self):
        config = replace(PlanarContactConfig(), contact_start=0.0, contact_end=0.04)
        for nx, nz in ((1, 1), (7, 4), (11, 9)):
            with self.subTest(nx=nx, nz=nz):
                model = PlanarContactModel(config, nx=nx, nz=nz, contact_conductance=800.0)
                actual = model.steady_state(top_heat_flux=1200.0, te_temperature=292.0)
                exact = 292.0 + 1200.0 / 800.0 + 1200.0 * model.z_grid / 15.0
                np.testing.assert_allclose(actual, exact, atol=2e-10, rtol=0.0)
                self.assertAlmostEqual(
                    model.power_balance(0.0, actual, top_heat_flux=1200.0,
                                        te_temperature=292.0).residual,
                    0.0, places=10,
                )
                out = model.solve(
                    [0.0, 1.0], initial_temperature=actual,
                    top_heat_flux=1200.0, te_temperature=292.0,
                )
                np.testing.assert_allclose(out.temperature[-1], exact, atol=2e-10, rtol=0.0)
                self.assertAlmostEqual(
                    float(out.sample(0.02, config.thickness, 1.0)),
                    292.0 + 1200.0 / 800.0 + 1200.0 * config.thickness / 15.0,
                    places=9,
                )

    def test_theta_integration_closes_global_energy_for_both_heat_flow_signs(self):
        model = PlanarContactModel(nx=13, nz=5)
        times = [0.0, 0.3, 1.2, 3.0, 7.0]
        for theta in (0.5, 1.0):
            with self.subTest(theta=theta):
                out = model.solve(
                    times, initial_temperature=300.0,
                    top_heat_flux=lambda t: 500.0 * np.sin(t),
                    te_temperature=lambda t: 300.0 + 8.0 * np.cos(t),
                    volumetric_source=lambda x, z, t: 1e4 * np.cos(t) * (1 + x),
                    max_step=0.11, theta=theta,
                )
                self.assertLess(abs(out.energy_balance_residual).max(), 2e-9)
                self.assertLess(out.cumulative_contact_heat[1], 0.0)

    def test_schedule_breakpoint_prevents_spurious_half_step_of_forcing(self):
        model = PlanarContactModel(nx=3, nz=2, contact_conductance=0.0)
        out = model.solve(
            [0.0, 1.0], initial_temperature=300.0,
            top_heat_flux=lambda t: 0.0 if t < 0.4 else 1000.0,
            breakpoints=[0.4], max_step=1.0,
        )
        expected_heat = 0.6 * 1000.0 * model.config.width * model.config.depth
        self.assertAlmostEqual(out.cumulative_top_heat[-1], expected_heat, places=14)
        self.assertAlmostEqual(
            out.stored_energy[-1] - out.stored_energy[0], expected_heat, places=9,
        )

    @staticmethod
    def _unit_config():
        return PlanarContactConfig(
            width=1.0, thickness=0.7, depth=0.2,
            thermal_conductivity=0.8, density=2.0, specific_heat=1.5,
            contact_start=0.0, contact_end=1.0,
        )

    def test_two_dimensional_neumann_mode_has_second_order_mesh_convergence(self):
        config = self._unit_config()
        rate = config.diffusivity * np.pi**2 * (1.0 + 1.0 / config.thickness**2)
        errors = []
        for nx, nz in ((10, 7), (20, 14), (40, 28)):
            model = PlanarContactModel(config, nx=nx, nz=nz, contact_conductance=0.0)
            shape = np.cos(np.pi * model.x_grid) * np.cos(np.pi * model.z_grid / 0.7)
            out = model.solve(
                [0.0, 0.15], initial_temperature=300.0 + 3.0 * shape,
                max_step=0.0005,
            )
            exact = 300.0 + 3.0 * shape * np.exp(-rate * 0.15)
            errors.append(np.sqrt(np.mean((out.temperature[-1] - exact) ** 2)))
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertGreater(errors[1] / errors[2], 3.8)

    def test_crank_nicolson_has_second_order_time_convergence(self):
        config = self._unit_config()
        model = PlanarContactModel(config, nx=20, nz=14, contact_conductance=0.0)
        shape = np.cos(np.pi * model.x_grid) * np.cos(np.pi * model.z_grid / 0.7)
        # Compare with the exact eigenmode of this spatial discretization,
        # so the spatial error cannot hide or imitate time convergence.
        rate = 4.0 * config.diffusivity * (
            np.sin(np.pi / (2.0 * model.nx))**2 / model.dx**2
            + np.sin(np.pi / (2.0 * model.nz))**2 / model.dz**2
        )
        exact = 300.0 + shape * np.exp(-rate * 0.12)
        errors = []
        for dt in (0.03, 0.015, 0.0075):
            out = model.solve([0.0, 0.12], initial_temperature=300.0 + shape, max_step=dt)
            errors.append(np.sqrt(np.mean((out.temperature[-1] - exact)**2)))
        self.assertGreater(errors[0] / errors[1], 3.9)
        self.assertGreater(errors[1] / errors[2], 3.9)

    def test_manufactured_two_dimensional_robin_solution_converges(self):
        config = self._unit_config()
        h = 2.5
        heat_capacity = config.density * config.specific_heat
        k = config.thermal_conductivity

        def exact(x, z, t):
            a, b = 0.5 + 0.3 * np.sin(2 * t), 0.4 * np.cos(3 * t)
            return 300.0 + a * (1 + h * z / k) + b * np.cos(np.pi * x) * (
                np.sin(np.pi * z / 0.7)**2
            )

        def source(x, z, t):
            shape = np.cos(np.pi * x) * np.sin(np.pi * z / 0.7)**2
            rate = 0.6 * np.cos(2 * t) * (1 + h * z / k) - 1.2 * np.sin(3 * t) * shape
            laplacian = 0.4 * np.cos(3 * t) * np.cos(np.pi * x) * (
                -np.pi**2 * np.sin(np.pi * z / 0.7)**2
                + 2 * (np.pi / 0.7)**2 * np.cos(2 * np.pi * z / 0.7)
            )
            return heat_capacity * rate - k * laplacian

        errors = []
        for nx, nz in ((10, 7), (20, 14), (40, 28)):
            model = PlanarContactModel(config, nx=nx, nz=nz, contact_conductance=h)
            out = model.solve(
                [0.0, 0.1], initial_temperature=lambda x, z: exact(x, z, 0.0),
                te_temperature=300.0,
                top_heat_flux=lambda t: h * (0.5 + 0.3 * np.sin(2 * t)),
                volumetric_source=source, max_step=0.001,
            )
            errors.append(np.sqrt(np.mean((
                out.temperature[-1] - exact(model.x_grid, model.z_grid, 0.1)
            )**2)))
            self.assertLess(abs(out.energy_balance_residual).max(), 1e-9)
        self.assertGreater(errors[0] / errors[1], 3.5)
        self.assertGreater(errors[1] / errors[2], 3.5)

    @staticmethod
    def _canonical_solution(nx, nz, step, h=1000.0):
        return PlanarContactModel(nx=nx, nz=nz, contact_conductance=h).solve(
            np.arange(0.0, 121.0, 10.0), initial_temperature=300.0,
            te_temperature=lambda t: 300.0 + 5.0 * np.expm1(-t / 10.0),
            top_heat_flux=lambda t: -500.0 * np.expm1(-t / 10.0), max_step=step,
        )

    def test_partial_contact_refinement_and_lateral_spreading_are_resolved(self):
        coarse = self._canonical_solution(40, 10, 0.125)
        medium = self._canonical_solution(80, 20, 0.125)
        fine = self._canonical_solution(160, 40, 0.125)
        times = coarse.times[:, None]
        sensors = np.array([0.004, 0.020, 0.036])[None, :]
        coarse_error = abs(coarse.sample(sensors, 0.003, times) - fine.sample(sensors, 0.003, times))
        medium_error = abs(medium.sample(sensors, 0.003, times) - fine.sample(sensors, 0.003, times))
        self.assertLess(medium_error.max(), 0.002)
        self.assertGreater(coarse_error.max() / medium_error.max(), 3.0)
        x, z = np.meshgrid((np.arange(20) + 0.5) * 0.04 / 20,
                           (np.arange(10) + 0.5) * 0.004 / 10)
        field_error = medium.sample(x, z, 120.0) - fine.sample(x, z, 120.0)
        self.assertLess(np.sqrt(np.mean(field_error**2)), 0.005)
        self.assertLess(abs(field_error).max(), 0.01)
        final_sensors = fine.sample(sensors[0], 0.003, 120.0)
        self.assertGreater(np.ptp(final_sensors), 1.0)

    def test_contact_conductance_changes_observations_in_identifiable_direction(self):
        times = np.arange(10.0, 121.0, 10.0)[:, None]
        x = np.array([0.004, 0.020, 0.036])[None, :]
        predictions = [
            self._canonical_solution(40, 10, 0.25, h).sample(x, 0.003, times)
            for h in (600.0, 1000.0, 1600.0)
        ]
        self.assertTrue((predictions[0] > predictions[1]).all())
        self.assertTrue((predictions[1] > predictions[2]).all())
        self.assertGreater(np.sqrt(np.mean((predictions[0] - predictions[1])**2)), 0.1)
        self.assertGreater(np.sqrt(np.mean((predictions[2] - predictions[1])**2)), 0.1)

    def test_input_validation_and_sampling_domain(self):
        for changes in ({"density": 0.0}, {"contact_end": 0.05}, {"depth": float("nan")}):
            with self.assertRaises(ValueError):
                replace(PlanarContactConfig(), **changes)
        for kwargs in ({"nx": 0}, {"nz": 1.5}, {"contact_conductance": -1.0}):
            with self.assertRaises(ValueError):
                PlanarContactModel(**kwargs)
        model = PlanarContactModel(nx=2, nz=1)
        for times in ([1.0], [0.0, 0.0], [0.0, float("nan")]):
            with self.assertRaises(ValueError):
                model.solve(times)
        with self.assertRaises(ValueError):
            model.solve([0.0, 1.0], theta=0.49)
        out = model.solve([0.0, 1.0])
        self.assertEqual(float(out.sample(0.02, 0.002, 0.0)), 300.0)
        with self.assertRaises(ValueError):
            out.sample(-0.01, 0.002, 0.0)


if __name__ == "__main__":
    unittest.main()
