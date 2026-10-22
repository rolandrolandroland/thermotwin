from dataclasses import replace,asdict
import unittest
import numpy as np
import torch
from thermotwin.physics.planar_contact import PlanarContactConfig
from thermotwin.pinn.planar_contact_inverse import PlanarObservations
from thermotwin.pinn.planar_contact_revision import (
    PlanarRevisionConfig,PlanarRevisionPINN,revision_points,revision_residuals,
    train_planar_revision,gate_planar_revision,
)


class PlanarRevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(73)
        self.config=PlanarRevisionConfig(width=6,layers=2,adam_epochs=2,lbfgs_iterations=0,
                                         interior_points=16,boundary_points=8,interface_points=8)

    def test_arbitrary_random_fields_obey_all_lifted_outer_flux_conditions(self):
        for physics in (PlanarContactConfig(),replace(PlanarContactConfig(),contact_start=0.,contact_end=.04)):
            with self.subTest(contact=physics.contact_start):
                model=PlanarRevisionPINN(physics,self.config,known_h=800.)
                points=revision_points(model,12,independent=True)
                residuals=revision_residuals(model,points)
                for family in ("left","right","top","bottom_contact","bottom_insulated"):
                    if family in residuals:
                        self.assertLess(float(abs(residuals[family]).max().detach()),1e-10,family)

    def test_hard_initial_temperature_and_free_top_trace(self):
        model=PlanarRevisionPINN(config=self.config,known_h=800.)
        points=torch.tensor([[.004,.001,0.],[.020,.003,0.],[.036,.004,0.]],dtype=torch.float64)
        np.testing.assert_array_equal(model(points).detach().numpy(),[300.,300.,300.])
        q=torch.tensor([[.4,1.,.5]],dtype=torch.float64,requires_grad=True)
        value=model.dimensionless_temperature(q,region=1)
        grad=torch.autograd.grad(value,tuple(model.networks[1].parameters()),allow_unused=True)
        # The B output has a nonzero top-value effect: it was not removed by
        # a bubble vanishing at both surfaces.
        self.assertNotEqual(float(grad[-1][1]),0.)

    def test_interface_temperature_and_flux_are_explicit_independent_families(self):
        model=PlanarRevisionPINN(config=self.config,known_h=800.)
        values=revision_residuals(model,revision_points(model,123))
        self.assertIn("interface_0_temperature",values)
        self.assertIn("interface_1_flux",values)
        self.assertGreater(float(abs(values["interface_0_temperature"]).max().detach()),.05)

    def test_pde_units_match_manufactured_dimensionless_polynomial(self):
        model=PlanarRevisionPINN(config=self.config,known_h=800.)
        model.dimensionless_temperature=lambda q: .2*q[:,0:1]**2+.3*q[:,1:2]**2-.1*q[:,2:3]
        actual=revision_residuals(model,{"interior":np.array([[.4,.5,.6]])})["interior"]
        p=model.physics
        expected=-.1-self.config.duration*p.diffusivity*(.4/p.width**2+.6/p.thickness**2)
        self.assertAlmostEqual(float(actual.detach()),expected,places=11)

    def test_inverse_rejects_truth_and_supports_nominal_contact_updates(self):
        with self.assertRaises(ValueError):PlanarRevisionPINN(config=self.config,mode="inverse",known_h=800.)
        model=PlanarRevisionPINN(config=self.config,mode="inverse")
        self.assertAlmostEqual(float(model.contact.detach()),self.config.initial_h)
        model.set_contact(1100.)
        self.assertAlmostEqual(float(model.contact.detach()),1100.)
        with self.assertRaises(ValueError):model.set_contact(400.)

    def test_physics_only_warmstart_freezes_h_and_refreshes_config_without_seed_alias(self):
        observations=PlanarObservations(np.array([[.004,.003,10.],[.020,.003,20.]]),np.array([299.9,299.8]))
        original=PlanarRevisionPINN(config=self.config,mode="inverse")
        pure=replace(self.config,observation_weight=0.,refresh_adam_every=1)
        model,report=train_planar_revision(observations,config=pure,mode="inverse",initial_model=original,
                                           sampling_seed=2,independent_seed=3,freeze_contact=True)
        self.assertIs(model,original)
        self.assertEqual(model.config.observation_weight,0.)
        self.assertAlmostEqual(report["contact_estimate"],pure.initial_h)
        seeds=report["derived_training_seeds"]
        self.assertNotIn(3,seeds["adam"]+[seeds["lbfgs"]])
        self.assertEqual(len(set(seeds["adam"]+[seeds["lbfgs"]])),3)
        incompatible=replace(pure,width=7)
        with self.assertRaisesRegex(ValueError,"mismatch"):
            train_planar_revision(observations,config=incompatible,mode="inverse",initial_model=model)

    def test_missing_residual_family_cannot_pass_gate(self):
        model=PlanarRevisionPINN(config=self.config,known_h=800.)
        values=revision_residuals(model,revision_points(model,1))
        report={"physics":asdict(model.physics),"config":asdict(self.config),"normalized_observation_mse":1.,
                "independent_residuals":{"normalized_rms":{k:0. for k in values}}}
        self.assertTrue(gate_planar_revision(report,{"rmse_K":.01,"maximum_error_K":.02})["all"])
        del report["independent_residuals"]["normalized_rms"]["bottom_insulated"]
        self.assertFalse(gate_planar_revision(report,{"rmse_K":.01,"maximum_error_K":.02})["all"])

    def test_conditioned_lifting_bc_and_omitted_identity_loss_gradients(self):
        config=replace(self.config,temporal_envelope="ramp",time_feature="log_diffusion",skip_hard_bc_training=True)
        for mode in ("known_contact","inverse"):
            model=PlanarRevisionPINN(config=config,mode=mode,known_h=800. if mode=="known_contact" else None)
            points=revision_points(model,918)
            residuals=revision_residuals(model,points)
            full=sum(torch.mean(v*v) for v in residuals.values())
            selected=sum(torch.mean(v*v) for k,v in residuals.items() if k=="interior" or k.startswith("interface_"))
            self.assertAlmostEqual(float((full-selected).detach()),0.,places=10)
            parameters=[p for p in model.parameters() if p.requires_grad]
            ga=torch.autograd.grad(full,parameters,retain_graph=True,allow_unused=True)
            gb=torch.autograd.grad(selected,parameters,allow_unused=True)
            for a,b in zip(ga,gb):
                if a is not None:torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-10)
            for name in ("top","bottom_contact","bottom_insulated","left","right"):
                if name in residuals:self.assertLess(float(abs(residuals[name]).max().detach()),1e-10)


if __name__=="__main__":unittest.main()
