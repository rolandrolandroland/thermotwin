# ThermoTwin portfolio bullets

These bullets are deliberately concise and preserve the synthetic-validation
boundary. Select two or three according to the role rather than using all of
them in one résumé entry.

- Developed **ThermoTwin**, an open-source, CPU-first thermoelectric digital
  twin connecting contact-aware transient physics, imperfect virtual sensors,
  inverse parameter estimation, COP/control analysis, and constrained
  next-experiment selection; supported by more than 500 automated physics,
  numerical, inference, and reporting tests.

- Designed a controlled five-trial PINN benchmark with identically initialized
  networks and matched sparse observations; physics constraints reduced
  missing-interval error by **87.86%**, hidden-state error by **99.68%**, and
  independently calculated energy-rate imbalance by **99.19%** versus a
  data-only network on declared synthetic truth.

- Built a D-optimal thermoelectric pulse planner and validated it with 20
  complete nonlinear multistart refits; the selected experiment reduced mean
  joint log-parameter error by **81.46%** versus a naive pulse and **11.77%**
  versus a similar-energy grid control.

- Implemented identifiability, uncertainty, sensor-ablation, model-mismatch,
  and withheld-regime checks that distinguish optimization convergence from a
  physically supportable inverse estimate and retain biased or
  non-identifiable results.

- Connected published thermoelectric property records to geometry, electrical
  and thermal interfaces, drive constraints, exchanger performance, and
  application-specific COP; reported null optimization results and a nominal
  design with only a **55.3%** pass rate under declared virtual manufacturing
  variation.

## Short project description

**ThermoTwin — physics-informed thermoelectric digital twin.** Built modular
conventional and neural forward/inverse solvers, realistic virtual sensors,
identifiability analysis, control comparisons, and experiment-selection tools
for thermoelectric heat pumps. All numerical claims are reproducible synthetic
results and are explicitly separated from hardware validation.
