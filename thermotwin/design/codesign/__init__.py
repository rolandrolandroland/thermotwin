"""Material, geometry, assembly, and application co-design."""

from .campaign import format_codesign_campaign_report, run_codesign_campaign
from .evaluation import (
    averaged_contact_steady_state_for_parameters,
    evaluate_design_current,
    module_electrical_resistance_components,
    module_thermoelectric_parameters,
    optimize_design_current,
    prototype_cost_index,
)
from .models import (
    APPLICATION_SPECIFICATIONS,
    ApplicationSpecification,
    BayesianOptimizationResult,
    CodesignCampaignConfig,
    CodesignCampaignResult,
    DesignOperatingPoint,
    InitialDesignSummary,
    ModuleAssemblyAssumptions,
    ModuleElectricalResistanceComponents,
    ModuleGeometry,
    PropertyMultipliers,
    PrototypeDesign,
    RobustnessResult,
)
from .optimization import (
    expected_improvement,
    gaussian_process_predict,
    run_bayesian_optimization,
)
from .robustness import run_robustness_study
from .sampling import design_features, generate_space_filling_designs, latin_hypercube

__all__ = [
    "APPLICATION_SPECIFICATIONS",
    "ApplicationSpecification",
    "BayesianOptimizationResult",
    "CodesignCampaignConfig",
    "CodesignCampaignResult",
    "DesignOperatingPoint",
    "InitialDesignSummary",
    "ModuleAssemblyAssumptions",
    "ModuleElectricalResistanceComponents",
    "ModuleGeometry",
    "PropertyMultipliers",
    "PrototypeDesign",
    "RobustnessResult",
    "averaged_contact_steady_state_for_parameters",
    "design_features",
    "evaluate_design_current",
    "expected_improvement",
    "format_codesign_campaign_report",
    "gaussian_process_predict",
    "generate_space_filling_designs",
    "latin_hypercube",
    "module_electrical_resistance_components",
    "module_thermoelectric_parameters",
    "optimize_design_current",
    "prototype_cost_index",
    "run_bayesian_optimization",
    "run_codesign_campaign",
    "run_robustness_study",
]
