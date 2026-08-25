"""Opt-in literature material records kept outside the StarryData catalog.

These records support targeted, source-specific studies without changing the
fixed ``P_TYPE_SAMPLES`` and ``N_TYPE_SAMPLES`` tuples that define the
reproducible baseline co-design campaign.
"""

from dataclasses import dataclass
import math

from .materials import MaterialSample


AG2SE_2026_DOI = "10.1039/D6MH00220J"
AG2SE_2026_URL = "https://doi.org/10.1039/D6MH00220J"


@dataclass(frozen=True)
class LiteratureMaterialRecord:
    """One complete same-sample property triplet with explicit provenance."""

    key: str
    material: MaterialSample
    source_doi: str
    source_url: str
    process_description: str

    def __post_init__(self) -> None:
        if not self.key or not self.source_doi or not self.source_url:
            raise ValueError("literature material provenance must be nonempty")
        if not self.process_description:
            raise ValueError("literature material process description is required")


# Internal ID 2026001 is a ThermoTwin identifier, not a StarryData sample ID.
# The optimized values are reported together for the 9% excess-selenium sample
# at room temperature. They must not be mixed with the paper's baseline-sample
# Seebeck coefficient or 0.75 W/(m K) baseline thermal conductivity.
AG2SE_2026_OPTIMIZED = LiteratureMaterialRecord(
    key="ag2se_2026_optimized",
    material=MaterialSample(
        2026001,
        "n",
        "Ag2Se with 9% excess Se",
        "optimized ink-processed Ag2Se",
        -153.3e-6,
        117_400.0,
        0.85,
        300.0,
    ),
    source_doi=AG2SE_2026_DOI,
    source_url=AG2SE_2026_URL,
    process_description=(
        "9% excess selenium; synthesized at 350 C for 90 min and sintered "
        "at 375 C for 60 min"
    ),
)


@dataclass(frozen=True)
class PublishedUnicoupleElectricalCase:
    """Electrical quantities reported for the paper's 1.5 mm unicouple."""

    leg_length: float = 1.5e-3
    leg_area: float = 2.25e-6
    p_seebeck_coefficient: float = 210.0e-6
    p_power_factor: float = 2.1e-3
    n_electrical_conductivity: float = 117_400.0
    resistance_per_contact: float = 7.4e-3
    reported_total_device_resistance: float = 50.0e-3
    interface_count: int = 4

    def __post_init__(self) -> None:
        positive = (
            self.leg_length,
            self.leg_area,
            self.p_seebeck_coefficient,
            self.p_power_factor,
            self.n_electrical_conductivity,
            self.resistance_per_contact,
            self.reported_total_device_resistance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("published unicouple quantities must be positive")
        if self.interface_count <= 0:
            raise ValueError("interface count must be positive")

    @property
    def p_electrical_conductivity(self) -> float:
        """Infer p-leg conductivity from the reported S and power factor."""

        return self.p_power_factor / self.p_seebeck_coefficient**2

    @property
    def p_electrical_resistivity(self) -> float:
        return 1.0 / self.p_electrical_conductivity

    @property
    def n_electrical_resistivity(self) -> float:
        return 1.0 / self.n_electrical_conductivity

    @property
    def inferred_specific_contact_resistivity(self) -> float:
        """Return Rc*A assuming the complete leg face is effective area."""

        return self.resistance_per_contact * self.leg_area

    @property
    def half_contact_share_resistivity(self) -> float:
        """Return rho_c at which four interfaces equal both bulk legs."""

        return (
            self.leg_length
            * (self.p_electrical_resistivity + self.n_electrical_resistivity)
            / self.interface_count
        )

    @property
    def reported_contact_share(self) -> float:
        """Return four reported contacts divided by total device resistance."""

        return (
            self.interface_count
            * self.resistance_per_contact
            / self.reported_total_device_resistance
        )

    def modeled_contact_share(self, specific_contact_resistivity: float) -> float:
        """Return the ideal full-area contact share for a supplied rho_c."""

        if (
            not math.isfinite(specific_contact_resistivity)
            or specific_contact_resistivity < 0.0
        ):
            raise ValueError("specific contact resistivity must be nonnegative")
        contact = (
            self.interface_count
            * specific_contact_resistivity
            / self.leg_area
        )
        bulk = (
            self.leg_length
            / self.leg_area
            * (self.p_electrical_resistivity + self.n_electrical_resistivity)
        )
        return contact / (contact + bulk)


PUBLISHED_AG2SE_UNICOUPLE = PublishedUnicoupleElectricalCase()


__all__ = [
    "AG2SE_2026_DOI",
    "AG2SE_2026_OPTIMIZED",
    "AG2SE_2026_URL",
    "LiteratureMaterialRecord",
    "PUBLISHED_AG2SE_UNICOUPLE",
    "PublishedUnicoupleElectricalCase",
]
