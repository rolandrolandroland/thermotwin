"""Small, provenance-rich room-temperature thermoelectric material catalog.

The catalog is a deliberately small extract from the fixed StarryData
thermoelectric snapshot published on Figshare in 2019.  Every property triplet
comes from one sample row at 300 K.  The values are experimental literature
data digitized and interpolated by StarryData; they are not ThermoTwin
measurements and they are not a process-to-property model.
"""

from dataclasses import dataclass
import math
from typing import Tuple


STARRYDATA_SNAPSHOT_DOI = "10.6084/m9.figshare.11340935.v1"
STARRYDATA_SNAPSHOT_URL = (
    "https://figshare.com/articles/dataset/"
    "Starrydata_thermoelectric_data_snapshot_interpolated_data_/11340935"
)
STARRYDATA_SNAPSHOT_FILENAME = "Starrydata_interpolated_20190816.csv"
STARRYDATA_SNAPSHOT_MD5 = "5ae1d38f76fd872d40bff37c2bec29f6"
STARRYDATA_SNAPSHOT_LICENSE = "CC BY 4.0"
STARRYDATA_SOURCE_TEMPERATURE = 300.0


@dataclass(frozen=True)
class MaterialSample:
    """One same-row material-property record used for a TEC leg.

    Seebeck coefficient is in V/K, electrical conductivity is in S/m,
    thermal conductivity is in W/(m K), and temperature is in K.
    """

    sample_id: int
    carrier_type: str
    composition: str
    sample_name: str
    seebeck_coefficient: float
    electrical_conductivity: float
    thermal_conductivity: float
    temperature: float = STARRYDATA_SOURCE_TEMPERATURE

    def __post_init__(self) -> None:
        if self.sample_id <= 0:
            raise ValueError("sample ID must be positive")
        if self.carrier_type not in {"p", "n"}:
            raise ValueError("carrier type must be 'p' or 'n'")
        if not self.composition or not self.sample_name:
            raise ValueError("composition and sample name must be nonempty")
        values = (
            self.seebeck_coefficient,
            self.electrical_conductivity,
            self.thermal_conductivity,
            self.temperature,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("material properties must be finite")
        if self.electrical_conductivity <= 0.0:
            raise ValueError("electrical conductivity must be positive")
        if self.thermal_conductivity <= 0.0 or self.temperature <= 0.0:
            raise ValueError("thermal conductivity and temperature must be positive")
        if self.carrier_type == "p" and self.seebeck_coefficient <= 0.0:
            raise ValueError("p-type sample must have positive Seebeck coefficient")
        if self.carrier_type == "n" and self.seebeck_coefficient >= 0.0:
            raise ValueError("n-type sample must have negative Seebeck coefficient")

    @property
    def electrical_resistivity(self) -> float:
        """Return resistivity in ohm metres."""

        return 1.0 / self.electrical_conductivity

    @property
    def power_factor(self) -> float:
        """Return the material power factor S**2 sigma in W/(m K**2)."""

        return self.seebeck_coefficient**2 * self.electrical_conductivity

    @property
    def calculated_zt(self) -> float:
        """Return S**2 sigma T / k using this same-row property triplet."""

        return self.power_factor * self.temperature / self.thermal_conductivity


# Selection rule used for this compact catalog:
# * exact 300 K interpolated row;
# * complete canonical S, sigma, and k columns on that same row;
# * Bi/Te-family composition with no fabricated cross-row property mixing;
# * plausible cooling-material envelope of 50--400 uV/K, 10--500 kS/m,
#   and 0.3--4 W/(m K);
# * six records of each carrier sign retained to span property trade-offs.
P_TYPE_SAMPLES: Tuple[MaterialSample, ...] = (
    MaterialSample(
        9107,
        "p",
        "BixSb2-xTe3 (Bi:Sb:Te = 8:44:48)",
        "Nano-crystalline bulk sample",
        0.00019858654368585285,
        117417.29594263893,
        1.002887049666645,
    ),
    MaterialSample(
        10561,
        "p",
        "p-type (Bi,Sb)2Te3",
        "p-type (Bi,Sb)2Te3",
        0.00022488394089584075,
        86333.74622797583,
        1.2213597628870971,
    ),
    MaterialSample(
        10879,
        "p",
        "Bi0.5 Sb1.5 Te3",
        "BST",
        0.00023514918949434484,
        47912.685665598794,
        1.0392242152365523,
    ),
    MaterialSample(
        7986,
        "p",
        "Sb1.52Bi0.48Te3",
        "MS aligned",
        0.00018040310834491912,
        80832.39281302493,
        1.1641518295172604,
    ),
    MaterialSample(
        5550,
        "p",
        "Bi0.5Sb1.5Te3",
        "NBH1",
        0.00015849781101417168,
        69509.18386353916,
        1.0256044332241319,
    ),
    MaterialSample(
        5553,
        "p",
        "Bi0.5Sb1.5Te3",
        "NBS2",
        0.00017398727824207552,
        15890.232160466314,
        0.7261599336714091,
    ),
)


N_TYPE_SAMPLES: Tuple[MaterialSample, ...] = (
    MaterialSample(
        10562,
        "n",
        "Bi2(Te,Se)3",
        "Bi2(Te,Se)3",
        -0.00020138807984181417,
        74633.51137657845,
        0.8070812713239163,
    ),
    MaterialSample(
        5606,
        "n",
        "Bi2Te2.7Se0.3",
        "parallel",
        -0.00019410356892790722,
        63457.16345179891,
        0.8348864944146988,
    ),
    MaterialSample(
        14771,
        "n",
        "Bi2Te3",
        "perpendicular (press direction)",
        -0.00016884763194223296,
        71662.93979743088,
        0.9220517201671229,
    ),
    MaterialSample(
        5792,
        "n",
        "Bi2Te2.25Se0.75",
        "c",
        -0.00014412111654406545,
        88641.94691156081,
        0.9029556552620389,
    ),
    MaterialSample(
        16848,
        "n",
        "Bi2Te2.7Se0.3",
        "SPS-250C",
        -0.00026298502215357165,
        16700.12871131228,
        0.6019345004857167,
    ),
    MaterialSample(
        16850,
        "n",
        "Bi2Te2.7Se0.3",
        "SPS-350C",
        -0.0001341984413496255,
        30784.11621377705,
        1.1005980039421872,
    ),
)


def material_sample(sample_id: int) -> MaterialSample:
    """Return one curated sample by its StarryData sample ID."""

    matches = tuple(
        sample
        for sample in P_TYPE_SAMPLES + N_TYPE_SAMPLES
        if sample.sample_id == sample_id
    )
    if len(matches) != 1:
        raise KeyError(f"unknown curated StarryData sample ID: {sample_id}")
    return matches[0]
