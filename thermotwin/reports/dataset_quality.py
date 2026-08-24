"""Human-readable quality report for the frozen reference datasets."""

from ..inference.contact_resistance import (
    reference_contact_resistance_dataset_split,
)
from ..observations.quality import (
    format_dataset_quality_report,
    summarize_dataset_collection,
)


def reference_dataset_quality_report() -> str:
    """Build the frozen whole-regime quality report used by the documentation."""

    split = reference_contact_resistance_dataset_split()
    datasets = tuple(
        item.observations
        for group in (split.train, split.validation, split.test)
        for item in group
    )
    return format_dataset_quality_report(
        summarize_dataset_collection(datasets)
    )


def main() -> None:
    """Print the frozen train/validation/test dataset-quality report."""

    print(reference_dataset_quality_report())


if __name__ == "__main__":
    main()
