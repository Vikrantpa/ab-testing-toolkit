"""abtest — a small, rigorous A/B-testing toolkit.

Public API:

    Power analysis
        sample_size_means, power_means
        sample_size_proportions, power_proportions

    Fixed-horizon tests
        two_sample_test, two_proportion_test, TestResult

    Variance reduction
        cuped_adjust, run_cuped_test, CUPEDResult

    Sequential / always-valid inference
        MSPRT, SequentialResult, always_valid_pvalue, confidence_sequence
"""

from .power import (
    sample_size_means,
    power_means,
    sample_size_proportions,
    power_proportions,
    PowerCurve,
)
from .frequentist import two_sample_test, two_proportion_test, TestResult
from .cuped import cuped_adjust, run_cuped_test, CUPEDResult
from .sequential import (
    MSPRT,
    SequentialResult,
    always_valid_pvalue,
    confidence_sequence,
)

__version__ = "0.1.0"

__all__ = [
    "sample_size_means",
    "power_means",
    "sample_size_proportions",
    "power_proportions",
    "PowerCurve",
    "two_sample_test",
    "two_proportion_test",
    "TestResult",
    "cuped_adjust",
    "run_cuped_test",
    "CUPEDResult",
    "MSPRT",
    "SequentialResult",
    "always_valid_pvalue",
    "confidence_sequence",
]
