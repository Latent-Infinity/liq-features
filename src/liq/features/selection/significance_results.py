"""Result dataclasses for statistical significance testing.

These classes store the outputs of bootstrap confidence interval estimation,
permutation testing, and paired difference tests for MI scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation for MI.

    Attributes:
        feature: Feature name.
        point_estimate: Original MI score (not resampled).
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        std_error: Standard error from bootstrap distribution.
        n_bootstrap: Number of bootstrap iterations.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        n_samples: Number of valid samples used.
    """

    feature: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    confidence_level: float
    n_samples: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "point_estimate": self.point_estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std_error": self.std_error,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BootstrapResult:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PermutationResult:
    """Result of permutation test for MI significance.

    Tests H0: MI = 0 (no relationship between feature and target).

    Attributes:
        feature: Feature name.
        observed_mi: Observed MI score on original data.
        p_value: Proportion of permuted MI >= observed MI.
        null_mean: Mean of null distribution.
        null_std: Standard deviation of null distribution.
        n_permutations: Number of permutation iterations.
        n_samples: Number of valid samples used.
    """

    feature: str
    observed_mi: float
    p_value: float
    null_mean: float
    null_std: float
    n_permutations: int
    n_samples: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if MI is significantly different from 0."""
        return self.p_value < alpha

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "observed_mi": self.observed_mi,
            "p_value": self.p_value,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "n_permutations": self.n_permutations,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PermutationResult:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PairedDifferenceResult:
    """Result of paired bootstrap difference test.

    Tests H0: MI_midrange <= MI_close (midrange is not better).

    Attributes:
        feature: Feature name.
        close_mi: MI score for close target.
        midrange_mi: MI score for midrange target.
        difference: midrange_mi - close_mi.
        ci_lower: Lower bound of difference CI.
        ci_upper: Upper bound of difference CI.
        p_value: Proportion of bootstrap differences <= 0.
        n_bootstrap: Number of bootstrap iterations.
        n_samples: Number of valid samples used.
    """

    feature: str
    close_mi: float
    midrange_mi: float
    difference: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_bootstrap: int
    n_samples: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if midrange MI is significantly higher than close MI."""
        return self.p_value < alpha

    @property
    def relative_improvement(self) -> float:
        """Calculate relative improvement: (midrange - close) / close."""
        if self.close_mi == 0:
            return float("inf") if self.midrange_mi > 0 else 0.0
        return self.difference / self.close_mi

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "close_mi": self.close_mi,
            "midrange_mi": self.midrange_mi,
            "difference": self.difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "n_bootstrap": self.n_bootstrap,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PairedDifferenceResult:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SignificanceReport:
    """Aggregated significance testing results.

    Attributes:
        timeframe: Timeframe analyzed (e.g., "1h").
        close_bootstrap: Bootstrap results for close target.
        midrange_bootstrap: Bootstrap results for midrange target.
        close_permutation: Permutation test results for close target.
        midrange_permutation: Permutation test results for midrange target.
        paired_results: Paired difference test results.
        n_features: Number of features tested.
        n_bootstrap: Number of bootstrap iterations.
        n_permutations: Number of permutation iterations.
        confidence_level: Confidence level for CIs.
    """

    timeframe: str
    close_bootstrap: list[BootstrapResult] = field(default_factory=list)
    midrange_bootstrap: list[BootstrapResult] = field(default_factory=list)
    close_permutation: list[PermutationResult] = field(default_factory=list)
    midrange_permutation: list[PermutationResult] = field(default_factory=list)
    paired_results: list[PairedDifferenceResult] = field(default_factory=list)
    n_features: int = 0
    n_bootstrap: int = 0
    n_permutations: int = 0
    confidence_level: float = 0.95

    # FDR-corrected results (populated after correction)
    fdr_adjusted_p_values: list[float] = field(default_factory=list)
    fdr_significant: list[bool] = field(default_factory=list)

    @property
    def n_significant_permutation_close(self) -> int:
        """Number of features with significant MI for close target."""
        return sum(1 for r in self.close_permutation if r.is_significant())

    @property
    def n_significant_permutation_midrange(self) -> int:
        """Number of features with significant MI for midrange target."""
        return sum(1 for r in self.midrange_permutation if r.is_significant())

    @property
    def n_significant_paired(self) -> int:
        """Number of features where midrange MI > close MI significantly."""
        return sum(1 for r in self.paired_results if r.is_significant())

    @property
    def n_significant_paired_fdr(self) -> int:
        """Number of features significant after FDR correction."""
        return sum(1 for sig in self.fdr_significant if sig)

    def summary_dict(self) -> dict:
        """Generate summary statistics dictionary."""
        return {
            "timeframe": self.timeframe,
            "n_features": self.n_features,
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
            "confidence_level": self.confidence_level,
            "close_significant": self.n_significant_permutation_close,
            "midrange_significant": self.n_significant_permutation_midrange,
            "paired_significant": self.n_significant_paired,
            "paired_significant_fdr": self.n_significant_paired_fdr,
        }

    def to_dict(self) -> dict:
        """Convert full report to dictionary for JSON serialization."""
        return {
            "summary": self.summary_dict(),
            "close_bootstrap": [r.to_dict() for r in self.close_bootstrap],
            "midrange_bootstrap": [r.to_dict() for r in self.midrange_bootstrap],
            "close_permutation": [r.to_dict() for r in self.close_permutation],
            "midrange_permutation": [r.to_dict() for r in self.midrange_permutation],
            "paired_results": [r.to_dict() for r in self.paired_results],
            "fdr_adjusted_p_values": self.fdr_adjusted_p_values,
            "fdr_significant": self.fdr_significant,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SignificanceReport:
        """Create from dictionary."""
        summary = data.get("summary", {})
        report = cls(
            timeframe=summary.get("timeframe", ""),
            n_features=summary.get("n_features", 0),
            n_bootstrap=summary.get("n_bootstrap", 0),
            n_permutations=summary.get("n_permutations", 0),
            confidence_level=summary.get("confidence_level", 0.95),
        )

        report.close_bootstrap = [
            BootstrapResult.from_dict(r) for r in data.get("close_bootstrap", [])
        ]
        report.midrange_bootstrap = [
            BootstrapResult.from_dict(r) for r in data.get("midrange_bootstrap", [])
        ]
        report.close_permutation = [
            PermutationResult.from_dict(r) for r in data.get("close_permutation", [])
        ]
        report.midrange_permutation = [
            PermutationResult.from_dict(r) for r in data.get("midrange_permutation", [])
        ]
        report.paired_results = [
            PairedDifferenceResult.from_dict(r) for r in data.get("paired_results", [])
        ]
        report.fdr_adjusted_p_values = data.get("fdr_adjusted_p_values", [])
        report.fdr_significant = data.get("fdr_significant", [])

        return report
