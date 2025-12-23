"""Result dataclasses for validation and robustness analysis.

These classes store the outputs of validation analyses including:
- Out-of-sample validation
- Model comparison
- Effect size calculations
- MI estimator sensitivity
- Temporal stability analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OutOfSampleResult:
    """Result of out-of-sample validation for MI rankings.

    Compares MI scores computed on training vs test splits to assess
    whether feature rankings are stable out-of-sample.

    Attributes:
        features: List of feature names tested.
        train_mi: MI scores on training set (feature -> MI).
        test_mi: MI scores on test set (feature -> MI).
        train_ranking: Rank of each feature on training set.
        test_ranking: Rank of each feature on test set.
        spearman_correlation: Spearman correlation between rankings.
        n_train: Number of training samples.
        n_test: Number of test samples.
        test_ratio: Proportion of data in test set.
        temporal_split: Whether temporal (non-shuffled) split was used.
    """

    features: list[str]
    train_mi: dict[str, float]
    test_mi: dict[str, float]
    train_ranking: dict[str, int]
    test_ranking: dict[str, int]
    spearman_correlation: float
    n_train: int
    n_test: int
    test_ratio: float
    temporal_split: bool

    @property
    def is_stable(self) -> bool:
        """Check if rankings are highly correlated (rho > 0.8)."""
        return self.spearman_correlation > 0.8

    def top_k_overlap(self, k: int = 10) -> float:
        """Calculate overlap in top-k features between train and test.

        Args:
            k: Number of top features to compare.

        Returns:
            Overlap proportion (0-1).
        """
        train_top = set(
            sorted(self.train_ranking.keys(), key=lambda f: self.train_ranking[f])[:k]
        )
        test_top = set(
            sorted(self.test_ranking.keys(), key=lambda f: self.test_ranking[f])[:k]
        )
        return len(train_top & test_top) / k

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "features": self.features,
            "train_mi": self.train_mi,
            "test_mi": self.test_mi,
            "train_ranking": self.train_ranking,
            "test_ranking": self.test_ranking,
            "spearman_correlation": self.spearman_correlation,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_ratio": self.test_ratio,
            "temporal_split": self.temporal_split,
            "is_stable": self.is_stable,
            "top_10_overlap": self.top_k_overlap(10),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutOfSampleResult:
        """Create from dictionary."""
        return cls(
            features=data["features"],
            train_mi=data["train_mi"],
            test_mi=data["test_mi"],
            train_ranking=data["train_ranking"],
            test_ranking=data["test_ranking"],
            spearman_correlation=data["spearman_correlation"],
            n_train=data["n_train"],
            n_test=data["n_test"],
            test_ratio=data["test_ratio"],
            temporal_split=data["temporal_split"],
        )


@dataclass
class ModelComparisonResult:
    """Result of model comparison between close and midrange targets.

    Compares LightGBM models trained on the same features but different
    targets to validate that MI improvements translate to model performance.

    Attributes:
        close_mae: Mean Absolute Error for close target.
        close_rmse: Root Mean Squared Error for close target.
        close_directional_accuracy: Directional accuracy for close target.
        close_sharpe: Sharpe ratio of strategy for close target.
        midrange_mae: Mean Absolute Error for midrange target.
        midrange_rmse: Root Mean Squared Error for midrange target.
        midrange_directional_accuracy: Directional accuracy for midrange target.
        midrange_sharpe: Sharpe ratio of strategy for midrange target.
        n_samples: Number of samples in test set.
        n_features: Number of features used.
        features: List of feature names used.
        cv_folds: Number of cross-validation folds (0 if not CV).
    """

    close_mae: float
    close_rmse: float
    close_directional_accuracy: float
    close_sharpe: float
    midrange_mae: float
    midrange_rmse: float
    midrange_directional_accuracy: float
    midrange_sharpe: float
    n_samples: int
    n_features: int
    features: list[str]
    cv_folds: int = 0

    @property
    def mae_improvement(self) -> float:
        """MAE improvement: (close_mae - midrange_mae) / close_mae."""
        if self.close_mae == 0:
            return 0.0
        return (self.close_mae - self.midrange_mae) / self.close_mae

    @property
    def rmse_improvement(self) -> float:
        """RMSE improvement: (close_rmse - midrange_rmse) / close_rmse."""
        if self.close_rmse == 0:
            return 0.0
        return (self.close_rmse - self.midrange_rmse) / self.close_rmse

    @property
    def directional_improvement(self) -> float:
        """Directional accuracy improvement."""
        return self.midrange_directional_accuracy - self.close_directional_accuracy

    @property
    def sharpe_improvement(self) -> float:
        """Sharpe ratio improvement."""
        return self.midrange_sharpe - self.close_sharpe

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "close_mae": self.close_mae,
            "close_rmse": self.close_rmse,
            "close_directional_accuracy": self.close_directional_accuracy,
            "close_sharpe": self.close_sharpe,
            "midrange_mae": self.midrange_mae,
            "midrange_rmse": self.midrange_rmse,
            "midrange_directional_accuracy": self.midrange_directional_accuracy,
            "midrange_sharpe": self.midrange_sharpe,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "features": self.features,
            "cv_folds": self.cv_folds,
            "mae_improvement": self.mae_improvement,
            "rmse_improvement": self.rmse_improvement,
            "directional_improvement": self.directional_improvement,
            "sharpe_improvement": self.sharpe_improvement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelComparisonResult:
        """Create from dictionary."""
        return cls(
            close_mae=data["close_mae"],
            close_rmse=data["close_rmse"],
            close_directional_accuracy=data["close_directional_accuracy"],
            close_sharpe=data["close_sharpe"],
            midrange_mae=data["midrange_mae"],
            midrange_rmse=data["midrange_rmse"],
            midrange_directional_accuracy=data["midrange_directional_accuracy"],
            midrange_sharpe=data["midrange_sharpe"],
            n_samples=data["n_samples"],
            n_features=data["n_features"],
            features=data["features"],
            cv_folds=data.get("cv_folds", 0),
        )


@dataclass
class EffectSizeResult:
    """Result of effect size calculation (Cohen's d).

    Quantifies the magnitude of the difference between two groups
    independent of sample size.

    Attributes:
        cohens_d: Effect size value (standardized mean difference).
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        n_group1: Sample size of first group.
        n_group2: Sample size of second group.
        pooled_std: Pooled standard deviation used.
        mean_diff: Raw mean difference.
        interpretation: Effect size interpretation (small/medium/large).
        n_bootstrap: Number of bootstrap iterations for CI.
        confidence_level: Confidence level for CI.
    """

    cohens_d: float
    ci_lower: float
    ci_upper: float
    n_group1: int
    n_group2: int
    pooled_std: float
    mean_diff: float
    interpretation: str
    n_bootstrap: int = 0
    confidence_level: float = 0.95

    @staticmethod
    def interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            d: Effect size value.

        Returns:
            Interpretation string.
        """
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "cohens_d": self.cohens_d,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_group1": self.n_group1,
            "n_group2": self.n_group2,
            "pooled_std": self.pooled_std,
            "mean_diff": self.mean_diff,
            "interpretation": self.interpretation,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EffectSizeResult:
        """Create from dictionary."""
        return cls(
            cohens_d=data["cohens_d"],
            ci_lower=data["ci_lower"],
            ci_upper=data["ci_upper"],
            n_group1=data["n_group1"],
            n_group2=data["n_group2"],
            pooled_std=data["pooled_std"],
            mean_diff=data["mean_diff"],
            interpretation=data["interpretation"],
            n_bootstrap=data.get("n_bootstrap", 0),
            confidence_level=data.get("confidence_level", 0.95),
        )


@dataclass
class SensitivityResult:
    """Result of MI estimator sensitivity analysis.

    Tests robustness of MI estimates across different k-NN values
    (or other parameter variations).

    Attributes:
        feature: Feature name.
        parameter_name: Name of parameter varied (e.g., "k_neighbors").
        parameter_values: List of parameter values tested.
        mi_values: MI value at each parameter setting.
        rank_at_each: Feature rank at each parameter setting.
        mean_mi: Mean MI across parameter settings.
        std_mi: Standard deviation of MI across settings.
        cv_mi: Coefficient of variation (std/mean).
        rank_correlation: Spearman correlation of rankings across settings.
    """

    feature: str
    parameter_name: str
    parameter_values: list
    mi_values: list[float]
    rank_at_each: list[int]
    mean_mi: float
    std_mi: float
    cv_mi: float
    rank_correlation: float

    @property
    def is_stable(self) -> bool:
        """Check if MI is stable across parameters (CV < 0.1)."""
        return self.cv_mi < 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature": self.feature,
            "parameter_name": self.parameter_name,
            "parameter_values": self.parameter_values,
            "mi_values": self.mi_values,
            "rank_at_each": self.rank_at_each,
            "mean_mi": self.mean_mi,
            "std_mi": self.std_mi,
            "cv_mi": self.cv_mi,
            "rank_correlation": self.rank_correlation,
            "is_stable": self.is_stable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SensitivityResult:
        """Create from dictionary."""
        return cls(
            feature=data["feature"],
            parameter_name=data["parameter_name"],
            parameter_values=data["parameter_values"],
            mi_values=data["mi_values"],
            rank_at_each=data["rank_at_each"],
            mean_mi=data["mean_mi"],
            std_mi=data["std_mi"],
            cv_mi=data["cv_mi"],
            rank_correlation=data["rank_correlation"],
        )


@dataclass
class TemporalStabilityResult:
    """Result of rolling window temporal stability analysis.

    Assesses how stable MI rankings are over time by computing
    MI in rolling windows.

    Attributes:
        features: List of feature names analyzed.
        window_size: Size of rolling window in rows.
        step_size: Step size between windows.
        n_windows: Number of windows computed.
        window_starts: Start indices of each window.
        mi_by_window: MI values per window (window_idx -> feature -> MI).
        rank_by_window: Rankings per window (window_idx -> feature -> rank).
        adjacent_correlations: Spearman correlations between adjacent windows.
        mean_correlation: Mean of adjacent window correlations.
        min_correlation: Minimum correlation (worst stability).
        regime_changes: Indices where correlation dropped significantly.
    """

    features: list[str]
    window_size: int
    step_size: int
    n_windows: int
    window_starts: list[int]
    mi_by_window: list[dict[str, float]]
    rank_by_window: list[dict[str, int]]
    adjacent_correlations: list[float]
    mean_correlation: float
    min_correlation: float
    regime_changes: list[int] = field(default_factory=list)

    @property
    def is_stable(self) -> bool:
        """Check if rankings are stable over time (mean rho > 0.8)."""
        return self.mean_correlation > 0.8

    @property
    def n_regime_changes(self) -> int:
        """Number of detected regime changes."""
        return len(self.regime_changes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "features": self.features,
            "window_size": self.window_size,
            "step_size": self.step_size,
            "n_windows": self.n_windows,
            "window_starts": self.window_starts,
            "mi_by_window": self.mi_by_window,
            "rank_by_window": self.rank_by_window,
            "adjacent_correlations": self.adjacent_correlations,
            "mean_correlation": self.mean_correlation,
            "min_correlation": self.min_correlation,
            "regime_changes": self.regime_changes,
            "is_stable": self.is_stable,
            "n_regime_changes": self.n_regime_changes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalStabilityResult:
        """Create from dictionary."""
        return cls(
            features=data["features"],
            window_size=data["window_size"],
            step_size=data["step_size"],
            n_windows=data["n_windows"],
            window_starts=data["window_starts"],
            mi_by_window=data["mi_by_window"],
            rank_by_window=data["rank_by_window"],
            adjacent_correlations=data["adjacent_correlations"],
            mean_correlation=data["mean_correlation"],
            min_correlation=data["min_correlation"],
            regime_changes=data.get("regime_changes", []),
        )
