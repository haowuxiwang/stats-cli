"""Measurement Uncertainty evaluation following GUM (Guide to the Expression of Uncertainty in Measurement).

Implements ISO/IEC 17025 compliant uncertainty evaluation:
- Type A: statistical evaluation from repeated measurements
- Type B: systematic evaluation from calibration, resolution, environment
- Combined and expanded uncertainty with effective degrees of freedom (Welch-Satterthwaite)
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.output import r  # noqa: F401 (re-exported)


@dataclass
class UncertaintySource:
    """Single source of measurement uncertainty."""

    name: str
    distribution: str  # "normal", "rectangular", "triangular", "u_shaped", "type_a"
    value: float  # standard uncertainty (for type_b) or values (for type_a)
    sensitivity_coefficient: float = 1.0
    degrees_of_freedom: Optional[float] = None

    @property
    def standard_uncertainty(self) -> float:
        """Convert input value to standard uncertainty based on distribution."""
        if self.distribution == "type_a":
            if isinstance(self.value, (list, tuple, np.ndarray)):
                arr = np.array(self.value, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) < 2:
                    return 0.0
                std = np.std(arr, ddof=1)
                if self.degrees_of_freedom is None:
                    self.degrees_of_freedom = float(len(arr) - 1)
                return float(std)
            return float(self.value)
        elif self.distribution == "normal":
            return float(self.value)
        elif self.distribution == "rectangular":
            return float(self.value) / math.sqrt(3)
        elif self.distribution == "triangular":
            return float(self.value) / math.sqrt(6)
        elif self.distribution == "u_shaped":
            return float(self.value) / math.sqrt(2)
        else:
            return float(self.value)

    @property
    def variance(self) -> float:
        """Variance contribution = (sensitivity * standard_uncertainty)^2."""
        return (self.sensitivity_coefficient * self.standard_uncertainty) ** 2


def measurement_uncertainty(
    type_a_sources=None,
    type_b_sources=None,
    coverage_factor=2.0,
    confidence_level=0.95,
    sensitivity_coefficients=None,
    **kwargs,
):
    """Evaluate measurement uncertainty following GUM framework.

    Args:
        type_a_sources: List of dicts with 'name' and 'values' (repeated measurements)
        type_b_sources: List of dicts with 'name', 'distribution', 'half_width'/'std'
        coverage_factor: k factor for expanded uncertainty (default 2.0 for ~95%)
        confidence_level: Coverage probability (default 0.95)
        sensitivity_coefficients: Dict mapping source names to sensitivity coefficients

    Returns:
        Dict with complete uncertainty budget
    """
    if sensitivity_coefficients is None:
        sensitivity_coefficients = {}

    sources: list[UncertaintySource] = []

    # Process Type A sources
    if type_a_sources:
        for src in type_a_sources:
            name = src.get("name", f"type_a_{len(sources)}")
            values = src.get("values", src.get("data", []))
            if not values:
                continue
            sources.append(
                UncertaintySource(
                    name=name,
                    distribution="type_a",
                    value=values if isinstance(values, list) else float(values),
                    sensitivity_coefficient=sensitivity_coefficients.get(name, 1.0),
                    degrees_of_freedom=src.get("dof"),
                )
            )

    # Process Type B sources
    if type_b_sources:
        for src in type_b_sources:
            name = src.get("name", f"type_b_{len(sources)}")
            dist = src.get("distribution", "rectangular")
            # Input can be half_width or std directly
            if "half_width" in src:
                value = src["half_width"]
            elif "std" in src:
                value = src["std"]
                # If std given directly, treat as normal
                dist = "normal"
            elif "value" in src:
                value = src["value"]
            else:
                continue
            if value is None or value < 0:
                continue
            dof = src.get("dof")
            if dof is None:
                # Default dof for Type B based on reliability of information
                dof_map = {"normal": 50, "rectangular": 100, "triangular": 25, "u_shaped": 25}
                dof = dof_map.get(dist, 50)
            sources.append(
                UncertaintySource(
                    name=name,
                    distribution=dist,
                    value=value,
                    sensitivity_coefficient=sensitivity_coefficients.get(name, 1.0),
                    degrees_of_freedom=float(dof),
                )
            )

    if not sources:
        raise ValueError("At least one uncertainty source is required")

    # Calculate combined standard uncertainty
    combined_variance = sum(s.variance for s in sources)
    combined_uncertainty = math.sqrt(combined_variance) if combined_variance > 0 else 0.0

    # Welch-Satterthwaite effective degrees of freedom
    if combined_uncertainty > 0:
        numerator = combined_uncertainty**4
        denominator = 0.0
        for s in sources:
            uc_i = s.sensitivity_coefficient * s.standard_uncertainty
            dof_i = s.degrees_of_freedom if s.degrees_of_freedom and s.degrees_of_freedom > 0 else 1e10
            denominator += (uc_i**4) / dof_i
        effective_dof = numerator / denominator if denominator > 0 else 1e10
    else:
        effective_dof = 1e10

    # Expanded uncertainty
    expanded_uncertainty = coverage_factor * combined_uncertainty

    # Build uncertainty budget table
    budget = []
    for s in sources:
        budget.append(
            {
                "source": s.name,
                "distribution": s.distribution,
                "input_value": r(s.value) if not isinstance(s.value, list) else r(float(np.std(s.value, ddof=1))),
                "sensitivity_coefficient": r(s.sensitivity_coefficient),
                "standard_uncertainty": r(s.standard_uncertainty),
                "variance_contribution": r(s.variance),
                "percent_contribution": r((s.variance / combined_variance * 100) if combined_variance > 0 else 0),
                "degrees_of_freedom": r(s.degrees_of_freedom) if s.degrees_of_freedom else "inf",
            }
        )

    # Sort by contribution (largest first)
    budget.sort(key=lambda x: x["variance_contribution"], reverse=True)

    # Build measurement result summary
    result = {
        "measurement_uncertainty": {
            "type_a_sources": len([s for s in sources if s.distribution == "type_a"]),
            "type_b_sources": len([s for s in sources if s.distribution != "type_a"]),
            "total_sources": len(sources),
        },
        "uncertainty_budget": budget,
        "combined_standard_uncertainty": r(combined_uncertainty),
        "expanded_uncertainty": r(expanded_uncertainty),
        "coverage_factor": coverage_factor,
        "confidence_level": confidence_level,
        "effective_degrees_of_freedom": r(effective_dof) if effective_dof < 1e9 else "inf",
        "relative_uncertainty_percent": r(
            (expanded_uncertainty / abs(kwargs.get("measured_value", 1.0)) * 100)
            if kwargs.get("measured_value")
            else None
        ),
        "interpretation": (
            f"Expanded uncertainty = {r(expanded_uncertainty)} (k={coverage_factor}, "
            f"{confidence_level * 100:.0f}% coverage), "
            f"combining {len(sources)} uncertainty sources"
        ),
    }

    return result
