from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from typing import Literal

    import polars as pl
    from polars.datatypes import DataType, DataTypeClass

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series
    PolarsDataType: TypeAlias = DataType | DataTypeClass

    DistributionName: TypeAlias = Literal[
        "bernoulli",
        "beta",
        "binomial",
        "cauchy",
        "discreteuniform",
        "exponential",
        "geometric",
        "lognormal",
        "normal",
        "pareto",
        "uniform",
        "weibull",
    ]
    """Prefix of one distribution's Rust plugins, `<name>_<function>`."""

    ValueFunction: TypeAlias = Literal["pdf", "ln_pdf", "pmf", "ln_pmf", "cdf", "ln_cdf", "sf", "ln_sf", "ppf", "isf"]
    """Value-keyed plugins `f(value, *params)`, each with a constant-parameter `_scalar` twin."""

    SamplerFunction: TypeAlias = Literal["sample", "samples"]
    """Samplers `f(*params, row_index)`, each with a `_scalar` twin over the row index alone."""

    ValidatorFunction: TypeAlias = Literal["p", "params", "proba", "range", "rate", "scale", "shape", "sigma"]
    """Parameter-keyed validators `f(*params)`: raise on an invalid parameterisation, return a reusable quantity."""

    MomentFunction: TypeAlias = Literal["entropy", "mean", "std", "variance"]
    """Parameter-keyed moments `f(*params)` with no closed form, so the validation rides inside the moment."""

    ParamFunction: TypeAlias = ValidatorFunction | MomentFunction
    """Every parameter-keyed plugin. No `_scalar` twin: constant parameters run it once on length-1 literals."""

    PluginFunction: TypeAlias = ValueFunction | SamplerFunction | ParamFunction
