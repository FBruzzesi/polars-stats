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
        "discreteuniform",
        "exponential",
        "geometric",
        "lognormal",
        "normal",
        "uniform",
    ]
    """Prefix of one distribution's Rust plugins, `<name>_<function>`."""

    ValueFunction: TypeAlias = Literal["pdf", "ln_pdf", "pmf", "ln_pmf", "cdf", "ln_cdf", "sf", "ln_sf", "ppf", "isf"]
    """Value-keyed plugins `f(value, *params)`, each with a constant-parameter `_scalar` twin."""

    SamplerFunction: TypeAlias = Literal["sample", "samples"]
    """Samplers `f(*params, row_index)`, each with a `_scalar` twin over the row index alone."""

    ParamFunction: TypeAlias = Literal["entropy", "p", "params", "proba", "range", "rate", "sigma"]
    """Parameter-keyed plugins `f(*params)`: the validators and the moments with no closed form. No twin."""

    PluginFunction: TypeAlias = ValueFunction | SamplerFunction | ParamFunction
