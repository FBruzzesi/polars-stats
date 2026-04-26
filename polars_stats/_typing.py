from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import polars as pl
    from polars.datatypes import DataType, DataTypeClass

    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series
    PolarsDataType: TypeAlias = DataType | DataTypeClass
