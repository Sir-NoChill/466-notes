import polars as pl

def make_prediction(
        data: pl.DataFrame, 
        lags: pl.DataFrame | None
        ) -> pl.DataFrame:
    return data.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )

