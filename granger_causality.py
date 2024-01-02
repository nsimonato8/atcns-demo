"""
This module serves the purpose of detecting if two data series are Granger-Caused or not.

    Notes
    -----
    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the p-values are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

"""
from typing import Iterable
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

TimeSeries = pd.DataFrame
SeriesName = str | pd.Index


def grangers_causation_matrix(data: TimeSeries, variables: list[str], test: str = 'ssr_chi2test', verbose: bool = False,
                              maxlag: int | Iterable[int] = 4) -> pd.DataFrame:
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    :param maxlag: If an integer, computes the test for all lags up to maxlag. If an iterable, computes the tests only
                   for the lags in maxlag.
    :param bool verbose: If True, more output will be printed.
    :param test: String that defines the kind of test to perform to compute the p-value
    :param TimeSeries data: Pandas dataframe containing the time series variables.
    :param variables: List containing names of the time series variables.

    :return: DataFrame of the p-values of the Granger causality test, with the X and Y variables shown in the columns
             and rows respectively.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def is_granger_caused(granger_causality_table: pd.DataFrame, y: SeriesName, x: SeriesName,
                      threshold: float = .05) -> bool:
    """

    :param pd.DataFrame granger_causality_table: Pandas dataframe containing the p-values of the Granger Causality test,
                                                 as defined in the return value description
                                                 of the grangers_causation_matrix matrix.
    :param SeriesName y: The response variable to test.
    :param SeriesName x: The prediction variable.
        :param threshold:
    :return: True iff @y is Granger-Caused by @x.
    """
    return granger_causality_table.loc[y, x] < threshold
