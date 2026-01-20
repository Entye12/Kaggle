import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.ensemble import IsolationForest
import math





def detect_outliers_zscore(
    df: pd.DataFrame,
    col: str,
    *,
    id_col: str = "Id",
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Univariate z-score outlier detection
    """
    z = np.abs(stats.zscore(df[col], nan_policy="omit"))

    out = pd.DataFrame({
        id_col: df[id_col],
        "z_score": z,
        "is_outlier": z > threshold,
        "method": f"zscore_{col}",
    })

    return out


def detect_outliers_residuals(
        df: pd.DataFrame,
        x_cols: list[str],
        y_col: str,
        id_col: str = "Id",
        threshold: float = 3.0,
        log_y: bool = False,
) -> pd.DataFrame :
    """
    Outliers based on standardized regression residuals
    """

    data = df[x_cols + [y_col,id_col]].dropna()

    X = data[x_cols]
    y = np.log1p(data[y_col]) if log_y else data[y_col]

    model = LinearRegression()
    model.fit(X,y)

    residuals = y - model.predict(X)
    std_resid = np.abs(residuals / residuals.std())

    out = pd.DataFrame({
        id_col: data[id_col],
        "score": std_resid,
        "is_outlier": std_resid > threshold,
        "method": f"residuals_{y_col}"
    })

    return out


def detect_outliers_cook(
        df: pd.DataFrame,
        *,
        x_cols: list[str],
        y_col: str,
        id_col: str = "Id",
        log_y: bool = False,
) -> pd.DataFrame :
    """
    Cook's distance using statsmodels
    """

    data = df[x_cols + [y_col,id_col]].dropna()

    X = sm.add_constant(data[x_cols])
    y = np.log1p(data[y_col]) if log_y else data[y_col]

    model = sm.OLS(y, X).fit()
    cooks_d = model.get_influence().cooks_distance[0]

    threshold = 4/len(data)

    out = pd.DataFrame({
        id_col: data[id_col],
        "cooks_d": cooks_d,
        "is_outlier": cooks_d > threshold,
        "method": "cooks_distance",
    })

    return out


def detect_outliers_iforest(
    df: pd.DataFrame,
    *,
    x_cols: list[str],
    id_col: str = "Id",
    contamination: float = 0.01,
    random_state: int = 42,
    n_estimators: int = 250,
    max_samples: str = 'auto',
) -> pd.DataFrame:
    """
    Isolation Forest multivariate outlier detection
    """
    data = df[x_cols + [id_col]].dropna()

    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state
    )

    preds = iso.fit_predict(data[x_cols])
    scores = -iso.decision_function(data[x_cols])

    out = pd.DataFrame({
        id_col: data[id_col],
        "score": scores,
        "is_outlier": preds == -1,
        "method": "isolation_forest",
    })

    return out


def combine_vote_outliers(
    results: list[pd.DataFrame],
    *,
    id_col: str = "Id",
    min_votes: int = 2,
) -> pd.DataFrame:
    """
    Combine multiple outlier detectors using voting
    """
    df = results[0][[id_col]].copy()

    for r in results:
        df = df.merge(
            r[[id_col, "is_outlier"]],
            on=id_col,
            how="left",
        )

    vote_cols = df.columns.drop(id_col)
    df["votes"] = df[vote_cols].sum(axis=1)
    df["is_outlier"] = df["votes"] >= min_votes

    return df.sort_values("votes", ascending=False)
