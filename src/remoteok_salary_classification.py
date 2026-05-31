from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_N_TAGS = 80
TOP_N_LOCATIONS = 20
TOP_N_TITLE_TERMS = 50
IMPORTANCE_SAMPLE_SIZE = 3500
IMPORTANCE_REPEATS = 3
SHAP_SAMPLE_SIZE = 250
SHAP_BACKGROUND_SIZE = 200

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "dataset" / "remoteok_jobs_raw.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"


TITLE_STOPWORDS = {
    "and",
    "for",
    "the",
    "with",
    "remote",
    "work",
    "from",
    "home",
    "to",
    "of",
    "in",
    "on",
    "at",
    "a",
    "an",
    "ii",
    "iii",
    "iv",
}


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned or "blank"


def split_pipe(value: str) -> list[str]:
    return [part.strip().lower() for part in str(value).split("|") if part.strip()]


def tokenize_title(value: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]{1,}", str(value).lower())
    return [token.strip(".-+#") for token in tokens if token not in TITLE_STOPWORDS and len(token) >= 3]


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(path)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["salary_mid"] = (df["salary_min"] + df["salary_max"]) / 2

    valid_salary = (
        (df["salary_min"] > 0)
        & (df["salary_max"] > 0)
        & (df["salary_max"] >= df["salary_min"])
        & (df["salary_mid"] >= 10_000)
        & (df["salary_mid"] <= 400_000)
        & df["date_parsed"].notna()
    )
    cleaned = df.loc[valid_salary].copy()
    threshold = float(cleaned["salary_mid"].median())
    cleaned["high_salary"] = (cleaned["salary_mid"] >= threshold).astype(int)

    summary = {
        "raw_rows": int(len(df)),
        "cleaned_rows": int(len(cleaned)),
        "removed_rows": int(len(df) - len(cleaned)),
        "salary_mid_threshold": threshold,
        "date_min": cleaned["date_parsed"].min().isoformat(),
        "date_max": cleaned["date_parsed"].max().isoformat(),
        "positive_class_share": float(cleaned["high_salary"].mean()),
        "salary_mid_min": float(cleaned["salary_mid"].min()),
        "salary_mid_max": float(cleaned["salary_mid"].max()),
    }
    return cleaned, summary


def top_values(series: pd.Series, top_n: int, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    counts: dict[str, int] = {}
    for raw_value in series:
        for value in split_pipe(raw_value):
            if value in exclude:
                continue
            counts[value] = counts.get(value, 0) + 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]]


def top_title_terms(series: pd.Series, top_n: int) -> list[str]:
    counts: dict[str, int] = {}
    for raw_value in series:
        for token in set(tokenize_title(raw_value)):
            counts[token] = counts.get(token, 0) + 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict[str, list[str]]]:
    latest_date = df["date_parsed"].max()
    feature_data: dict[str, pd.Series] = {}

    feature_data["posting_age_days"] = (latest_date - df["date_parsed"]).dt.days.astype(float)
    feature_data["post_year"] = df["date_parsed"].dt.year.astype(float)
    feature_data["post_month"] = df["date_parsed"].dt.month.astype(float)
    feature_data["description_chars_log"] = np.log1p(df["description"].astype(str).str.len())
    feature_data["title_chars_log"] = np.log1p(df["position"].astype(str).str.len())
    feature_data["tag_count"] = df["tags"].map(lambda value: len(split_pipe(value))).astype(float)
    feature_data["location_count"] = df["location"].map(lambda value: len(split_pipe(value))).astype(float)

    labels = {
        "posting_age_days": "Posting age in days",
        "post_year": "Posting year",
        "post_month": "Posting month",
        "description_chars_log": "Description length, log",
        "title_chars_log": "Title length, log",
        "tag_count": "Number of tags",
        "location_count": "Number of location labels",
    }

    tags = top_values(df["tags"], TOP_N_TAGS)
    locations = top_values(df["location"], TOP_N_LOCATIONS, exclude={"anywhere"})
    title_terms = top_title_terms(df["position"], TOP_N_TITLE_TERMS)

    tag_sets = df["tags"].map(lambda value: set(split_pipe(value)))
    location_sets = df["location"].map(lambda value: set(split_pipe(value)))
    title_sets = df["position"].map(lambda value: set(tokenize_title(value)))

    for tag in tags:
        col = f"tag__{slug(tag)}"
        feature_data[col] = tag_sets.map(lambda values, tag=tag: int(tag in values))
        labels[col] = f"Tag: {tag}"

    for location in locations:
        col = f"location__{slug(location)}"
        feature_data[col] = location_sets.map(lambda values, location=location: int(location in values))
        labels[col] = f"Location: {location}"

    for term in title_terms:
        col = f"title__{slug(term)}"
        feature_data[col] = title_sets.map(lambda values, term=term: int(term in values))
        labels[col] = f"Title term: {term}"

    features = pd.DataFrame(feature_data, index=df.index)
    non_constant = features.nunique(dropna=False) > 1
    features = features.loc[:, non_constant]
    labels = {column: labels[column] for column in features.columns}

    vocab = {
        "tags": tags,
        "locations": locations,
        "title_terms": title_terms,
    }
    return features, labels, vocab


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.05,
            l2_regularization=0.02,
            random_state=RANDOM_STATE,
        ),
    }


def predict_scores(model: object, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(x)
        return 1 / (1 + np.exp(-decision))
    raise TypeError(f"Model {type(model).__name__} does not provide probabilities or decision scores.")


def metrics_for_split(model_name: str, split: str, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "model": model_name,
        "split": split,
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "g_score": math.sqrt(sensitivity * specificity),
        "roc_auc": roc_auc_score(y_true, y_score),
        "entropy_log_loss": log_loss(y_true, np.column_stack([1 - y_score, y_score]), labels=[0, 1]),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_model(model_name: str, model: object, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> list[dict[str, object]]:
    model.fit(x_train, y_train)
    rows: list[dict[str, object]] = []
    for split, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = model.predict(x)
        y_score = np.clip(predict_scores(model, x), 1e-9, 1 - 1e-9)
        rows.append(metrics_for_split(model_name, split, y, y_pred, y_score))
    return rows


def importance_sample(x_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(x_test) <= IMPORTANCE_SAMPLE_SIZE:
        return x_test, y_test
    _, x_sample, _, y_sample = train_test_split(
        x_test,
        y_test,
        test_size=IMPORTANCE_SAMPLE_SIZE,
        stratify=y_test,
        random_state=RANDOM_STATE,
    )
    return x_sample, y_sample


def sampled_frame(frame: pd.DataFrame, sample_size: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame
    return frame.sample(n=sample_size, random_state=random_state)


def compute_importance(
    model_name: str,
    model: object,
    x_sample: pd.DataFrame,
    y_sample: pd.Series,
    feature_labels: dict[str, str],
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_sample,
        y_sample,
        scoring="f1",
        n_repeats=IMPORTANCE_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rows = []
    order = np.argsort(result.importances_mean)[::-1]
    for rank, idx in enumerate(order, start=1):
        feature = x_sample.columns[idx]
        rows.append(
            {
                "model": model_name,
                "rank": rank,
                "feature": feature,
                "feature_label": feature_labels.get(feature, feature),
                "importance_mean": result.importances_mean[idx],
                "importance_std": result.importances_std[idx],
            }
        )
    return pd.DataFrame(rows)


def extract_positive_class_shap(values: shap.Explanation) -> np.ndarray:
    shap_values = np.asarray(values.values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def compute_shap_importance(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    feature_labels: dict[str, str],
) -> pd.DataFrame:
    if model_name == "Logistic Regression":
        scaler = model.named_steps["standardscaler"]
        estimator = model.named_steps["logisticregression"]
        background = sampled_frame(x_train, SHAP_BACKGROUND_SIZE)
        background_scaled = pd.DataFrame(scaler.transform(background), columns=x_train.columns)
        sample_scaled = pd.DataFrame(scaler.transform(x_sample), columns=x_sample.columns)
        explainer = shap.LinearExplainer(estimator, background_scaled)
        values = explainer(sample_scaled)
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer(x_sample, check_additivity=False)

    shap_values = extract_positive_class_shap(values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    std_abs = np.abs(shap_values).std(axis=0)

    rows = []
    order = np.argsort(mean_abs)[::-1]
    for rank, idx in enumerate(order, start=1):
        feature = x_sample.columns[idx]
        rows.append(
            {
                "model": model_name,
                "rank": rank,
                "feature": feature,
                "feature_label": feature_labels.get(feature, feature),
                "importance_mean": mean_abs[idx],
                "importance_std": std_abs[idx],
                "shap_mean_signed": mean_signed[idx],
            }
        )
    return pd.DataFrame(rows)


def common_features(importance_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    model_names = sorted(importance_df["model"].unique())
    top_sets = {
        model: set(importance_df.loc[(importance_df["model"] == model) & (importance_df["rank"] <= top_k), "feature"])
        for model in model_names
    }
    common = set.intersection(*top_sets.values()) if top_sets else set()
    rows = []
    for feature in sorted(common):
        feature_rows = importance_df[importance_df["feature"] == feature]
        row = {
            "top_k": top_k,
            "feature": feature,
            "feature_label": feature_rows["feature_label"].iloc[0],
        }
        for model in model_names:
            model_row = feature_rows[feature_rows["model"] == model].iloc[0]
            row[f"{slug(model)}_rank"] = int(model_row["rank"])
            row[f"{slug(model)}_importance"] = float(model_row["importance_mean"])
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(
    cleaned: pd.DataFrame,
    summary: dict[str, object],
    vocab: dict[str, list[str]],
    metrics_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    common_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_common_df: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summary = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in summary.items()
        ]
    )
    class_balance = (
        cleaned["high_salary"]
        .value_counts()
        .rename_axis("high_salary")
        .reset_index(name="row_count")
        .assign(class_label=lambda frame: frame["high_salary"].map({0: "Below threshold", 1: "At or above threshold"}))
    )

    dataset_summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    class_balance.to_csv(OUTPUT_DIR / "class_balance.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_train_test.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "top_features_by_model.csv", index=False)
    common_df.to_csv(OUTPUT_DIR / "common_important_features.csv", index=False)
    shap_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)
    shap_common_df.to_csv(OUTPUT_DIR / "shap_common_important_features.csv", index=False)
    with (OUTPUT_DIR / "feature_vocabulary.json").open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)


def main() -> None:
    cleaned, summary = load_and_clean_data(DATA_PATH)
    x, feature_labels, vocab = build_features(cleaned)
    y = cleaned["high_salary"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    metric_rows = []
    importance_frames = []
    shap_frames = []
    x_sample, y_sample = importance_sample(x_test, y_test)
    x_shap_sample = sampled_frame(x_test, SHAP_SAMPLE_SIZE)

    for model_name, model in models.items():
        metric_rows.extend(evaluate_model(model_name, model, x_train, x_test, y_train, y_test))
        importance_frames.append(compute_importance(model_name, model, x_sample, y_sample, feature_labels))
        shap_frames.append(compute_shap_importance(model_name, model, x_train, x_shap_sample, feature_labels))

    metrics_df = pd.DataFrame(metric_rows)
    importance_df = pd.concat(importance_frames, ignore_index=True)
    common_df = common_features(importance_df, top_k=20)
    if common_df.empty:
        common_df = common_features(importance_df, top_k=30)
    shap_df = pd.concat(shap_frames, ignore_index=True)
    shap_common_df = common_features(shap_df, top_k=20)
    if shap_common_df.empty:
        shap_common_df = common_features(shap_df, top_k=30)

    save_outputs(cleaned, summary, vocab, metrics_df, importance_df, common_df, shap_df, shap_common_df)

    print("Done. Outputs written to:")
    for path in sorted(OUTPUT_DIR.glob("*")):
        print(f"- {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
