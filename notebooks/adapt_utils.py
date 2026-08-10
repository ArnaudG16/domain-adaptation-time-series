import os
import glob
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from adapt.feature_based import CORAL, SA, TCA, fMMD, FA
from adapt.instance_based import KMM, KLIEP, TrAdaBoostR2
from adapt.parameter_based import RegularTransferLR, RegularTransferLC


def load_pair(filepath):
    data = np.load(filepath)
    shift_type, intensity_str = os.path.basename(filepath).replace(".npz", "").split("_intensity_")
    window_size = data["Xs"].shape[1]
    return shift_type, float(intensity_str), window_size, data["series_source"], data["Xt"], data["yt"]


def sliding_window(series, window_size, horizon=1):
    X, y = [], []
    for i in range(len(series) - window_size - horizon + 1):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size + horizon - 1])
    return np.array(X), np.array(y)


def build_train_test(series_source, window_size, test_ratio=0.2, horizon=1):
    split_idx = int(len(series_source) * (1 - test_ratio))
    series_train = series_source[:split_idx]
    series_test = series_source[split_idx - window_size:]

    Xs_train, ys_train = sliding_window(series_train, window_size, horizon)
    Xs_test, ys_test = sliding_window(series_test, window_size, horizon)
    return Xs_train, ys_train, Xs_test, ys_test


def split_target(Xt, yt, test_ratio=0.2):
    split_idx = int(len(Xt) * (1 - test_ratio))
    return Xt[:split_idx], yt[:split_idx], Xt[split_idx:], yt[split_idx:]


def default_model(task):
    if task == "forecast":
        return LinearRegression()
    if task == "classification":
        return LogisticRegression(penalty=None, max_iter=5000)
    raise ValueError(f"task inconnu : {task}")


def evaluate_prediction(task, X_train, y_train, X_test, y_test, model=None):
    if model is None:
        model = default_model(task)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if task == "forecast":
        return dict(mse=mean_squared_error(y_test, pred), r2=r2_score(y_test, pred))
    if task == "classification":
        return dict(accuracy=accuracy_score(y_test, pred))
    raise ValueError(f"task inconnu : {task}")


def scale_data(Xs_train, *arrays):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(Xs_train)
    rest = [scaler.transform(a) for a in arrays]
    return (Xs_train, *rest)


def compute_metrics(model, Xs_test, ys_test, Xt, yt):
    accepts_domain = "domain" in inspect.signature(model.predict).parameters

    if accepts_domain:
        pred_source = model.predict(Xs_test, domain="src")
        pred_target = model.predict(Xt, domain="tgt")
    else:
        pred_source = model.predict(Xs_test)
        pred_target = model.predict(Xt)

    return dict(
        mse_source=mean_squared_error(ys_test, pred_source),
        mse_target=mean_squared_error(yt, pred_target),
        r2_source=r2_score(ys_test, pred_source),
        r2_target=r2_score(yt, pred_target),
    )


def evaluate_baseline(series_source, window_size, Xt, yt, model, test_ratio=0.2, standardize=False, horizon=1):
    Xs_train, ys_train, Xs_test, ys_test = build_train_test(series_source, window_size, test_ratio, horizon)
    _, _, Xt_test, yt_test = split_target(Xt, yt, test_ratio)

    if standardize:
        Xs_train, Xs_test, Xt_test = scale_data(Xs_train, Xs_test, Xt_test)

    model.fit(Xs_train, ys_train)
    return compute_metrics(model, Xs_test, ys_test, Xt_test, yt_test)


def evaluate_adapt(adapt_class, series_source, window_size, Xt, yt, estimator, use_target_labels=False, pretrain_source=False, standardize=False, test_ratio=0.2, horizon=1, **adapt_kwargs):
    Xs_train, ys_train, Xs_test, ys_test = build_train_test(series_source, window_size, test_ratio, horizon)
    Xt_train, yt_train, Xt_test, yt_test = split_target(Xt, yt, test_ratio)

    if standardize:
        Xs_train, Xs_test, Xt, Xt_train, Xt_test = scale_data(Xs_train, Xs_test, Xt, Xt_train, Xt_test)

    if pretrain_source:
        estimator.fit(Xs_train, ys_train)
        model = adapt_class(estimator, random_state=0, verbose=0, **adapt_kwargs)
        model.fit(Xt_train, yt_train)
    elif use_target_labels:
        model = adapt_class(estimator, Xt=Xt_train, yt=yt_train, random_state=0, verbose=0, **adapt_kwargs)
        model.fit(Xs_train, ys_train)
    else:
        model = adapt_class(estimator, Xt=Xt, random_state=0, verbose=0, **adapt_kwargs)
        model.fit(Xs_train, ys_train)

    return compute_metrics(model, Xs_test, ys_test, Xt_test, yt_test)


def run_all_methods(shift_type, data_dir="../../data/synthetic", standardize=False):
    data_glob = f"{data_dir}/{shift_type}_intensity_*.npz"
    results = []

    for filepath in sorted(glob.glob(data_glob)):
        _, intensity, window_size, series_source, Xt, yt = load_pair(filepath)
        metrics = evaluate_baseline(series_source, window_size, Xt, yt, LinearRegression(), standardize=standardize)
        results.append(dict(method="sans adaptation", intensity=intensity, **metrics))

    unsupervised_methods = {"CORAL": CORAL, "SA": SA, "TCA": TCA, "fMMD": fMMD, "KMM": KMM, "KLIEP": KLIEP}
    for name, adapt_class in unsupervised_methods.items():
        for filepath in sorted(glob.glob(data_glob)):
            _, intensity, window_size, series_source, Xt, yt = load_pair(filepath)
            metrics = evaluate_adapt(adapt_class, series_source, window_size, Xt, yt, LinearRegression(), standardize=standardize)
            results.append(dict(method=name, intensity=intensity, **metrics))

    supervised_methods = {"FA": FA, "TrAdaBoostR2": TrAdaBoostR2}
    for name, adapt_class in supervised_methods.items():
        for filepath in sorted(glob.glob(data_glob)):
            _, intensity, window_size, series_source, Xt, yt = load_pair(filepath)
            metrics = evaluate_adapt(adapt_class, series_source, window_size, Xt, yt, LinearRegression(), use_target_labels=True, standardize=standardize)
            results.append(dict(method=name, intensity=intensity, **metrics))

    for filepath in sorted(glob.glob(data_glob)):
        _, intensity, window_size, series_source, Xt, yt = load_pair(filepath)
        metrics = evaluate_adapt(RegularTransferLR, series_source, window_size, Xt, yt, LinearRegression(),pretrain_source=True, standardize=standardize)
        results.append(dict(method="RegularTransferLR", intensity=intensity, **metrics))

    return pd.DataFrame(results)









def predict(task, X_train, y_train, X_test, y_test, estimator=None):
    if estimator is None:
        estimator = LinearRegression() if task == "forecast" else LogisticRegression(penalty=None, max_iter=5000)
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    if task == "forecast":
        return dict(mse=mean_squared_error(y_test, pred), r2=r2_score(y_test, pred))
    if task == "classification":
        return dict(accuracy=accuracy_score(y_test, pred))
    raise ValueError(f"tache inconnue : {task}")






























def forecast_series(series, window_size=20, horizon=1, test_ratio=0.2, estimator=None):
    Xs_train, ys_train, Xs_test, ys_test = build_train_test(series, window_size, test_ratio, horizon)
    return predict("forecast", Xs_train, ys_train, Xs_test, ys_test, estimator)


def courbe_apprentissage(X_train, y_train, X_test, y_test, task, n_values, seed=0):
    perm = np.random.default_rng(seed).permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    ns, perf = [], []
    for n in n_values:
        if n > len(X_train):
            continue
        if task == "classification" and len(np.unique(y_train[:n])) < 2:
            continue
        ns.append(n)
        r = predict(task, X_train[:n], y_train[:n], X_test, y_test)
        perf.append(r["r2"] if task == "forecast" else r["accuracy"])
    return ns, perf


def split_har(Xs, ys, subj_s, Xt, yt, subj_t, n_test_source=4, n_train_cible=6):
    src, tgt = sorted(set(subj_s)), sorted(set(subj_t))
    m_src = np.isin(subj_s, src[-n_test_source:])
    m_tgt_tr = np.isin(subj_t, tgt[:n_train_cible])
    m_tgt_te = np.isin(subj_t, tgt[n_train_cible:])
    return (Xs[~m_src], ys[~m_src], Xs[m_src], ys[m_src],
            Xt[m_tgt_tr], yt[m_tgt_tr], Xt[m_tgt_te], yt[m_tgt_te])


def comparer_adapt(task, Xs_train, ys_train, Xt_train, yt_train, Xt_test, yt_test, max_size=1000, score=None):
    if score is None:
        score = accuracy_score if task == "classification" else r2_score
    transfert = RegularTransferLC if task == "classification" else RegularTransferLR

    source = default_model(task).fit(Xs_train, ys_train)
    res = {"sans adaptation": score(yt_test, source.predict(Xt_test))}

    m = CORAL(default_model(task), Xt=Xt_train, random_state=0, verbose=0)
    m.fit(Xs_train, ys_train)
    res["CORAL (feature)"] = score(yt_test, m.predict(Xt_test))

    m = KMM(default_model(task), Xt=Xt_train, random_state=0, verbose=0, max_size=max_size)
    m.fit(Xs_train, ys_train)
    res["KMM (instance)"] = score(yt_test, m.predict(Xt_test))

    m = transfert(source, lambda_=1.0, random_state=0, verbose=0)
    m.fit(Xt_train, yt_train)
    res[transfert.__name__ + " (parametre)"] = score(yt_test, m.predict(Xt_test))

    cible = default_model(task).fit(Xt_train, yt_train)
    res["cible -> cible"] = score(yt_test, cible.predict(Xt_test))
    return res


def barres_adapt(res, titre, ylabel, ylim=None, plancher=None):
    res = dict(res)
    reference = res.pop("cible -> cible", None)
    plt.figure(figsize=(9, 4))
    plt.bar(list(res), [round(v, 3) for v in res.values()],
            color=["gray"] + ["tab:blue"] * (len(res) - 1))
    if reference is not None:
        plt.axhline(reference, color="k", linestyle="--", label="cible -> cible")
    if plancher is not None:
        plt.axhline(plancher, color="tab:red", linestyle=":", label="plancher de bruit")
    if reference is not None or plancher is not None:
        plt.legend()
    plt.ylabel(ylabel)
    plt.title(titre)
    plt.xticks(rotation=20, ha="right")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.show()


def courbe_melange(Xs_train, ys_train, Xt_train, yt_train, X_test, y_test, task, n_values, seed=0):
    perm = np.random.default_rng(seed).permutation(len(Xt_train))
    Xt_train, yt_train = Xt_train[perm], yt_train[perm]
    ns, perf = [], []
    for n in n_values:
        if n > len(Xt_train):
            continue
        if n == 0:
            X, y = Xs_train, ys_train
        else:
            X = np.vstack([Xs_train, Xt_train[:n]])
            y = np.concatenate([ys_train, yt_train[:n]])
        ns.append(n)
        r = predict(task, X, y, X_test, y_test)
        perf.append(r["r2"] if task == "forecast" else r["accuracy"])
    return ns, perf
