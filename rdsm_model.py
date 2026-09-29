#!/usr/bin/env python3
"""
RDSM helper used by the divorce analysis.

The RDSM architecture and recurrent-step likelihood follow the RDSM approach
in auton-survival. The rest of this file handles the longitudinal sequences,
validation/refit steps, and the landmark predictions used in this analysis.

Reference implementation: https://github.com/autonlab/auton-survival
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

def _as_object_array(items: Sequence[np.ndarray]) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = np.asarray(item)
    return out

def _feature_vector(
    x_row: np.ndarray,
    missing_row: np.ndarray,
    delta_t: float,
    calendar_time_years: float,
    max_time_years: float,
) -> np.ndarray:
    denom = float(max_time_years) if float(max_time_years) > 0 else 1.0
    return np.concatenate(
        [
            np.asarray(x_row, dtype=np.float64),
            np.asarray(missing_row, dtype=np.float64),
            np.asarray([float(delta_t)], dtype=np.float64),
            np.asarray([float(calendar_time_years) / denom], dtype=np.float64),
        ],
        axis=0,
    )

# Build one training sequence per person.
def build_person_level_rdsm_sequences(
    X_imp: np.ndarray,
    visit_mask: np.ndarray,
    x_missing_mask: np.ndarray,
    delta_t: np.ndarray,
    observed_time: np.ndarray,
    observed_year: np.ndarray,
    event_observed: np.ndarray,
    assessment_years: np.ndarray,
    min_history_visits: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    X_imp = np.asarray(X_imp)
    visit_mask = np.asarray(visit_mask)
    x_missing_mask = np.asarray(x_missing_mask)
    delta_t = np.asarray(delta_t)
    observed_time = np.asarray(observed_time).astype(int)
    observed_year = np.asarray(observed_year, dtype=float)
    event_observed = np.asarray(event_observed).astype(float)
    assessment_years = np.asarray(assessment_years, dtype=float)
    max_year = float(np.max(assessment_years))

    x_streams: List[np.ndarray] = []
    t_streams: List[np.ndarray] = []
    e_streams: List[np.ndarray] = []

    skipped_too_few = 0
    n_targets = 0

    for i in range(X_imp.shape[0]):
        obs_t = int(observed_time[i])
        obs_y = float(observed_year[i])

        visit_idx = np.where(
            (visit_mask[i] > 0) & ((np.arange(visit_mask.shape[1]) + 1) < obs_t)
        )[0]

        if len(visit_idx) < int(min_history_visits):
            skipped_too_few += 1
            continue

        x_i = []
        t_i = []
        e_i = []

        for idx in visit_idx:
            residual_years = float(obs_y - assessment_years[idx])
            if residual_years <= 0:
                continue

            x_i.append(
                _feature_vector(
                    X_imp[i, idx],
                    x_missing_mask[i, idx],
                    delta_t[i, idx],
                    calendar_time_years=float(assessment_years[idx]),
                    max_time_years=max_year,
                )
            )
            t_i.append(residual_years)
            e_i.append(float(event_observed[i]))

        if len(x_i) < int(min_history_visits):
            skipped_too_few += 1
            continue

        x_streams.append(np.asarray(x_i, dtype=np.float64))
        t_streams.append(np.asarray(t_i, dtype=np.float64))
        e_streams.append(np.asarray(e_i, dtype=np.float64))
        n_targets += int(np.isfinite(np.asarray(t_i, dtype=np.float64)).sum())

    if not x_streams:
        raise ValueError("No eligible person-level RDSM training sequences were created.")

    info = {
        "n_sequences": int(len(x_streams)),
        "n_targets": int(n_targets),
        "n_skipped_too_few_visits": int(skipped_too_few),
        "input_dim": int(x_streams[0].shape[1]),
    }
    return (
        _as_object_array(x_streams),
        _as_object_array(t_streams),
        _as_object_array(e_streams),
        info,
    )

def build_landmark_rdsm_sequences(
    landmark_meta: Dict[str, np.ndarray],
    assessment_years: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(landmark_meta["X"])
    vm = np.asarray(landmark_meta["visit_mask"])
    xm = np.asarray(landmark_meta["x_missing_mask"])
    dt = np.asarray(landmark_meta["delta_t"])
    lm = np.asarray(landmark_meta["landmark_time"]).astype(int)
    assessment_years = np.asarray(assessment_years, dtype=float)
    max_year = float(np.max(assessment_years))

    streams: List[np.ndarray] = []
    lengths: List[int] = []

    for r in range(X.shape[0]):
        idxs = np.where(
            (vm[r] > 0) & ((np.arange(vm.shape[1]) + 1) <= lm[r])
        )[0]
        if len(idxs) == 0:
            raise ValueError(f"Landmark record {r} has no observed visit in its history.")

        seq = [
            _feature_vector(
                X[r, idx],
                xm[r, idx],
                dt[r, idx],
                calendar_time_years=float(assessment_years[idx]),
                max_time_years=max_year,
            )
            for idx in idxs
        ]
        streams.append(np.asarray(seq, dtype=np.float64))
        lengths.append(len(seq))

    return _as_object_array(streams), np.asarray(lengths, dtype=np.int64)

# RDSM network.
class _LocalRDSMTorch(__import__("torch").nn.Module):
    def __init__(
        self,
        inputdim: int,
        k: int,
        layers: int = 1,
        hidden: int = 32,
        dist: str = "Weibull",
        temp: float = 1000.0,
        discount: float = 1.0,
        optimizer: str = "Adam",
        typ: str = "LSTM",
        risks: int = 1,
    ):
        import torch
        import torch.nn as nn

        super().__init__()
        self.k = int(k)
        self.dist = str(dist)
        self.temp = float(temp)
        self.discount = float(discount)
        self.optimizer = str(optimizer)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.typ = str(typ).upper()
        self.risks = int(risks)

        if self.dist == "Weibull":
            self.act = nn.SELU()
            init_shape = -1.0
            init_scale = -1.0
        elif self.dist == "LogNormal":
            self.act = nn.Tanh()
            init_shape = 1.0
            init_scale = 1.0
        else:
            raise NotImplementedError(f"Distribution {self.dist!r} is not supported.")

        self.shape = nn.ParameterDict(
            {str(r + 1): nn.Parameter(torch.full((self.k,), init_shape)) for r in range(self.risks)}
        )
        self.scale = nn.ParameterDict(
            {str(r + 1): nn.Parameter(torch.full((self.k,), init_scale)) for r in range(self.risks)}
        )
        self.gate = nn.ModuleDict(
            {str(r + 1): nn.Linear(self.hidden, self.k, bias=False) for r in range(self.risks)}
        )
        self.scaleg = nn.ModuleDict(
            {str(r + 1): nn.Linear(self.hidden, self.k, bias=True) for r in range(self.risks)}
        )
        self.shapeg = nn.ModuleDict(
            {str(r + 1): nn.Linear(self.hidden, self.k, bias=True) for r in range(self.risks)}
        )

        if self.typ == "LSTM":
            self.embedding = nn.LSTM(
                inputdim, self.hidden, self.layers, bias=False, batch_first=True
            )
        elif self.typ == "GRU":
            self.embedding = nn.GRU(
                inputdim, self.hidden, self.layers, bias=False, batch_first=True
            )
        elif self.typ == "RNN":
            self.embedding = nn.RNN(
                inputdim,
                self.hidden,
                self.layers,
                bias=False,
                batch_first=True,
                nonlinearity="relu",
            )
        else:
            raise ValueError("RDSM type must be one of LSTM, GRU, RNN.")

    def forward(self, x, risk="1"):
        import torch
        import torch.nn as nn

        x = x.detach().clone()
        inputmask = ~torch.isnan(x[:, :, 0]).reshape(-1)
        x[torch.isnan(x)] = 0.0
        xrep, _ = self.embedding(x)
        xrep = xrep.contiguous().view(-1, self.hidden)
        xrep = xrep[inputmask]
        xrep = nn.ReLU6()(xrep)
        dim = xrep.shape[0]
        return (
            self.act(self.shapeg[risk](xrep)) + self.shape[risk].expand(dim, -1),
            self.act(self.scaleg[risk](xrep)) + self.scale[risk].expand(dim, -1),
            self.gate[risk](xrep) / self.temp,
        )

    def get_shape_scale(self, risk="1"):
        return self.shape[risk], self.scale[risk]

def _pad_feature_streams(streams: np.ndarray) -> np.ndarray:
    streams = np.asarray(streams, dtype=object)
    if len(streams) == 0:
        raise ValueError("Cannot pad an empty stream collection.")
    max_len = max(len(np.asarray(x)) for x in streams)
    feat_dim = int(np.asarray(streams[0]).shape[-1])
    out = np.full((len(streams), max_len, feat_dim), np.nan, dtype=np.float64)
    for i, x in enumerate(streams):
        x = np.asarray(x, dtype=np.float64)
        out[i, : len(x)] = x
    return out

def _pad_target_streams(streams: np.ndarray) -> np.ndarray:
    streams = np.asarray(streams, dtype=object)
    if len(streams) == 0:
        raise ValueError("Cannot pad an empty target collection.")
    max_len = max(len(np.asarray(x)) for x in streams)
    out = np.full((len(streams), max_len, 1), np.nan, dtype=np.float64)
    for i, x in enumerate(streams):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        out[i, : len(x), 0] = x
    return out

def _reshape_tensor_with_nans(data):
    data = data.reshape(-1)
    return data[~__import__("torch").isnan(data)]

def _get_optimizer(model, lr: float):
    import torch

    name = str(model.optimizer)
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if name == "RMSProp":
        return torch.optim.RMSprop(model.parameters(), lr=lr)
    raise NotImplementedError(f"Optimizer {name!r} is not implemented.")

def _unconditional_single_component_loss(shape, scale, t, e, dist: str):
    import torch

    t = t.double()
    e = e.double()
    event = e == 1
    if dist == "Weibull":
        k = shape
        b = scale
        log_s = -(torch.pow(torch.exp(b) * t, torch.exp(k)))
        log_f = k + b + ((torch.exp(k) - 1.0) * (b + torch.log(t))) + log_s
    elif dist == "LogNormal":
        mu = shape
        sigma = scale
        log_f = -sigma - 0.5 * np.log(2.0 * np.pi)
        log_f = log_f - ((torch.log(t) - mu) ** 2) / (2.0 * torch.exp(2.0 * sigma))
        z = (torch.log(t) - mu) / (torch.exp(sigma) * np.sqrt(2.0))
        surv = 0.5 - 0.5 * torch.erf(z)
        log_s = torch.log(torch.clamp(surv, min=1e-12))
    else:
        raise NotImplementedError(dist)
    return -torch.where(event, log_f, log_s).mean()

def _pretrain_shape_scale(torch_model, t_train, e_train, t_valid, e_valid):
    import torch

    device = next(torch_model.parameters()).device
    dtype = next(torch_model.parameters()).dtype
    t_train = t_train.to(device=device, dtype=dtype)
    e_train = e_train.to(device=device, dtype=dtype)
    t_valid = t_valid.to(device=device, dtype=dtype)
    e_valid = e_valid.to(device=device, dtype=dtype)

    if torch_model.dist == "Weibull":
        init_shape, init_scale = -1.0, -1.0
    else:
        init_shape, init_scale = 1.0, 1.0

    shape = torch.nn.Parameter(torch.tensor(init_shape, dtype=dtype, device=device))
    scale = torch.nn.Parameter(torch.tensor(init_scale, dtype=dtype, device=device))
    opt = torch.optim.Adam([shape, scale], lr=1e-2)
    old = float("inf")
    patience = 0

    for _ in range(10000):
        opt.zero_grad()
        loss = _unconditional_single_component_loss(
            shape, scale, t_train, e_train, torch_model.dist
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            val = float(
                _unconditional_single_component_loss(
                    shape, scale, t_valid, e_valid, torch_model.dist
                ).detach().cpu()
            )
        if abs(val - old) < 1e-4:
            patience += 1
            if patience == 3:
                break
        else:
            patience = 0
        old = val

    for r in range(torch_model.risks):
        torch_model.shape[str(r + 1)].data.fill_(float(shape.detach().cpu()))
        torch_model.scale[str(r + 1)].data.fill_(float(scale.detach().cpu()))

class _LocalRDSMInterface:
    def __init__(
        self,
        k=3,
        layers=1,
        hidden=32,
        distribution="Weibull",
        temp=1000.0,
        discount=1.0,
        typ="LSTM",
        random_seed=0,
    ):
        self.k = int(k)
        self.layers = int(layers)
        self.hidden = int(hidden)
        self.dist = str(distribution)
        self.temp = float(temp)
        self.discount = float(discount)
        self.typ = str(typ).upper()
        self.random_seed = int(random_seed)
        self.fitted = False
        self.torch_model = None

    def _gen_torch_model(self, inputdim, optimizer, risks):
        import torch

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        return _LocalRDSMTorch(
            inputdim=inputdim,
            k=self.k,
            layers=self.layers,
            hidden=self.hidden,
            dist=self.dist,
            temp=self.temp,
            discount=self.discount,
            optimizer=optimizer,
            typ=self.typ,
            risks=risks,
        )

    def _preprocess_training_data(self, x, t, e, vsize, val_data, random_seed):
        import torch

        x = np.asarray(x, dtype=object)
        t = np.asarray(t, dtype=object)
        e = np.asarray(e, dtype=object)
        idx = np.arange(len(x))
        rng = np.random.default_rng(int(random_seed))
        rng.shuffle(idx)

        x_pad = _pad_feature_streams(x)
        t_pad = _pad_target_streams(t)
        e_pad = _pad_target_streams(e)
        x_train = torch.from_numpy(x_pad[idx]).double()
        t_train = torch.from_numpy(t_pad[idx]).double()
        e_train = torch.from_numpy(e_pad[idx]).double()

        if val_data is None:
            n_val = int(float(vsize) * x_train.shape[0])
            if n_val <= 0:
                raise ValueError("RDSM validation split would be empty.")
            x_val, t_val, e_val = x_train[-n_val:], t_train[-n_val:], e_train[-n_val:]
            x_train, t_train, e_train = x_train[:-n_val], t_train[:-n_val], e_train[:-n_val]
        else:
            xv, tv, ev = val_data
            x_val = torch.from_numpy(_pad_feature_streams(np.asarray(xv, dtype=object))).double()
            t_val = torch.from_numpy(_pad_target_streams(np.asarray(tv, dtype=object))).double()
            e_val = torch.from_numpy(_pad_target_streams(np.asarray(ev, dtype=object))).double()

        return x_train, t_train, e_train, x_val, t_val, e_val

    def predict_survival(self, x, t, risk=1):
        import torch
        import torch.nn as nn

        if not self.fitted or self.torch_model is None:
            raise RuntimeError("RDSM model must be fitted before prediction.")
        times = list(t) if isinstance(t, (list, tuple, np.ndarray)) else [t]
        x_pad = torch.from_numpy(_pad_feature_streams(np.asarray(x, dtype=object))).double()
        self.torch_model.eval()
        with torch.no_grad():
            shape, scale, logits = self.torch_model.forward(x_pad, str(risk))
            log_mix = nn.LogSoftmax(dim=1)(logits)
            cols = []
            for tt in times:
                tt_t = torch.full(
                    (shape.shape[0],),
                    float(tt),
                    dtype=shape.dtype,
                    device=shape.device,
                )
                if self.dist == "Weibull":
                    comp = []
                    for g in range(self.k):
                        k = shape[:, g]
                        b = scale[:, g]
                        comp.append(-(torch.pow(torch.exp(b) * tt_t, torch.exp(k))))
                    log_s = torch.stack(comp, dim=1)
                elif self.dist == "LogNormal":
                    comp = []
                    for g in range(self.k):
                        mu = shape[:, g]
                        sigma = scale[:, g]
                        z = (torch.log(tt_t) - mu) / (torch.exp(sigma) * np.sqrt(2.0))
                        surv = 0.5 - 0.5 * torch.erf(z)
                        comp.append(torch.log(torch.clamp(surv, min=1e-12)))
                    log_s = torch.stack(comp, dim=1)
                else:
                    raise NotImplementedError(self.dist)
                cols.append(torch.exp(torch.logsumexp(log_mix + log_s, dim=1)).cpu().numpy())
        return np.stack(cols, axis=1)

def _new_rdsm_interface(
    seed: int,
    k: int = 3,
    hidden: int = 32,
    layers: int = 1,
    typ: str = "LSTM",
    distribution: str = "Weibull",
):
    typ = str(typ).upper()
    if typ not in {"LSTM", "GRU", "RNN"}:
        raise ValueError("RDSM type must be one of LSTM, GRU, RNN.")
    return _LocalRDSMInterface(
        k=int(k),
        layers=int(layers),
        hidden=int(hidden),
        distribution=str(distribution),
        typ=typ,
        random_seed=int(seed),
    )

def _native_rdsm_conditional_loss(model, x, t, e, elbo=True, risk="1"):
    import torch
    import torch.nn as nn

    shape, scale, logits = model.forward(x, risk)

    t_target = _reshape_tensor_with_nans(t)
    e_target = _reshape_tensor_with_nans(e)

    if shape.shape[0] != t_target.numel():
        raise ValueError(
            f"RDSM output/target mismatch: {shape.shape[0]} outputs but "
            f"{t_target.numel()} finite recurrent-step targets."
        )
    if t_target.numel() == 0:
        raise ValueError("RDSM batch contains no recurrent-step targets.")

    lossf = []
    losss = []

    if model.dist == "Weibull":
        for g in range(model.k):
            k = shape[:, g]
            b = scale[:, g]
            surv_log = -(torch.pow(torch.exp(b) * t_target, torch.exp(k)))
            dens_log = (
                k
                + b
                + ((torch.exp(k) - 1.0) * (b + torch.log(t_target)))
                + surv_log
            )
            lossf.append(dens_log)
            losss.append(surv_log)
    elif model.dist == "LogNormal":
        for g in range(model.k):
            mu = shape[:, g]
            sigma = scale[:, g]
            dens_log = -sigma - 0.5 * np.log(2.0 * np.pi)
            dens_log = dens_log - torch.div(
                (torch.log(t_target) - mu) ** 2,
                2.0 * torch.exp(2.0 * sigma),
            )
            z = torch.div(
                torch.log(t_target) - mu,
                torch.exp(sigma) * np.sqrt(2.0),
            )
            surv = 0.5 - 0.5 * torch.erf(z)
            surv_log = torch.log(torch.clamp(surv, min=1e-12))
            lossf.append(dens_log)
            losss.append(surv_log)
    else:
        raise NotImplementedError(
            f"Native RDSM loss does not support distribution={model.dist!r}."
        )

    lossf = torch.stack(lossf, dim=1)
    losss = torch.stack(losss, dim=1)

    if elbo:
        mix = nn.Softmax(dim=1)(logits)
        log_f = (mix * lossf).sum(dim=1)
        log_s = (mix * losss).sum(dim=1)
    else:
        log_mix = nn.LogSoftmax(dim=1)(logits)
        log_f = torch.logsumexp(log_mix + lossf, dim=1)
        log_s = torch.logsumexp(log_mix + losss, dim=1)

    event = e_target == int(risk)
    alpha = float(model.discount)
    loglik = torch.where(event, log_f, alpha * log_s)

    return -loglik.mean()

def _train_rdsm_torch(
    torch_model,
    x_train,
    t_train,
    e_train,
    x_val,
    t_val,
    e_val,
    n_iter: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    early_stopping: bool,
):
    import gc
    from copy import deepcopy
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    t_train_flat = _reshape_tensor_with_nans(t_train)
    e_train_flat = _reshape_tensor_with_nans(e_train)
    t_val_flat = _reshape_tensor_with_nans(t_val)
    e_val_flat = _reshape_tensor_with_nans(e_val)

    _pretrain_shape_scale(
        torch_model, t_train_flat, e_train_flat, t_val_flat, e_val_flat
    )

    torch_model.double()
    opt = _get_optimizer(torch_model, float(learning_rate))
    nbatches = int(x_train.shape[0] / int(batch_size)) + 1

    best_state = deepcopy(torch_model.state_dict())
    best_loss = float("inf")
    best_iter = 1
    patience = 0
    oldcost = float("inf")

    for epoch0 in range(max(1, int(n_iter))):
        rng = np.random.default_rng(epoch0)
        perm = rng.permutation(x_train.shape[0])
        x_train = x_train[perm]
        t_train = t_train[perm]
        e_train = e_train[perm]
        for j in range(nbatches):
            xb = x_train[j * batch_size:(j + 1) * batch_size]
            tb = t_train[j * batch_size:(j + 1) * batch_size]
            eb = e_train[j * batch_size:(j + 1) * batch_size]
            if xb.shape[0] == 0:
                continue
            opt.zero_grad()
            loss = 0
            for r in range(torch_model.risks):
                loss += _native_rdsm_conditional_loss(
                    torch_model, xb, tb, eb, elbo=True, risk=str(r + 1)
                )
            loss.backward()
            opt.step()

        if early_stopping:
            val_loss = 0
            for r in range(torch_model.risks):
                val_loss += _native_rdsm_conditional_loss(
                    torch_model, x_val, t_val, e_val, elbo=False, risk=str(r + 1)
                )
            val_loss = float(val_loss.detach().cpu().numpy())
            if val_loss < best_loss:
                best_loss = val_loss
                best_iter = epoch0 + 1
                best_state = deepcopy(torch_model.state_dict())

            if val_loss >= oldcost:
                if patience == 2:
                    break
                patience += 1
            else:
                patience = 0
            oldcost = val_loss
        else:
            best_iter = epoch0 + 1
            best_state = deepcopy(torch_model.state_dict())

    torch_model.load_state_dict(best_state)
    del best_state
    gc.collect()
    return torch_model, int(best_iter), float(best_loss)

# First fit: use the inner validation sample to choose the stopping iteration.
def fit_rdsm_with_external_validation(
    x_train_streams: np.ndarray,
    t_train_streams: np.ndarray,
    e_train_streams: np.ndarray,
    x_val_streams: np.ndarray,
    t_val_streams: np.ndarray,
    e_val_streams: np.ndarray,
    seed: int,
    k: int = 3,
    hidden: int = 32,
    layers: int = 1,
    typ: str = "LSTM",
    distribution: str = "Weibull",
    iters: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    optimizer: str = "Adam",
):
    model = _new_rdsm_interface(
        seed=seed, k=k, hidden=hidden, layers=layers,
        typ=typ, distribution=distribution,
    )
    x_train, t_train, e_train, x_val, t_val, e_val = model._preprocess_training_data(
        x_train_streams,
        t_train_streams,
        e_train_streams,
        vsize=0.0,
        val_data=(x_val_streams, t_val_streams, e_val_streams),
        random_seed=int(seed),
    )
    inputdim = x_train.shape[-1]
    torch_model = model._gen_torch_model(inputdim, str(optimizer), risks=1)
    torch_model, best_iter, best_val_nll = _train_rdsm_torch(
        torch_model,
        x_train,
        t_train,
        e_train,
        x_val,
        t_val,
        e_val,
        n_iter=iters,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        early_stopping=True,
    )
    model.torch_model = torch_model.eval()
    model.fitted = True
    return model, best_iter, best_val_nll

# Refit from scratch on the full development sample.
def refit_rdsm_on_full_development(
    x_dev_streams: np.ndarray,
    t_dev_streams: np.ndarray,
    e_dev_streams: np.ndarray,
    selected_iters: int,
    seed: int,
    k: int = 3,
    hidden: int = 32,
    layers: int = 1,
    typ: str = "LSTM",
    distribution: str = "Weibull",
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    optimizer: str = "Adam",
):
    model = _new_rdsm_interface(
        seed=seed, k=k, hidden=hidden, layers=layers,
        typ=typ, distribution=distribution,
    )

    x_train, t_train, e_train, x_mon, t_mon, e_mon = model._preprocess_training_data(
        x_dev_streams,
        t_dev_streams,
        e_dev_streams,
        vsize=0.0,
        val_data=(x_dev_streams, t_dev_streams, e_dev_streams),
        random_seed=int(seed),
    )
    inputdim = x_train.shape[-1]
    torch_model = model._gen_torch_model(inputdim, str(optimizer), risks=1)
    torch_model, _, _ = _train_rdsm_torch(
        torch_model,
        x_train,
        t_train,
        e_train,
        x_mon,
        t_mon,
        e_mon,
        n_iter=max(1, int(selected_iters)),
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        early_stopping=False,
    )
    model.torch_model = torch_model.eval()
    model.fitted = True
    return model

def fit_rdsm(
    x_streams: np.ndarray,
    t_streams: np.ndarray,
    e_streams: np.ndarray,
    seed: int,
    k: int = 3,
    hidden: int = 32,
    layers: int = 1,
    typ: str = "LSTM",
    distribution: str = "Weibull",
    iters: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    internal_val_fraction: float = 0.15,
    optimizer: str = "Adam",
):
    n = len(x_streams)
    n_val = max(1, int(round(float(internal_val_fraction) * n)))
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(n)
    val_idx = idx[-n_val:]
    train_idx = idx[:-n_val]
    if len(train_idx) == 0:
        raise ValueError("RDSM internal split leaves no training sequences.")
    model, _, _ = fit_rdsm_with_external_validation(
        x_streams[train_idx],
        t_streams[train_idx],
        e_streams[train_idx],
        x_streams[val_idx],
        t_streams[val_idx],
        e_streams[val_idx],
        seed=seed,
        k=k,
        hidden=hidden,
        layers=layers,
        typ=typ,
        distribution=distribution,
        iters=iters,
        learning_rate=learning_rate,
        batch_size=batch_size,
        optimizer=optimizer,
    )
    return model

def _select_last_step_predictions(
    predictions: np.ndarray,
    sequence_lengths: np.ndarray,
    n_times: int,
) -> np.ndarray:
    pred = np.asarray(predictions, dtype=np.float64)
    lengths = np.asarray(sequence_lengths, dtype=np.int64)

    if pred.ndim == 1:
        pred = pred[:, None]

    total_steps = int(lengths.sum())
    if pred.shape[0] != total_steps and pred.shape[1] == total_steps:
        pred = pred.T

    if pred.shape[0] == len(lengths):
        if pred.shape[1] != n_times:
            raise ValueError(
                f"Unexpected RDSM prediction shape {pred.shape}; expected second "
                f"dimension {n_times}."
            )
        return pred

    if pred.shape[0] != total_steps:
        raise ValueError(
            f"Unexpected RDSM prediction shape {pred.shape}. Expected either "
            f"{total_steps} recurrent-step rows or {len(lengths)} stream rows."
        )

    endpoint_idx = np.cumsum(lengths) - 1
    out = pred[endpoint_idx]
    if out.shape[1] != n_times:
        raise ValueError(
            f"Unexpected selected RDSM prediction shape {out.shape}; "
            f"expected {n_times} residual-time columns."
        )
    return out

def survival_to_discrete_hazards(survival: np.ndarray) -> np.ndarray:
    S = np.asarray(survival, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("survival must have shape (n_landmarks, n_residual_times).")

    S = np.clip(S, 1e-12, 1.0)

    S = np.minimum.accumulate(S, axis=1)

    S_prev = np.concatenate([np.ones((S.shape[0], 1)), S[:, :-1]], axis=1)
    hazards = 1.0 - (S / np.clip(S_prev, 1e-12, None))
    return np.clip(hazards, 1e-8, 1.0 - 1e-8)

# Put RDSM predictions on the same landmark grid used by the other models.
def predict_landmark_residual_hazards(
    model,
    landmark_meta: Dict[str, np.ndarray],
    assessment_years: np.ndarray,
) -> np.ndarray:
    assessment_years = np.asarray(assessment_years, dtype=float)
    n_time = len(assessment_years)
    x_landmarks, lengths = build_landmark_rdsm_sequences(
        landmark_meta, assessment_years=assessment_years
    )
    lm = np.asarray(landmark_meta["landmark_time"], dtype=int)
    out = np.full((len(lm), n_time), 1e-8, dtype=np.float64)

    for s in np.unique(lm):
        rows = np.where(lm == s)[0]
        max_r = n_time - int(s)
        if max_r <= 0:
            continue
        residual_grid = [
            float(assessment_years[int(s) + r - 1] - assessment_years[int(s) - 1])
            for r in range(1, max_r + 1)
        ]
        streams_s = _as_object_array([x_landmarks[r] for r in rows])
        lengths_s = lengths[rows]
        survival_all_steps = model.predict_survival(streams_s, t=residual_grid)
        survival_last = _select_last_step_predictions(
            survival_all_steps,
            sequence_lengths=lengths_s,
            n_times=len(residual_grid),
        )
        hz = survival_to_discrete_hazards(survival_last)
        out[rows, :max_r] = hz

    return out

