import torch
import numpy as np
from sklearn.metrics import mean_squared_error


def test_model_performance(
    model, X_test, df_test, unit_indices_test, device, target_scaler
):
    model.eval()
    preds = []
    true_rul = df_test["RUL"].dropna().values
    with torch.no_grad():
        for data in X_test:
            # Move data to device
            data = torch.tensor(data).float().to(device).unsqueeze(0)

            output = model(data).squeeze().cpu().numpy()
            preds.append(
                target_scaler.inverse_transform(output.reshape(-1, 1)).reshape(-1)
            )

    preds_per_engine = {i + 1: [] for i in range(len(true_rul))}
    for i, pred in enumerate(preds):
        preds_per_engine[unit_indices_test[i]].append(pred)

    mean_preds_per_engine = [np.mean(preds) for preds in preds_per_engine.values()]

    RMSE = np.sqrt(mean_squared_error(true_rul, mean_preds_per_engine))
    return RMSE, mean_preds_per_engine, preds_per_engine, true_rul


def refine_preds(true_rul, mean_preds_per_engine):
    condensed_true_rul = np.array([i for i in true_rul if i < 125])
    condensed_true_rul_indexes = [i for i, rul in enumerate(true_rul) if rul < 125]
    condensed_mean_preds_per_engine = np.array(
        [mean_preds_per_engine[i] for i in condensed_true_rul_indexes]
    )

    condensed_RMSE = np.sqrt(
        mean_squared_error(condensed_true_rul, condensed_mean_preds_per_engine)
    )

    return condensed_RMSE, condensed_mean_preds_per_engine, condensed_true_rul


def compute_s_score(rul_true, rul_pred):
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1))
