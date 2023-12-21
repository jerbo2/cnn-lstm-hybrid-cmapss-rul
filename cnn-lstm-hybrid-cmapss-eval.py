import os, argparse, torch, logging, json
from helpers import get_cmapss_data, process_cmapss_data, test_model
from helpers.model import RULPredictionModel
from sklearn.preprocessing import MinMaxScaler

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--engine_name",
    type=str,
    default="FD001",
    help="Name of the engine i.e. FD001, FD002, FD003, FD004m OR all",
)

args = argparser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

engine_number = int(args.engine_name[-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 32

current_engine_config = json.load(
    open(f"configs/best_config_engine_{engine_number}.json")
)
model_path = [
    state
    for state in os.listdir("./best_saved_models")
    if f"engine_{engine_number}" in state
][0]
engine_data = get_cmapss_data.run(args.engine_name)

target_scaler = MinMaxScaler(feature_range=(0, 1))

(
    _,
    _,
    X_test,
    _,
    unit_indices_test,
    num_test_windows_list,
) = process_cmapss_data.create_sequences(
    engine_data[0][0],
    engine_data[0][1],
    sequence_length=int(current_engine_config["sequence_length"]),
    rul_max=int(current_engine_config["rul_max"]),
    num_test_windows=5,
    target_scaler=target_scaler,
)

num_features = X_test.shape[2]

model = RULPredictionModel(
    num_features,
    int(current_engine_config["hidden_size"]),
    num_lstms=int(current_engine_config["num_lstms"]),
).to(device)
model.load_state_dict(torch.load(f"./best_saved_models/{model_path}"))

(
    RMSE,
    mean_preds_per_engine,
    preds_per_engine,
    true_rul,
) = test_model.test_model_performance(
    model, X_test, engine_data[0][1], unit_indices_test, device, target_scaler
)


(
    condensed_RMSE,
    condensed_mean_preds_per_engine,
    condensed_true_rul,
) = test_model.refine_preds(true_rul, mean_preds_per_engine)

s_score = test_model.compute_s_score(true_rul, mean_preds_per_engine)
condensed_s_score = test_model.compute_s_score(
    condensed_true_rul, condensed_mean_preds_per_engine
)

logger.info("*" * 50)
logger.info(f"RMSE: {RMSE:.4f}")
logger.info(f"Condensed RMSE: {condensed_RMSE:.4f}")
logger.info(f"S-Score: {s_score:.4f}")
logger.info(f"Condensed S-Score: {condensed_s_score:.4f}")
logger.info("*" * 50)
