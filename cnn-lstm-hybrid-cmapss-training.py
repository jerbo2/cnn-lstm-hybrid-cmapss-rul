from helpers import get_cmapss_data, process_cmapss_data, train_model
from helpers.model import RULPredictionModel
from sklearn.preprocessing import MinMaxScaler
import torch, logging, argparse
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def comma_separated_ints(string):
    return [int(s) for s in string.split(",")]


def comma_separated_floats(string):
    return [float(s) for s in string.split(",")]


def comma_separated_range(string):
    return range(
        int(string.split(",")[0]), int(string.split(",")[1]), int(string.split(",")[2])
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--engine_name",
    type=str,
    default="FD001",
    help="Name of the engine i.e. FD001, FD002, FD003, FD004m OR all",
)
parser.add_argument(
    "--sequence_length_offsets",
    type=comma_separated_range,
    default=range(0, 15, 5),
    help="Offsets from the max possible sequence/window length, used as hyperparameter",
)
parser.add_argument(
    "--num_lstms", type=comma_separated_ints, default=[2], help="Number of LSTM layers"
)
parser.add_argument(
    "--lrs", type=comma_separated_floats, default=[0.0005], help="Learning rates"
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument(
    "--rul_max",
    type=int,
    default=125,
    help="Maximum RUL value (for piecewise linear degradation function)",
)
parser.add_argument(
    "--hidden_size", type=int, default=32, help="Hidden size of LSTM layers"
)

args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("train_model.log")
logger.addHandler(console_handler)
logger.addHandler(file_handler)

target_scaler = MinMaxScaler(feature_range=(0, 1))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
engine_data = get_cmapss_data.run(args.engine_name)

criterion = nn.MSELoss()


# Helper function to set up data loaders
def setup_data_loaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(
        val_dataset, batch_size=batch_size
    )


# Find the maximum possible sequence length for a given engine based on the number of cycles in the test data
def get_max_possible_sequence_length(df_test):
    return df_test.groupby("unit_number").size().min()


logger.info(
    f"Training will require {len(engine_data) * len(args.num_lstms) * len(args.lrs) * len(args.sequence_length_offsets)} iterations"
)

# Sweep over hyperparameters
for i, (data_train, data_test) in enumerate(engine_data):
    engine_number = i + 1 if len(engine_data) > 1 else int(args.engine_name[-1])
    logger.info(f"Engine {engine_number}")
    max_sequence_length = get_max_possible_sequence_length(data_test)

    for offset in args.sequence_length_offsets:
        logger.info(f"Offset: {offset}")
        sequence_length = max_sequence_length - offset
        logger.info(f"Sequence length: {sequence_length}")

        # Creating sequences
        (
            X,
            y,
            X_test,
            y_test,
            unit_indices_test,
            num_test_windows_list,
        ) = process_cmapss_data.create_sequences(
            data_train,
            data_test,
            sequence_length,
            args.rul_max,
            num_test_windows=5,
            target_scaler=target_scaler,
        )

        # Splitting the data into training and validation sets (80-20 split)
        (
            X_train,
            X_val,
            y_train,
            y_val,
        ) = train_test_split(X, y, test_size=0.2, random_state=83)

        train_loader, val_loader = setup_data_loaders(
            X_train, y_train, X_val, y_val, args.batch_size
        )

        num_features = X_train.shape[2]

        for num_lstms in args.num_lstms:
            for lr in args.lrs:
                logger.info(
                    f"Training with {num_lstms} LSTM layers and learning rate: {lr}"
                )
                model = RULPredictionModel(
                    num_features, args.hidden_size, num_lstms
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                train_model.train_model(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    num_epochs=30,
                    engine_number=engine_number,
                    current_config={
                        "sequence_length": sequence_length,
                        "num_lstms": num_lstms,
                        "learning_rate": lr,
                        "rul_max": args.rul_max,
                        "hidden_size": args.hidden_size,
                    },
                    device=device,
                    target_scaler=target_scaler,
                    engine_test_data=X_test,
                    engine_test_df=data_test,
                    engine_test_unit_indices=unit_indices_test,
                    patience=3,
                )
