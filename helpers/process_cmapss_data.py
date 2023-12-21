import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
logger.addHandler(console_handler)


def drop_noncontrib_features(df):
    features = df.copy().columns[
        5:-2
    ]  # Excluding unit number, time in cycles, RUL, operational settings
    # find and drop constant sensor measurement columns
    cols_to_drop = []
    for col in features:
        if df[col].std() < 0.01:
            features = features.drop(col)
            cols_to_drop.append(col)
    logger.info(f"Columns dropped due to standard dev < 0.01: {cols_to_drop}")
    return df[features].to_numpy()


def allocate_memory(data, num_sequences, sequence_length):
    return np.repeat(
        np.nan, repeats=sequence_length * num_sequences * data.shape[1]
    ).reshape(num_sequences, sequence_length, data.shape[1])


def get_train_targets(num_sequences_train, rul_max):
    if num_sequences_train < rul_max:
        return np.arange(num_sequences_train - 1, -1, -1).tolist()
    return np.append(
        np.repeat(rul_max, num_sequences_train - rul_max),
        np.arange(rul_max - 1, -1, -1),
    ).tolist()


def prepare_data(input_data, scaler, type="fit_transform"):
    data = (
        scaler.fit_transform(drop_noncontrib_features(input_data))
        if type == "fit_transform"
        else scaler.transform(drop_noncontrib_features(input_data))
    )
    unit_numbers = input_data["unit_number"].to_numpy()
    return data, unit_numbers


def map_test_targets(test_input_data):
    return {
        unit: rul
        for unit, rul in zip(
            test_input_data["unit_number"].unique(), test_input_data["RUL"].to_numpy()
        )
    }


def get_unit_num_sequences(
    unit_numbers, unit_number, sequence_length, num_test_windows=None, mode="train"
):
    base_count = np.count_nonzero(unit_numbers == unit_number) - sequence_length + 1
    return base_count if mode == "train" else min(base_count, num_test_windows)


def create_sequences(
    train_input_data,
    test_input_data,
    sequence_length,
    rul_max,
    target_scaler,
    num_test_windows=1,
):
    train_sequences = []
    test_sequences = []
    train_target_values = []
    test_target_values = []
    unit_indices_test = []
    num_test_windows_per_unit = []

    scaler = MinMaxScaler(feature_range=(-1, 1))

    data_train, unit_numbers_train = prepare_data(train_input_data, scaler)
    data_test, unit_numbers_test = prepare_data(
        test_input_data, scaler, type="transform"
    )

    targets_test_map = map_test_targets(test_input_data)

    unit_numbers = np.arange(
        1, min(np.unique(unit_numbers_train)[-1], np.unique(unit_numbers_test)[-1]) + 1
    )

    for unit_number in unit_numbers:
        # number of possible test and train sequences for this unit
        num_sequences_train = get_unit_num_sequences(
            unit_numbers_train, unit_number, sequence_length
        )
        num_sequences_test = get_unit_num_sequences(
            unit_numbers_test, unit_number, sequence_length, num_test_windows, "test"
        )

        assert (num_sequences_test > 0) and (
            num_sequences_train > 0
        ), "Window size too large for unit"

        num_test_windows_per_unit.append(num_sequences_test)

        unit_targets = get_train_targets(num_sequences_train, rul_max)
        train_target_values.extend(unit_targets)

        # pre allocate memory
        train_sequences.append(
            allocate_memory(data_train, num_sequences_train, sequence_length)
        )

        test_sequences.append(
            allocate_memory(data_test, num_sequences_test, sequence_length)
        )

        unit_index = (
            np.where(unit_numbers_train == unit_number)[0][0] if unit_number > 1 else 0
        )

        # train starts from. . . start of unit data
        for seq_idx in range(num_sequences_train):
            end = seq_idx + sequence_length + unit_index
            start = seq_idx + unit_index
            train_sequences[-1][seq_idx] = data_train[start:end]

        unit_index = np.where(unit_numbers_test == unit_number)[0][-1]

        # test starts from end of unit data
        for seq_idx in range(num_sequences_test):
            start = unit_index - seq_idx
            end = start - sequence_length + 1
            test_sequences[-1][seq_idx] = data_test[end : start + 1]
            test_target_values.append(targets_test_map[unit_number])
            unit_indices_test.append(unit_number)

    logger.info(
        f"Train sequences: {np.concatenate(train_sequences).shape}, Train targets: {len(train_target_values)}"
    )
    logger.info(
        f"Test sequences: {np.concatenate(test_sequences).shape}, Test targets: {len(test_target_values)}"
    )

    return (
        np.concatenate(train_sequences),
        target_scaler.fit_transform(
            np.array(train_target_values).reshape(-1, 1)
        ).reshape(-1),
        np.concatenate(test_sequences),
        np.array(test_target_values),
        np.array(unit_indices_test),
        np.array(num_test_windows_per_unit),
    )
