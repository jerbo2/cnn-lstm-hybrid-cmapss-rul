import pandas as pd


# Read data using pandas read_csv
def read_data(path, columns):
    df = pd.read_csv(
        path,
        sep=" ",
        names=columns,
        header=None,
        engine="python",
        skipinitialspace=True,
        index_col=False,
    )
    df.dropna(axis=1, how="all", inplace=True)
    return df


# Get the raw data from the CMAPSS dataset, append basic linear degradation trend as RUL
def get_raw_data(engine_name):
    path_train = f"CMAPSSData/train_{engine_name}.txt"
    path_test = f"CMAPSSData/test_{engine_name}.txt"
    path_test_rul = f"CMAPSSData/RUL_{engine_name}.txt"
    columns = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_measurement_{i}" for i in range(1, 23)]
    )
    # Reading the file into a DataFrame
    df_train = read_data(path_train, columns)
    # linear degradation trend
    df_train["RUL"] = (
        df_train.groupby("unit_number")["time_in_cycles"].transform("max")
        - df_train["time_in_cycles"]
    )

    df_test = read_data(path_test, columns)
    test_rul = pd.read_csv(
        path_test_rul,
        sep=" ",
        names=["RUL"],
        header=None,
        engine="python",
        skipinitialspace=True,
        index_col=False,
    )
    df_test["RUL"] = test_rul["RUL"]
    return df_train, df_test


def run(engine_name):
    engine_data = []
    for engine in ["FD001", "FD002", "FD003", "FD004"]:
        if engine == engine_name or engine_name == "all":
            df_train, df_test = get_raw_data(engine)
            engine_data.append((df_train, df_test))
    return engine_data
