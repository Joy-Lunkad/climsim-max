import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import enum
import xarray as xr

# These features have the same mean, min, and max so we can drop them
USELESS_FEATURES = [
    "pbuf_CH4_27",
    "pbuf_CH4_28",
    "pbuf_CH4_29",
    "pbuf_CH4_30",
    "pbuf_CH4_31",
    "pbuf_CH4_32",
    "pbuf_CH4_33",
    "pbuf_CH4_34",
    "pbuf_CH4_35",
    "pbuf_CH4_36",
    "pbuf_CH4_37",
    "pbuf_CH4_38",
    "pbuf_CH4_39",
    "pbuf_CH4_40",
    "pbuf_CH4_41",
    "pbuf_CH4_42",
    "pbuf_CH4_43",
    "pbuf_CH4_44",
    "pbuf_CH4_45",
    "pbuf_CH4_46",
    "pbuf_CH4_47",
    "pbuf_CH4_48",
    "pbuf_CH4_49",
    "pbuf_CH4_50",
    "pbuf_CH4_51",
    "pbuf_CH4_52",
    "pbuf_CH4_53",
    "pbuf_CH4_54",
    "pbuf_CH4_55",
    "pbuf_CH4_56",
    "pbuf_CH4_57",
    "pbuf_CH4_58",
    "pbuf_CH4_59",
    "pbuf_N2O_27",
    "pbuf_N2O_28",
    "pbuf_N2O_29",
    "pbuf_N2O_30",
    "pbuf_N2O_31",
    "pbuf_N2O_32",
    "pbuf_N2O_33",
    "pbuf_N2O_34",
    "pbuf_N2O_35",
    "pbuf_N2O_36",
    "pbuf_N2O_37",
    "pbuf_N2O_38",
    "pbuf_N2O_39",
    "pbuf_N2O_40",
    "pbuf_N2O_41",
    "pbuf_N2O_42",
    "pbuf_N2O_43",
    "pbuf_N2O_44",
    "pbuf_N2O_45",
    "pbuf_N2O_46",
    "pbuf_N2O_47",
    "pbuf_N2O_48",
    "pbuf_N2O_49",
    "pbuf_N2O_50",
    "pbuf_N2O_51",
    "pbuf_N2O_52",
    "pbuf_N2O_53",
    "pbuf_N2O_54",
    "pbuf_N2O_55",
    "pbuf_N2O_56",
    "pbuf_N2O_57",
    "pbuf_N2O_58",
    "pbuf_N2O_59",
]

ABLATED_COL_NAMES = [
    "ptend_q0001_0",
    "ptend_q0001_1",
    "ptend_q0001_2",
    "ptend_q0001_3",
    "ptend_q0001_4",
    "ptend_q0001_5",
    "ptend_q0001_6",
    "ptend_q0001_7",
    "ptend_q0001_8",
    "ptend_q0001_9",
    "ptend_q0001_10",
    "ptend_q0001_11",
    "ptend_q0002_0",
    "ptend_q0002_1",
    "ptend_q0002_2",
    "ptend_q0002_3",
    "ptend_q0002_4",
    "ptend_q0002_5",
    "ptend_q0002_6",
    "ptend_q0002_7",
    "ptend_q0002_8",
    "ptend_q0002_9",
    "ptend_q0002_10",
    "ptend_q0002_11",
    "ptend_q0003_0",
    "ptend_q0003_1",
    "ptend_q0003_2",
    "ptend_q0003_3",
    "ptend_q0003_4",
    "ptend_q0003_5",
    "ptend_q0003_6",
    "ptend_q0003_7",
    "ptend_q0003_8",
    "ptend_q0003_9",
    "ptend_q0003_10",
    "ptend_q0003_11",
    "ptend_u_0",
    "ptend_u_1",
    "ptend_u_2",
    "ptend_u_3",
    "ptend_u_4",
    "ptend_u_5",
    "ptend_u_6",
    "ptend_u_7",
    "ptend_u_8",
    "ptend_u_9",
    "ptend_u_10",
    "ptend_u_11",
    "ptend_v_0",
    "ptend_v_1",
    "ptend_v_2",
    "ptend_v_3",
    "ptend_v_4",
    "ptend_v_5",
    "ptend_v_6",
    "ptend_v_7",
    "ptend_v_8",
    "ptend_v_9",
    "ptend_v_10",
    "ptend_v_11",
]


def min_max_norm(df: pd.DataFrame, norm_path: str = "./"):
    df_stats = pd.read_csv(
        os.path.join(norm_path, "train_stats.csv"),
        index_col=0,
    )
    for col in df.columns.to_list():
        df[col] = (df[col] - df_stats[col]["min"]) / (
            df_stats[col]["max"] - df_stats[col]["min"]
        )
    return df


def z_norm(df: pd.DataFrame, norm_path: str = "./"):
    df_stats = pd.read_csv(
        os.path.join(norm_path, "train_stats.csv"),
        index_col=0,
    )
    for col in df.columns.to_list():
        df[col] = (df[col] - df_stats[col]["mean"]) / df_stats[col]["std"]
    return df


# Enum for types of norms
class NormType(enum.Enum):
    MIN_MAX = 1
    Z_NORM = 2


def process_data(
    csv_path: str,
    norm_path: str,
    which_norm: NormType = NormType.Z_NORM,
    train_nrows: int | None = None,
    test_nrows: int | None = None,
):

    # ---------------------- Training Data --------------------------------

    train_df = pd.read_csv(
        os.path.join(csv_path, "train.csv"),
        nrows=train_nrows,
    )
    
    train_ids = train_df['sample_id']

    x_train = train_df.iloc[:, 1:557]
    y_train = train_df.iloc[:, 557:]

    x_train.drop(USELESS_FEATURES, axis=1, inplace=True)
    y_train.drop(ABLATED_COL_NAMES, axis=1, inplace=True)
    
    # ---------------------- Test Data --------------------------------

    test_df = pd.read_csv(
        os.path.join(csv_path, "test.csv"),
        nrows=test_nrows,
    )

    test_ids = test_df['sample_id']
    
    x_test = test_df.iloc[:, 1:557]

    x_test.drop(USELESS_FEATURES, axis=1, inplace=True)

    # ---------------------- Sample Submission ---------------------------

    sub_df = pd.read_csv(
        os.path.join(csv_path, "sample_submission.csv"),
        nrows=1,
    )

    # ---------------------- Normalization ------------------------------

    if which_norm == NormType.Z_NORM:
        x_train = z_norm(x_train, norm_path)
        y_train = z_norm(y_train, norm_path)
        x_test = z_norm(x_test, norm_path)

    elif which_norm == NormType.MIN_MAX:
        x_train = min_max_norm(x_train, norm_path)
        y_train = min_max_norm(y_train, norm_path)
        x_test = min_max_norm(x_test, norm_path)

    