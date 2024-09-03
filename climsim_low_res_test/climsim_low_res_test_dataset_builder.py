"""climsim_low_res_test dataset."""

import tensorflow_datasets as tfds
import pandas as pd
import tensorflow as tf

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


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for climsim_low_res_test dataset."""

    VERSION = tfds.core.Version("6.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "Don't remove columns, processed like high-res",
        "3.0.0": "No processing",
        "4.0.0": "2.0.0 but with the new test set",
        "5.0.0": "3.0.0 but with the new test set",
        "6.0.0": "5.0.0 but float64",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        if self.VERSION == "1.0.0":
            N_IN = 490
        elif self.VERSION in ["2.0.0", "3.0.0", "4.0.0", "5.0.0", "6.0.0"]:
            N_IN = 556
            
        dtype = tf.float64 if self.VERSION == "6.0.0" else tf.float32

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "inputs": tfds.features.Tensor(shape=(N_IN,), dtype=dtype),
                    "sample_id": tfds.features.Text(),
                }
            ),
            homepage="https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        test_path = "../test.csv"

        return {
            "test": self._generate_examples(test_path),
        }

    def _generate_examples(self, test_path):
        """Yields examples."""
        train_stats_path = (
            "../train_stats.csv" if self.VERSION == "1.0.0" else "../climsim/climsim_high_res_train/train_stats.csv"
        )
        train_stats_df = pd.read_csv(train_stats_path, index_col=0)

        for test_chunk in pd.read_csv(test_path, chunksize=100000):
            N_IN = 556
            if self.VERSION == "1.0.0":
                test_chunk.drop(USELESS_FEATURES, axis=1, inplace=True)
                N_IN = 490

            sample_ids = test_chunk["sample_id"]
            x_test = test_chunk.iloc[:, 1 : N_IN + 1]

            if self.VERSION == "1.0.0":
                for col in x_test.columns.to_list():
                    x_test[col] = (
                        x_test[col] - train_stats_df[col]["mean"]
                    ) / train_stats_df[col]["std"]

            if self.VERSION in ["2.0.0", "4.0.0"]:
                for col in x_test.columns.to_list():
                    std = train_stats_df[col]["std"]
                    if std != 0:
                        x_test[col] = (x_test[col] - train_stats_df[col]["mean"]) / std

            dtype = "float64" if self.VERSION == "6.0.0" else "float32"
            
            for sample_id, x in zip(sample_ids, x_test.values):
                yield sample_id, {
                    "inputs": x.astype(dtype),
                    "sample_id": sample_id,
                }
