"""climsim_lr_train_grid dataset."""

import os
import numpy as np
import tensorflow_datasets as tfds
import tfds_utils


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for climsim_lr_train_grid dataset."""

    VERSION = tfds.core.Version("2.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
        "2.0.0": "grid-pos baked in",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""

        if self.VERSION == "1.0.0":
            x_shape = (556,)
            y_shape = (368,)
            grid_pos_feature = tfds.features.Scalar(dtype=np.int32)
        elif self.VERSION == "2.0.0":
            x_shape = (384, 556)
            y_shape = (384, 368)
            grid_pos_feature = tfds.features.Tensor(shape=(384,), dtype=np.int32)

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "sample_id": tfds.features.Text(),
                    "grid_pos": grid_pos_feature,
                    "inputs": tfds.features.Tensor(shape=x_shape, dtype=np.float32),
                    "targets": tfds.features.Tensor(shape=y_shape, dtype=np.float32),
                }
            ),
            supervised_keys=("inputs", "targets"),
            homepage="https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        """Yields examples."""

        all_file_paths = np.load(
            "../climsim/all_files_in_low_res.npy", allow_pickle=True
        )
        grid_file_name = "ClimSim_low-res_grid-info.nc"

        # n_splits decides how many files to load into ram at once
        time_stride = 7 if self.VERSION in ["1.0.0", "2.0.0"] else 1
        n_splits = 32 if self.VERSION in ["1.0.0", "2.0.0"] else 512
        splits = tfds_utils.get_splits(
            all_file_paths, n_splits=n_splits, time_stride=time_stride
        )

        delete_path = "/home/joylunkad/.cache/huggingface/hub/datasets--LEAP--ClimSim_low-res"
        for train_files in splits:

            if os.path.exists(delete_path):
                print(os.listdir(delete_path))
            
            train_files.extend([f.replace("mli", "mlo") for f in train_files])
            train_files.append(grid_file_name)

            _, data_path = tfds_utils.get_dataset_from_file_names(
                train_files, repo_id="LEAP/ClimSim_low-res"
            )

            sample_ids, X, Y, grid_ids = tfds_utils.extract_data_from_file_paths(
                data_path,
                grid_file_name,
                keep_grid=self.VERSION == "2.0.0",
            )

            print(f"Extracted data from {len(train_files)} files")
            print(X.shape, Y.shape)

            for sample_id, x, y, grid_pos in zip(sample_ids, X, Y, grid_ids):
                yield sample_id, {
                    "sample_id": sample_id,
                    "inputs": x.astype(np.float32),
                    "targets": y.astype(np.float32),
                    "grid_pos": grid_pos.astype(np.int32),
                }
