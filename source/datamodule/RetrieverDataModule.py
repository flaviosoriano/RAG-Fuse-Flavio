import logging
import pickle

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.dataset.LabelDataset import LabelDataset
from source.dataset.RetrieverFitDataset import RetrieverFitDataset
from source.dataset.TextDataset import TextDataset


class RetrieverDataModule(pl.LightningDataModule):
    def __init__(self, params, tokenizer, fold_idx):
        super(RetrieverDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold_idx = fold_idx
        logging.basicConfig(level=logging.INFO)

    def prepare_data(self):
        with open(f"{self.params.dir}samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

        self.pseudo_labels = None
        self.labels_descriptions = None

        if self.params.label_enhancement == "LLM":
            with open(f"resource/dataset/{self.params.name}/fold_{self.fold_idx}/labels_descriptions.pkl",
                      "rb") as labels_desc_file:
                self.labels_descriptions = pickle.load(labels_desc_file)
                logging.info(f"Loaded {len(self.labels_descriptions)} labels descriptions")



        elif self.params.label_enhancement == "PMI":
            logging.info(f"Using {self.params.dir}fold_{self.fold_idx}/{self.params.pseudo_labels}.pkl")
            with open(f"{self.params.dir}fold_{self.fold_idx}/{self.params.pseudo_labels}.pkl",
                      "rb") as pseudo_labels_file:
                self.pseudo_labels = pickle.load(pseudo_labels_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = RetrieverFitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/train.pkl",
                labels_descriptions=self.labels_descriptions,
                pseudo_labels=self.pseudo_labels,
                text_features_source=self.params.text_features_source,
                label_enhancement=self.params.label_enhancement,
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

            self.val_dataset = RetrieverFitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold_idx}/val.pkl",
                labels_descriptions=self.labels_descriptions,
                pseudo_labels=self.pseudo_labels,
                text_features_source=self.params.text_features_source,
                label_enhancement=self.params.label_enhancement,
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

        if stage == "predict":
            self.text_dataset = TextDataset(
                samples=self.samples,
                text_features_source=self.params.text_features_source,
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length
            )
            self.label_dataset = LabelDataset(
                samples=self.samples,
                labels_descriptions=self.labels_descriptions,
                pseudo_labels=self.pseudo_labels,
                label_enhancement=self.params.label_enhancement,
                tokenizer=self.tokenizer,
                label_max_length=self.params.label_max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return [
            DataLoader(self.text_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
            DataLoader(self.label_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
        ]
