from binary_classifier_using_gpt2.prepare_training_data  import SpamDataset
from torch.utils.data import DataLoader
import tiktoken
import torch


class Executer:

    def __init__(self, path_to_data=None):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        self.folder_path =  path_to_data



    def execute_datasets(self):
        self.train_dataset = SpamDataset(
            csv_file = f"{self.folder_path}/train.csv",
            max_length=None,
            tokenizer=self.tokenizer
            )
        
        self.val_dataset = SpamDataset(
            csv_file= f"{self.folder_path}/validation.csv",
            max_length=self.train_dataset.max_length,
            tokenizer=self.tokenizer
        )
        self.test_dataset = SpamDataset(
            csv_file= f"{self.folder_path}/test.csv",
            max_length=self.train_dataset.max_length,
            tokenizer=self.tokenizer
        )

        

    def execute_data_loaders(self):

        num_workers = 0
        batch_size = 8

        torch.manual_seed(123)

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )


    def execute(self):
        self.execute_datasets()
        self.execute_data_loaders()
        return self.train_loader, self.val_loader, self.test_loader
    





        

        