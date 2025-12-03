from binary_classifier_using_gpt2.prepare_training_data  import SpamDataset
from torch.utils.data import DataLoader
import tiktoken
import torch


class Executer:

    def __init__(self, train_path=None, test_path=None, val_path=None):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.train_loader, self.val_loader, self.test_loader = None, None, None
        #self.folder_path =  path_to_data
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path



    def execute_datasets(self):
        self.train_dataset = SpamDataset(
            csv_file = self.train_path,
            max_length=None,
            tokenizer=self.tokenizer
            )
        
        self.val_dataset = SpamDataset(
            csv_file= self.val_path,
            max_length=self.train_dataset.max_length,
            tokenizer=self.tokenizer
        )
        self.test_dataset = SpamDataset(
            csv_file= self.test_path,
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
    





        

        