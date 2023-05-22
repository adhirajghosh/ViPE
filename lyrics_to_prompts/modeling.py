import torch
import pytorch_lightning as pl
# from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW


class GPT2Convertor(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2Convertor, self).__init__()
        self.save_hyperparameters()

        self.hparams = hparams
        self.model = GPT2LMHeadModel.from_pretrained(self.hparams.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.hparams.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_length=hparams.context_length
        self.batch_size=hparams.batch_size
        self.lr=hparams.learning_rate


        #tokenizer([tokenizer.eos_token+ 'hello' + tokenizer.eos_token],['how are you?' + tokenizer.eos_token],padding=True,return_token_type_ids=True,return_tensor=True)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch['token_type_ids']

        labels = torch.tensor(
            [[-100 if token == self.tokenizer.pad_token_id else token for token in label] for label in
             input_ids]).to(self.hparams.device)

        return self.model(input_ids, attention_mask=attention_mask, labels=labels,token_type_ids=token_type_ids)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).loss
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        return optimizer

        ####################
        # DATA RELATED HOOKS
        ####################

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = Dataset(self.data_dir,  training=True)
        self.valid_dataset = Dataset(self.data_dir,  training=False)
        self.data_collator = ContextAwareDataCollator(self.tokenizer,context_length=self.context_length)

    def train_dataloader(self):

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=16, collate_fn=self.data_collator, prefetch_factor=3)
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=16, collate_fn=self.data_collator, prefetch_factor=3)
        return valid_dataloader


