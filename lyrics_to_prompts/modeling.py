import torch
import pytorch_lightning as pl
from utils import Dataset,ContextAwareDataCollator
from transformers import GPT2LMHeadModel, AdamW, GPT2Tokenizer,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW


class GPT2Convertor(pl.LightningModule):
    def __init__(self, hparams):
        super(GPT2Convertor, self).__init__()

        self.params=hparams
        self.data_dir=hparams.data_dir
        model_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/gpt2_v1.0/pretrained_gpt2/'
        # self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.context_length=hparams.context_length
        self.batch_size=hparams.batch_size
        self.learning_rate=hparams.learning_rate
        self.adam_epsilon = 1e-8
        self.warmup_steps= 1e100
        self.weight_decay=0

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch['token_type_ids']

        labels = torch.tensor(
            [[-100 if token_type_ids[j][i] == 0.0 else token for i, token in enumerate(label)] for j, label in
             enumerate(input_ids)]).to(self.params.device)

        return self.model(input_ids, attention_mask=attention_mask, labels=labels,token_type_ids=token_type_ids)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch).loss
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        #model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)
        #optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     # Total number of training steps is [number of batches] x [number of epochs].
    #     # (Note that this is not the same as the number of training samples).
    #     total_steps = len(self.train_dataloader) * epochs
    #
    #     # Create the learning rate scheduler.
    #     # This changes the learning rate as the training loop progresses
    #     scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=warmup_steps,
    #                                                 num_training_steps=total_steps)
    #     optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
    #     return optimizer

        ####################
        # DATA RELATED HOOKS
        ####################

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = Dataset(self.data_dir,context_size=self.context_length, training=True)
        self.valid_dataset = Dataset(self.data_dir,context_size=self.context_length,  training=False)
        self.data_collator = ContextAwareDataCollator(self.tokenizer)

    def train_dataloader(self):

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=8, collate_fn=self.data_collator, prefetch_factor=3)
        return train_dataloader

    def val_dataloader(self):
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=8, collate_fn=self.data_collator, prefetch_factor=3)
        return valid_dataloader
