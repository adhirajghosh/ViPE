from pytorch_lightning import LightningModule
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Convertor(LightningModule):
    def __init__(self, hparams):
        super(GPT2Convertor, self).__init__()

        self.params=hparams
        self.data_dir=hparams.data_dir
        #model path if we wanna use heracleum since it does not have internet connection
        #model_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/gpt2_v1.0/pretrained_gpt2/'
        #model_path='/graphics/scratch2/staff/Hassan/checkpoints/lyrics_to_prompts/gpt2-medium_v1.0/pretrained_gpt2/'
        self.model = GPT2LMHeadModel.from_pretrained(hparams.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(hparams.model_name,padding_side="left")
        # self.model = GPT2LMHeadModel.from_pretrained(model_path)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(model_path,padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.context_length=hparams.context_length
        self.batch_size=hparams.batch_size
        self.learning_rate=hparams.learning_rate
        self.adam_epsilon = 1e-8
        self.warmup_steps= hparams.warmup_steps
        self.weight_decay=0

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch['token_type_ids']

        labels = input_ids.clone()
        labels[token_type_ids == 0] = -100

        # labels = torch.tensor(
        #     [[-100 if token_type_ids[j][i] == 0.0 else token for i, token in enumerate(label)] for j, label in
        #      enumerate(input_ids)]).to(self.params.device)

        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        #print("Validation step is being executed")
        loss = self(batch).loss
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

