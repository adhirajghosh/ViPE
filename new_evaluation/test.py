# import torch
# from transformers import BartForConditionalGeneration, BartTokenizer
#
sentence = " my room is a pigsty" \
#                  "that bm has on our lives and the fact that is has turned my so into a bitter angry" \
#                  " person who is not always particularly kind to the people around him when he is feeling stressed"
#
# model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
# model = model.to(device)
# tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
# batch = tokenizer(input_sentence, return_tensors='pt')
# generated_ids = model.generate(batch['input_ids'])
# generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#
# print(generated_sentence)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')
#sentence = "This is something which i cannot understand at all"

def gen(sentence, t):
    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")



    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        early_stopping=True,
        num_return_sequences=5,
        temperature=t
    )

    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print(line)

d=3
