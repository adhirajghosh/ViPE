import evaluate
sentence1 = "the cat sat on the mat"
sentence2 = "the cat ate the mat"
google_bleu = evaluate.load("google_bleu")
result = google_bleu.compute(predictions=[sentence1,'wpw'], references=[sentence2,sentence1])
print(result)

# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# bleurt = load("bleurt", module_type="metric")
# results = bleurt.compute(predictions=predictions, references=references)