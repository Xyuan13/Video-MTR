import bert_score

from rouge_score import rouge_scorer

def compute_rouge_score(reference, hypothesis, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    scores = scorer.score(reference, hypothesis)
    average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
    return average_fmeasure

def compute_bert_score(reference, hypothesis, use_stemmer=True):
    P, R, F1 = bert_score.score(reference, hypothesis, lang="en", 
									verbose=False,
									model_type='bert-large-uncased') 
    return F1.item() if len(F1) == 1 else F1[0].item()

# TEST CASES LIST FOR BERT SCORE

TEST_CASES = [
    ("I have one apple.", "I have 1 apple."),
    ("Two times", "Twice."),
    ("Moderately paced", "Normal speed"),
]

BERT_MODELS=["‌bert-base-uncased",'bert-large-uncased' ]


for pred, ref in TEST_CASES:
    print(f"[Debug] pred: {pred}, ref: {ref}")

    rouge_score = compute_rouge_score(pred, ref)

    # 在调用bert_score.score之前添加检查和转换
    if isinstance(pred, str):
        pred = [pred]
    if isinstance(ref, str):
        ref = [ref]
    assert len(pred) == len(ref), f"Length mismatch: pred={len(pred)}, ref={len(ref)}"

    bert_score_result = compute_bert_score(pred, ref)
    print(f"[Debug] bert_score: {bert_score_result}, rouge_score: {rouge_score}")


#tensor([0.8176]) tensor([0.8176]) tensor([0.8176])