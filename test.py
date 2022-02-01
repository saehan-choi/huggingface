


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")




text = r"""
We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from
Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from
unlabeled text by jointly conditioning on both
left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer
to create state-of-the-art models for a wide
range of tasks, such as question answering and
language inference, without substantial taskspecific architecture modifications.
BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results on eleven natural language processing
tasks, including pushing the GLUE score to
80.5% (7.7% point absolute improvement),
MultiNLI accuracy to 86.7% (4.6% absolute
improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1
(5.1 point absolute improvement).
1 Introduction
Language model pre-training has been shown to
be effective for improving many natural language
processing tasks (Dai and Le, 2015; Peters et al.,
2018a; Radford et al., 2018; Howard and Ruder,
"""

questions = [
    r"""how many did you experiments?
    """,
    r"""what is the bert?
    """,
    r"""what is the abbreviation of bert?
    """
]

print(tokenizer(text,return_tensors="pt")['input_ids'])
print(tokenizer(text,return_tensors="pt")['input_ids'].shape)


for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )

    print(f"Question: {question}")
    print(f"Answer: {answer}\n")



# test해본결과 input embedding 수인 512를 넘어가면 안됨,, 글자수가 제한적