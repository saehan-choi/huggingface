from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
Currently, the deep learning image recognition rate is being developed day by day by researchers around the world. Among them, Convolution Neural Network (CNN) is one of the solutions that stands out in the image recognition method. Intrigued by this method, I built a layer using pytorch, a deep learning framework, learned the images I took myself, classifies the images, and started working on it after thinking of making a drone. There are a total of 8 images, and after presenting a traffic sign, it is designed to recognize the sign through a webcam, count the number of times the image is recognized, and perform an appropriate action when a specific count is reached. ex) U-turn, circular turn, left, right turn, landing
"""

questions = [
    r"""how many have images in this experiments?
    """,
    r"""where did you feel interest?
    """
]

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
    print(f"Answer: {answer}")