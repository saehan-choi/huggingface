from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer
from transformers import FillMaskPipeline


model = TFBertForMaskedLM.from_pretrained('bert-large-uncased')
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

inputs = tokenizer('Soccer is a really fun [MASK].', return_tensors='tf')

pip = FillMaskPipeline(model=model, tokenizer=tokenizer)

print(pip('moly is a really fun [MASK].'))

