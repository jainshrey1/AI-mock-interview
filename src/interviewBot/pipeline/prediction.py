# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer

class PredictionPipeline:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("jshrey01/interview-bot-finetuned-phi-2", use_fast=False)
        self.pipe = pipeline("text-generation", model="jshrey01/interview-bot-finetuned-phi-2", tokenizer=self.tokenizer)
    
    def generate_text(self, prompt):
        output = self.pipe(prompt, max_length=200, num_return_sequences=1, do_sample=True)
        return output[0]['generated_text'] 