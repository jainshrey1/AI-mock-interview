import os
import pandas as pd
from interviewBot.logging import logger
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from functools import partial
from interviewBot.entity import DataTransformationConfig
from transformers import set_seed, pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer
)
import torch

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.data_path= config.data_path
        self.model_name = config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer Name Inside DataTransformation:", self.config.tokenizer_name)
        # Automatically detect GPU or use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def get_bnb_config(self):
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=False, # Load model in 4-bit mode
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        )
        return bnb_config

    def get_original_model(self):
        if torch.cuda.is_available():
            self.bnb_config = self.get_bnb_config()  # Only needed if 4-bit were enabled
        original_model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                      #quantization_config=self.bnb_config, # Keep this line only if using 4-bit
                                                      trust_remote_code=True,
                                                      use_auth_token=True).to(self.device)
        return original_model
    def load_data(self) -> DatasetDict:
        # Load data from CSV file
        df= pd.read_csv(self.data_path)

        train_df = df.sample(frac=0.7, random_state=42)
        remaining_df = df.drop(train_df.index)

        validation_df = remaining_df.sample(frac=0.5, random_state=42)
        test_df = remaining_df.drop(validation_df.index)

        # Converting DataFrames to Hugging Face Dataset format
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(validation_df),
            "test": Dataset.from_pandas(test_df)
        })
        return dataset
    
    def create_prompt_formats(self,sample):
        """
        Format various fields of the sample ('instruction','output')
        Then concatenate them using two newline characters
        :param sample: Sample dictionnary
        """
        INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        INSTRUCTION_KEY = "### Instruct: Answer the below question."
        RESPONSE_KEY = "### Output:"
        END_KEY = "### End"

        blurb = f"\n{INTRO_BLURB}"
        instruction = f"{INSTRUCTION_KEY}"
        input_context = f"{sample['Question']}" if sample["Question"] else None
        response = f"{RESPONSE_KEY}\n{sample['Answer']}"
        end = f"{END_KEY}"

        parts = [part for part in [blurb, instruction, input_context, response, end] if part]

        formatted_prompt = "\n\n".join(parts)
        sample["text"] = formatted_prompt

        return sample
    
    def get_max_length(self,model):
        conf = model.config
        max_length = None
        for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
            max_length = getattr(model.config, length_setting, None)
            if max_length:
                print(f"Found max lenth: {max_length}")
                break
        if not max_length:
            max_length = 1024
            print(f"Using default max length: {max_length}")
        return max_length
    
    def preprocess_batch(self,batch, tokenizer, max_length):
        """
        Tokenizing a batch
        """
        return self.tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )
    
    def preprocess_dataset(self, tokenizer, max_length ,seed, dataset):
        """Format & tokenize it so it is ready for training
        :param tokenizer (AutoTokenizer): Model Tokenizer
        :param max_length (int): Maximum number of tokens to emit from tokenizer
        """

        # Add prompt to each sample
        print("Preprocessing dataset...")
        dataset = dataset.map(self.create_prompt_formats)#, batched=True)

        # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
        _preprocessing_function = partial(self.preprocess_batch, max_length=max_length, tokenizer= tokenizer)
        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=['Skill', '__index_level_0__', 'Question', 'Answer'],
        )

        # Filter out samples that have input_ids exceeding max_length
        dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

        # Shuffle dataset
        dataset = dataset.shuffle(seed=seed)

        return dataset
    
    def convert_dataset(self):
        max_length = self.get_max_length(self.get_original_model())
        print(max_length)
        seed = 42
        set_seed(seed)
        dataset= self.load_data()
        train_dataset = self.preprocess_dataset(self.config.tokenizer_name,max_length,seed, dataset['train'])
        eval_dataset = self.preprocess_dataset(self.config.tokenizer_name,max_length,seed, dataset['validation'])  

        train_dataset.save_to_disk(os.path.join(self.config.root_dir, "interview_dataset/train"))
        eval_dataset.save_to_disk(os.path.join(self.config.root_dir, "interview_dataset/validation"))  
        #return train_dataset, eval_dataset
