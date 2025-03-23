import transformers
from transformers import TrainingArguments, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from interviewBot.entity import ModelTrainerConfig
import torch
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        # Load the model before passing it to count_parameters
        self.original_model_name = AutoModelForCausalLM.from_pretrained(config.model_ckpt)

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    

    def model_quantization(self, original_model):
        lora_config = LoraConfig(
        r=32, #Rank
        lora_alpha=32,
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM")
        # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
        original_model.gradient_checkpointing_enable()

        #Before applying LoRA, ensure the model is prepped for quantization-aware training:
        peft_model = prepare_model_for_kbit_training(original_model)  # Add this step
        peft_model = get_peft_model(original_model, lora_config)

        for param in peft_model.parameters():
            if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                param.requires_grad = True

        return peft_model
    
    # Function to count total and trainable parameters
    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        percent_trainable = (trainable_params / total_params) * 100
        return total_params, trainable_params, percent_trainable

    def load_dataset(self):
        #loading data 
        train_dataset = load_from_disk(os.path.join(self.config.data_path, "train"))
        eval_dataset = load_from_disk(os.path.join(self.config.data_path, "validation"))
        return train_dataset, eval_dataset
    def count_parameters(self, original_model):
        total_params = sum(p.numel() for p in original_model.parameters())
        trainable_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
        percent_trainable = (trainable_params / total_params) * 100
        return total_params, trainable_params, percent_trainable

    def model_train(self):
        peft_training_args = TrainingArguments(
        output_dir = self.config.output_dir,
        num_train_epochs = self.config.num_train_epochs,
        max_steps = self.config.max_steps,
        learning_rate = self.config.learning_rate,
        optim = self.config.optim,
        warmup_steps = self.config.warmup_steps,
        per_device_train_batch_size = self.config.per_device_train_batch_size,
        weight_decay = self.config.weight_decay,
        logging_steps = self.config.logging_steps,
        logging_dir = self.config.logging_dir,
        save_strategy = self.config.save_strategy,
        save_steps = self.config.save_steps,
        evaluation_strategy = self.config.evaluation_strategy,
        eval_steps = self.config.eval_steps,
        do_eval = self.config.do_eval,
        report_to = self.config.report_to,
        overwrite_output_dir = self.config.overwrite_output_dir,
        group_by_length = self.config.group_by_length,
        gradient_checkpointing = self.config.gradient_checkpointing,
        gradient_accumulation_steps = self.config.gradient_accumulation_steps
        )

        
        ## Counting trainable and non trainable parameters
        total_params_before, trainable_params_before, percent_trainable_before= self.count_parameters(self.original_model_name)

        print(f"Before LoRA:")
        print(f"Total Parameters: {total_params_before:,}")
        print(f"Trainable Parameters: {trainable_params_before:,}")
        print(f"Percentage of Trainable Parameters: {percent_trainable_before:.4f}%\n")



        
        peft_model = self.model_quantization(self.original_model_name)
        peft_model.config.use_cache = False

        # After applying LoRA
        total_params_after, trainable_params_after, percent_trainable_after = self.count_parameters(peft_model)
        print(f"After LoRA:")
        print(f"Total Parameters: {total_params_after:,}")
        print(f"Trainable Parameters: {trainable_params_after:,}")
        print(f"Percentage of Trainable Parameters: {percent_trainable_after:.4f}%")

        
        
        ## load training & validation data
        train_dataset, eval_dataset = self.load_dataset()

        peft_trainer = transformers.Trainer(
        model=peft_model,
        train_dataset= eval_dataset,
        eval_dataset= eval_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        peft_trainer.train()