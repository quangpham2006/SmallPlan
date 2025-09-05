import unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
#     "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
#     "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
#     "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
#     "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
#     "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
#     "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
#     "unsloth/Phi-3-medium-4k-instruct",
#     "unsloth/gemma-2-9b-bnb-4bit",
#     "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

#     "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
#     "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
#     "unsloth/Llama-3.2-3B-bnb-4bit",
#     "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

#     "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
# ] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def formatting_prompts_func(examples):
    #print(examples)
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Load dataset from JSON files (train and test)
dataset = load_dataset("json", data_files={"train": "/workspace/khointn/GPT-4o-data/data_cleaned.jsonl", "test": "/workspace/khointn/GPT-4o-data/data_cleaned.jsonl"})

# Map the formatting function over both splits.
# This creates a new "text" field in each example using the chat template.
dataset = dataset.map(formatting_prompts_func, batched=True)

# Import required modules for training
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# Create the SFTTrainer using both train and evaluation datasets
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],  # Load the test split as evaluation data
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs = 3,  # Set this for 1 full training run.
        #max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this for WandB, etc.
    ),
)

trainer_stats = trainer.train()