from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import torch
import os
import trl 
from unsloth import FastLanguageModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("TRL version:", trl.__version__)  # Should show 0.10.1
class ChatModel:
    def __init__(self, model_name="checkpoint-183"):
        self.app = FastAPI()

        # Load base model and PEFT adapter
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        peft_model = PeftModel.from_pretrained(base_model, model_name, is_trainable=True)

        # base_model, self.tokenizer = FastLanguageModel.from_pretrained(
        #     model_name=model_name,
        #     max_seq_length=2048,
        #     dtype=torch.float16,
        #     load_in_4bit=True,
        # )

        # peft_model = FastLanguageModel.get_peft_model(
        #         base_model,
        #         r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        #         target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                         "gate_proj", "up_proj", "down_proj",],
        #         lora_alpha = 16,
        #         lora_dropout = 0, # Supports any, but = 0 is optimized
        #         bias = "none",    # Supports any, but = "none" is optimized
        #         # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        #         use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        #         random_state = 3407,
        #         use_rslora = False,  # We support rank stabilized LoRA
        #         loftq_config = None, # And LoftQ
        #     )

        # Wrap with value head for RL
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model).to("cuda")
        
        # Set up both RL and SFT components
        self.ppo_config = PPOConfig(
            batch_size=1,
            mini_batch_size=1,
            learning_rate=1e-5,
            gradient_accumulation_steps=1,
        )
        self.ppo_trainer = PPOTrainer(
            model=self.model,
            config=self.ppo_config,
            tokenizer=self.tokenizer,
        )
        self.sft_optimizer = torch.optim.AdamW(self.model.pretrained_model.parameters(), lr=1e-5)
        
        self.last_input = None
        self.last_output = None
        self.setup_routes()

    def setup_routes(self):
        @self.app.post("/chat")
        async def chat(conversation: Conversation):
            # Format and generate response
            text = self.tokenizer.apply_chat_template(
                [{"role": msg.role, "content": msg.content} for msg in conversation.messages],
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False, 
            )
            
                    
            # generated_ids = [
            #     output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
            # ]
            
            # Release old tensors from memory
            if hasattr(self, "last_input"):
                del self.last_input
            if hasattr(self, "last_output"):
                del self.last_output
            torch.cuda.empty_cache()  # Force memory cleanup

            # self.last_input = inputs.input_ids[0]
            # self.last_output = torch.unsqueeze(generated_ids[0], 0)[0]

            # response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            
            
            # return {"output": response}
            # Store for potential RL training
            self.last_input = inputs.input_ids[0]
            self.last_output = outputs[0][inputs.input_ids.shape[-1]:]
            
            response = self.tokenizer.decode(self.last_output, skip_special_tokens=True)
            return {"output": response}

        @self.app.post("/train")
        async def train(conversation: Conversation):
            results = {}
            # Supervised Fine-Tuning
            if conversation.target_response is not None:
                full_text = self.tokenizer.apply_chat_template(
                    [{"role": msg.role, "content": msg.content} for msg in conversation.messages],
                    tokenize=False,
                    add_generation_prompt=True
                ) + conversation.target_response
                
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to("cuda")
                
                # Create labels (mask prompt part)
                prompt_length = len(self.tokenizer.encode(full_text.split(conversation.target_response)[0]))
                labels = inputs.input_ids.clone()
                labels[:, :prompt_length] = -100
                
                # SFT Forward pass
                base_model = self.model.pretrained_model
        
                # Forward pass through base model
                outputs = base_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # SFT Backward pass
                self.sft_optimizer.zero_grad()
                loss.backward()
                self.sft_optimizer.step()
                results["sft_loss"] = loss.item()
                return {"status": "SFT updated" }
            # Reinforcement Learning
            if conversation.reward is not None and self.last_input is not None:
                self.ppo_trainer.step(
                    [self.last_input],
                    [self.last_output],
                    [torch.tensor([conversation.reward], dtype=torch.float16, device="cuda")],
                )

                return {"status": "RL updated" }

        @self.app.post("/save_model")
        async def save_model(checkpoint_name: str = "hybrid_checkpoint"):
            os.makedirs(checkpoint_name, exist_ok=True)
            self.model.save_pretrained(checkpoint_name)
            self.tokenizer.save_pretrained(checkpoint_name)
            return {"message": f"Model saved to {checkpoint_name}"}

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message] = None
    reward: Optional[float] = None
    target_response: Optional[str] = None

chat_model = ChatModel()
app = chat_model.app



"run this outside the docker"