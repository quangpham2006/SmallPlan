from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
import torch
import os
import logging
import trl
from dotenv import load_dotenv

load_dotenv()
logger= logging.getLogger(__name__)

logger.info("TRL version:", trl.__version__)  # Should show 0.10.1

class SLMTrainer:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        peft_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(model_name), 
                                               model_name, 
                                               is_trainable=True)
        # Wrap with value head for RL
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(peft_model).to("cuda")
        
        # Set up both RL and SFT components
        logger.info("Init SFT Trainer")
        self.sft_trainer = torch.optim.AdamW(self.model.pretrained_model.parameters(), lr=1e-5)

        logger.info("Init PPO Trainer")
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
        
        self.last_input = None
        self.last_output = None

    def chat(self, conversation):
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
        torch.cuda.empty_cache()
        self.last_input = inputs.input_ids[0]
        self.last_output = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(self.last_output, skip_special_tokens=True)
        return response

    def train(self, conversation):
        results = {}
        # Supervised Fine-Tuning
        if conversation.target_response is not None:
            logger.info(f"Running SFT step")
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
            
            prompt_length = len(self.tokenizer.encode(full_text.split(conversation.target_response)[0]))
            labels = inputs.input_ids.clone()
            labels[:, :prompt_length] = -100
            
            base_model = self.model.pretrained_model
            outputs = base_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            loss = outputs.loss
            self.sft_trainer.zero_grad()
            loss.backward()
            self.sft_trainer.step()
            # sft_loss = loss.item()
            results_message = "SFT updated"
        
        # Reinforcement Learning
        if conversation.reward is not None and self.last_input is not None:
            logger.info(f"Running PPO step")
            self.ppo_trainer.step(
                [self.last_input],
                [self.last_output],
                [torch.tensor([conversation.reward], dtype=torch.float16, device="cuda")],
            )
            results_message = "RL updated"
        return results_message

    def save_model(self, checkpoint_name="hybrid_checkpoint"):
        os.makedirs(checkpoint_name, exist_ok=True)
        self.model.save_pretrained(checkpoint_name)
        self.tokenizer.save_pretrained(checkpoint_name)
        logger.info(f"Model saved to {checkpoint_name}")