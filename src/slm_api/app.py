import sys
import logging
import yaml
from fastapi import FastAPI
from argparse import ArgumentParser

from dotenv import load_dotenv
sys.path.append('/workspace/khointn/SmallPlan')

load_dotenv()
logger= logging.getLogger(__name__)

from src.slm_api.setup_trainer import SLMTrainer
from src.slm_api.schema import Message, Conversation

parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True, help="Mode: 'train' or 'eval'")
mode = str(parser.parse_args().mode)

with open('configs/slm_training.yaml', 'r') as file:
            cfg = yaml.safe_load(file)

if mode == "train":
    model_path = f"{cfg['preadapted_outputs_path']}/{cfg['slm_api_model']}"
elif mode == "eval":
    model_path = f"{cfg['smallplan_outputs_path']}/{cfg['strategy']}-2e-{cfg['slm_api_model']}/{cfg['last_train_scene']}"
else:
     raise ValueError("Mode must be either 'train' or 'eval'")

print(f"Using model: {cfg['slm_api_model']}")
print(f"Model path: {model_path}")
app = FastAPI()
smallplan_trainer = SLMTrainer(model_name=model_path, mode=mode)

'''
The simulation and the training runs in 2 different environments (igibson vs smallplan).
There is some environment conflicts between the two. Therefore, for the training environment,
we expose an API that can be called from the simulation environment.
This help to prevent the environment conflicts during running.
'''

@app.post("/chat")
async def chat(conversation: Conversation):
    # Format and generate response
    response = smallplan_trainer.chat(conversation)
    return {"output": response}

@app.post("/train")
async def train(conversation: Conversation):
    response = smallplan_trainer.train(conversation)
    return {"status": response}

@app.post("/save_model")
async def save_model(checkpoint_name: str = "hybrid_checkpoint"):
    smallplan_trainer.save_model(checkpoint_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg["slm_api_host"], port=cfg["slm_api_port"])