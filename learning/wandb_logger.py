from dotenv import load_dotenv
import os
import wandb

load_dotenv()

class WandBLogger:

    def __init__(self, enabled=True, 
                 model=None, 
                 run_name=None):
        self.enabled = enabled

        if self.enabled:
            self.login()
            wandb.init(entity="max-seeli",
                       project="rooted-product-learning")
            if run_name is None:
                wandb.run.name = wandb.run.id    
            else:
                wandb.run.name = run_name  

            if model is not None:
                self.watch(model)         

    def login(self):
        wandb_api_key = os.getenv("WANDB_KEY")
        wandb.login(key=wandb_api_key)
        
    def watch(self, model, log_freq=1):
        wandb.watch(model, log="all", log_freq=log_freq)
            
    def log(self, log_dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)
 
    def finish(self):
        if self.enabled:
            wandb.finish()
