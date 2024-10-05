import os

from dotenv import load_dotenv

import wandb

load_dotenv()


class WandBLogger:

    def __init__(self, enabled=True,
                 model=None,
                 run_name=None,
                 notes=None):
        self.enabled = enabled
        self.run_name = run_name

        if self.enabled:
            self.login()
            wandb.init(entity="max-seeli",
                       project="neural-graph-gumbo",
                       name=run_name,
                       notes=notes)
            self.run_name = wandb.run.name

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
