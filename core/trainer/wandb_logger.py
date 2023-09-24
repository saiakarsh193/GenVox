import wandb
from typing import List, Tuple, Dict, Any

class WandbLogger:
    def __init__(
        self,
        project_name: str,
        experiment_id: str,
        notes: str,
        model_name: str,
        seed: int,
        epochs: int,
    ):
        wandb.init(
            project=project_name,
            name=experiment_id,
            notes=notes,
            config={
                "model": model_name,
                "seed": seed,
                "epochs": epochs,
            }
        )

    def define_metrics(self, metrics: List[Tuple[str, str]]) -> None:
        for value, summary in metrics:
            wandb.define_metric(value, summary=summary)
    
    def Image(self, img, caption=""):
        return wandb.Image(img, caption=caption)

    def log(self, values: Dict[str, Any], iteration: int, commit: bool = False):
        wandb.log(values, step=iteration, commit=commit)

    def finish(self):
        wandb.finish()
