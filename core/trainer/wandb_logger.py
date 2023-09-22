import wandb

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

    def define_metric(self, value, summary):
        wandb.define_metric(value, summary=summary)
    
    def Image(self, img, caption=""):
        return wandb.Image(img, caption=caption)

    def log(self, values, epoch, commit=False):
        wandb.log(values, step=epoch, commit=commit)

    def finish(self):
        wandb.finish()