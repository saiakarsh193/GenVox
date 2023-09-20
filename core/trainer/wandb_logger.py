class WandbLogger:
    def __init__(self, trainer):
        wandb.init(
            project=trainer.config.project_name,
            name=trainer.config.experiment_id,
            notes=trainer.config.notes,
            config={
                "architecture": trainer.model_config.model_name,
                "task": trainer.model_config.task,
                "epochs": trainer.config.epochs,
                "batch_size": trainer.config.batch_size,
                "seed": trainer.config.seed,
                "device": trainer.device
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