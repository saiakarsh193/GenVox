import os
import torch
from typing import Union, List, Dict, Any

from models import BaseModel
from utils import dump_json, load_json

class CheckpointManager:
    def __init__(
            self, 
            exp_dir: str = "exp",
            max_best_models: int = 3,
            save_optimizer_dict: bool = False,
        ):
        self.exp_dir = exp_dir
        self.max_best_models = max_best_models
        self.save_optimizer_dict = save_optimizer_dict
        self.manager_path = os.path.join(self.exp_dir, "checkpoint_manager.json")
        if not os.path.isfile(self.manager_path):
            dump_json(self.manager_path, [])
        # [
        #     {
        #         "priority_value": <how_important_is_this_model>,
        #         "model_path": <path_to_torch_pt_model>
        #     }
        # ]

    def save_model(
            self,
            iteration: int,
            model: BaseModel,
            optimizer: torch.optim.Optimizer,
            priority_value: Union[int, float]
        ) -> None:
        # load existing manager data and figure out where to add the current model
        manager_data: List[Dict[str, Any]] = load_json(self.manager_path)
        add_at_index = len(manager_data) # default -> end of list
        for ind, checkpoint in enumerate(manager_data):
            if (priority_value < checkpoint["priority_value"]):
                add_at_index = ind
                break
        # if add_at_index is at last position and that position is max value, then skip saving
        if (add_at_index == self.max_best_models):
            return
        # get model data and save it
        model_dict = model.get_checkpoint_statedicts(optimizer=(optimizer if self.save_optimizer_dict else None))
        model_dict["iteration"] = iteration
        model_path = os.path.join(self.exp_dir, f"checkpoint_{iteration}.pt")
        torch.save(model_dict, model_path)
        manager_data.insert(add_at_index, {
            "priority_value": priority_value,
            "model_path": model_path
        })
        # removing the last checkpoint in the manager if more than max value
        if (len(manager_data) > self.max_best_models):
            removed_checkpoint = manager_data.pop()
            os.remove(removed_checkpoint["model_path"])
        # update the manager data
        dump_json(self.manager_path, manager_data)
