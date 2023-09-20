class CheckpointManager:
    def __init__(self, exp_dir, max_best_models, save_optimizer_dict, task):
        self.exp_dir = exp_dir
        self.max_best_models = max_best_models
        self.save_optimizer_dict = save_optimizer_dict
        self.task = task
        assert os.path.isdir(self.exp_dir), f"experiments ({self.exp_dir}) directory does not exist"
        self.manager_path = os.path.join(self.exp_dir, "checkpoint_manager.json")
        if not os.path.isfile(self.manager_path):
            dump_json(self.manager_path, [])

    def save_model(self, iteration, model, optimizer, loss_value):
        # model: tts_model if task == TTS
        #        (gen_model, disc_model) if task == VOC
        # optimizer: tts_optim if task == TTS
        #            (gen_optim, disc_optim) if task == VOC
        # loss_value: lower the loss_value better the model
        #
        # [(checkpoint_iteration, loss_value)] (increasing order)
        manager_data = load_json(self.manager_path)
        add_at_index = len(manager_data)
        for index in range(len(manager_data)):
            if (loss_value < manager_data[index][1]):
                add_at_index = index
                break
        if (add_at_index == self.max_best_models): # if add_at_index is at last position and manager already has max models, then igno
            return
        checkpoint_path = os.path.join(self.exp_dir, f"checkpoint_{iteration}.pt")
        manager_data.insert(add_at_index, (checkpoint_path, loss_value))
        # for saving the torch model
        if self.task == "TTS":
            model_dict = {
                'task': self.task,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict() if self.save_optimizer_dict else None,
            }
        else:
            model_dict = {
                'task': self.task,
                'iteration': iteration,
                'model_state_dict': (model[0].state_dict(), model[1].state_dict()),
                'optim_state_dict': (optimizer[0].state_dict(), optimizer[1].state_dict()) if self.save_optimizer_dict else None,
            }
        torch.save(model_dict, checkpoint_path)
        # removing the last checkpoint in the manager if more than max models
        if (len(manager_data) > self.max_best_models):
            model_removed_path = manager_data[-1][0]
            manager_data = manager_data[: -1]
            os.remove(model_removed_path)
        dump_json(self.manager_path, manager_data)
