import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    # helper printing function that can be used by subclasses
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    # helper saving function that can be used by subclasses
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, load_path, network, strict=True,fastai_chkp=False):
        if isinstance(network, nn.DataParallel):
            network = network.module
        if self.opt['path']['fastai_chkp']=="true" or fastai_chkp:
            print("loading from fastai") 
            if self.opt['gpu_ids'] is not None:
                network.load_state_dict(torch.load(load_path)['model'], strict=strict)
            else:
                network.load_state_dict(torch.load(load_path,map_location='cpu')['model'], strict=strict)
        else:
            if self.opt['gpu_ids'] is not None:
                network.load_state_dict(torch.load(load_path), strict=strict)
            else:
                network.load_state_dict(torch.load(load_path,map_location='cpu'), strict=strict)
