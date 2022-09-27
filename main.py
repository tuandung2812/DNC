import yaml
from supervisor.DncSupervisor import Supervisor
import torch
from utils.training_utils import config_seed

if __name__ == "__main__":
    with open("./config/dnc.yaml", "rb") as input:
        config = yaml.load(input, yaml.Loader)

    config_seed(config['train']['seed'])
    # print(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Currently using ", device)
    supervisor = Supervisor(config, device)
    supervisor.train()
    supervisor.test()
