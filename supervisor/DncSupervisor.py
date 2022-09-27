import os
import torch
from torch import Tensor
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau


from dataloader.DncLoader import Dataloader
from model.dnc import Model
from Model.BaseModel.BaseSupervisor import BaseSupervisor


class Supervisor(BaseSupervisor):
    def __init__(self, config, device):
        super().__init__()
        
        self.device = device

        # dataloader settings
        self.dataloader = Dataloader(config)
        self.number_source = self.dataloader.get_number_sources()
        self.number_target = self.dataloader.get_number_targets()

        # Model settings
        self.input_size = config['model']['input_size']
        self.hidden_size = config['model']['hidden_size']
        self.rnn_type = config['model']['rnn_type']
        self.dropout = config['model']['dropout']
        self.memory_type = config['model']['memory_type']
        self.num_layers = config['model']['num_layers']
        self.num_hidden_layers = config['model']['num_hidden_layers']
        self.read_heads = config['model']['read_heads']
        self.mem_size = config['model']['mem_size']
        self.mem_slot = config['model']['mem_slot']

        # Train settings
        self.epochs = config['train']['epochs']
        self.lr = config['train']['lr']
        self.lr_decay = config['train']['lr_decay']
        self.weight_decay = config['train']['weight_decay']
        self.optim = config['train']['optim']
        self.grad_clip = config['train']['clip']
        self.patience = config['train']['patience']
        self.checkpoint = config['train']['checkpoint']
        self.seed = config['train']['seed']

        # result settings
        self.result_dir = config['result']['result_dir']
        self.metrics = config['result']['metrics']

        self.model = Model(
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            dim_hidden=self.dim_hidden,
            memory_feature=self.memory_feature,
            device=self.device
        )
        self.model = self.model.to(self.device)

        # Training setting
        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=self.lr_decay,
            patience=10,
        )

        self.criterion = torch.nn.MSELoss()

    def train(self):
        print("{action:-^50}".format(action="Training"))

        # Memory writing in location
        for epoch in tqdm(range(self.epochs)):
            
            for source_num in range(self.number_source):
                loader = self.dataloader.loader_sources[source_num]
                dataset = self.dataloader.dataset_sources[source_num]

                for x, y in loader:
                    self.optimizer.zero_grad()

                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.model.mem_to_device()

                    pred = self.model(x)
                    pred = pred.to(self.device)

                    loss = self.criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        # Save model
        model_path = os.path.join(self.model_path, f'{self.save_name}.pkl')
        self.model.save_model(model_path)

    def test(self):
        print("{action:-^50}".format(action="Testing"))
        eval_results = {}
        for t in range(self.number_target):
            test_data, name = self.dataloader.gen_test_data(t)
            scaler = test_data['y-scaler']
            x = test_data['X']
            y = test_data['Y']
            x = Tensor(x).to(self.device)
            pred = self.model(x)

            pred = pred.cpu().detach().numpy()
            gt = y

            # inverse transform and flatten
            pred = scaler.inverse_transform(pred).flatten()
            gt = scaler.inverse_transform(gt).flatten()

            eval_result = self.compute_metrics(y_true=gt,
                                               y_pred=pred,
                                               metrics=self.metrics)

            eval_results[name] = eval_result

        df = pd.DataFrame.from_dict(eval_results, orient='index')
        df_path = os.path.join(self.result_path, f'{self.save_name}.csv')
        df.to_csv(df_path)
        print(df)
