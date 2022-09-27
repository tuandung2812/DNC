import os
import torch
from torch import Tensor
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable as var


from dataloader.DncLoader import Dataloader
from model.dnc import DNC
from supervisor.BaseSupervisor import BaseSupervisor
from utils.metrics import RMSE, R2, MAE, MAPE
from utils.training_utils import EarlyStopping
import os
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
        self.output_length = config['data']['future']
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
        self.lr = float(config['train']['lr'])
        self.lr_decay = config['train']['lr_decay']
        self.weight_decay = float(config['train']['weight_decay'])
        self.optim = config['train']['optim']
        self.grad_clip = config['train']['clip']
        self.patience = config['train']['patience']
        self.checkpoint = config['train']['checkpoint']
        self.seed = config['train']['seed']

        # result settings
        self.result_dir = config['result']['result_dir']
        self.metrics = config['result']['metrics']

        self.model =  DNC(
                input_size=self.input_size,
                output_length = self.output_length,
                hidden_size=self.hidden_size,
                rnn_type=self.rnn_type,
                num_layers=self.num_layers,
                num_hidden_layers=self.num_hidden_layers,
                dropout=self.dropout,
                nr_cells=self.mem_slot,
                cell_size=self.mem_size,
                read_heads=self.read_heads,
                device=self.device,
                debug=False,
                batch_first=True,
                independent_linears=True
            )
        # self.model = self.model.to(self.device)


    def train(self):
        print("{action:-^50}".format(action="Training"))

        # Training setting
        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=self.lr_decay,
            patience=10,
        )

        self.criterion = torch.nn.MSELoss()

        self.es =  EarlyStopping(
                patience=self.patience,
                verbose=True,
                delta=0.0,
                path=self.checkpoint,
            )

        (chx, mhx, rv) = (None, None, None)

        # Train on source locations
        for epoch in range(self.epochs):
            train_loss = 0
            train_r2_loss = 0
            for source_num in tqdm(range(self.number_source)):
                loader = self.dataloader.loader_sources[source_num]
                dataset = self.dataloader.dataset_sources[source_num]

                for idx, (x, y) in enumerate(loader):
                    x = x.to(self.device)
                    x = x.unsqueeze(-1)
                    y = y.to(self.device)
                    pred, (chx, mhx, rv) = self.model(x, (None, None, None), reset_experience=True, pass_through_memory=True)
                    # print(pred.shape)
                    pred = pred.to(self.device)
                    # print(pred.shape)
                    # mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }
                    # print(mhx.shape)
                    loss = self.criterion(pred, y)                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += loss.item()
                    gt = y.cpu().detach().numpy()
                    pred = pred.cpu().detach().numpy()
                    r2_loss = R2(gt, pred)
                    train_r2_loss += r2_loss
            
            print("Epoch: %d, train_loss: %1.5f" % (epoch, train_loss))
            print("Epoch: %d, train_r2_loss: %1.5f" % (epoch, train_r2_loss))

                
            # Validation on target locations. RMSE as the main loss
            validation_loss = 0
            validation_r2_loss = 0
            for t in range(self.number_target):
                val_data = self.dataloader.gen_validation_data(t)
                scaler = val_data['y-scaler']
                x = val_data['X']
                y = val_data['Y']
                x = Tensor(x).to(self.device)
                x = x.unsqueeze(-1)
                pred, (chx, mhx, rv) = self.model(x, (None,None, None), reset_experience=True, pass_through_memory=True)
                pred = pred.to(self.device)
                pred = pred.cpu().detach().numpy()
                gt = y

                # inverse transform and flatten
                pred = scaler.inverse_transform(pred).flatten()
                gt = scaler.inverse_transform(gt).flatten()
                loss = RMSE(gt, pred)
                r2_loss = R2(gt,pred)
                validation_loss += loss
                validation_r2_loss += r2_loss
            
            validation_loss /= self.number_target
            validation_r2_loss /= self.number_target
            self.scheduler.step(validation_loss)
            self.es(validation_loss, self.model)



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
