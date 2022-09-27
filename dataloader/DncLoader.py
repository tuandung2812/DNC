import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


from BaseLoader import BaseLoader
from generator import get_set_and_loader
import yaml

class Dataloader(BaseLoader):

    def __init__(self, config):
        super().__init__(config)

        self.dataset_sources, self.loader_sources, self.data_sources = self.get_loader_and_dataset(mode="source")
        self.dataset_tests, self.loader_tests, self.data_tests = self.get_loader_and_dataset(mode="test")
        self.dataset_targets, self.loader_targets, self.data_targets = self.get_loader_and_dataset(mode="target")

    def gen_test_data(self, target_id):
        return self.data_targets[target_id]

    def get_loader_and_dataset(self, mode):
        domain_datasets = []
        domain_loaders = []
        domain_data = []

        if mode == "source":
            domain_paths = self.source_paths
            time = self.source_time
        if mode == "test":
            domain_paths = self.target_paths
            time = self.test_time
        if mode == "target":
            domain_paths = self.target_paths
            time = self.target_time

        for domain_path in domain_paths:
            data = self.get_data(domain_path, time)
            # print(mode)
            # print(data['X'].shape, data['Y'].shape)
            dataset, loader = get_set_and_loader(
                X=data['X'], Y=data['Y'],
                batch_size=self.batch_size,
                shuffle=False
            )
            domain_datasets.append(dataset)
            domain_loaders.append(loader)
            domain_data.append(data)

        return domain_datasets, domain_loaders, domain_data

    def get_sequence_data(self, sequence):
        n_sample = sequence.shape[0]
        x = [sequence[i: i + self.seq_len]
             for i in range(0, n_sample - self.seq_len - self.future, self.future)]
        y = [sequence[i + self.seq_len: i + self.seq_len + self.future]
             for i in range(0, n_sample - self.seq_len - self.future, self.future)]
        n_sample = min(len(x), len(y))
        x = x[:n_sample]
        y = y[:n_sample]
        x = np.array(x).squeeze()
        y = np.array(y)
        if len(np.array(y).shape) == 1:
            y = np.expand_dims(y, axis=1)
        x = x.reshape((n_sample, self.seq_len))
        y = y.reshape((n_sample, self.future))
        return x, y

    def get_data(self, file_path, time):
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], dayfirst=True)
        df = df.sort_values(by=df.columns[0])
        df = df[(df['time'] > time['start']) & (df['time'] < time['finish'])]
        # print(df.to_numpy().shape)
        array = df[df.columns[1]].to_numpy()
        is_array_nan = pd.isnull(df[df.columns[1]].to_frame()).to_numpy().squeeze()

        flag = False
        start = 0
        x = np.empty(shape=(0, self.seq_len))
        y = np.empty(shape=(0, self.future))

        for i in range(len(array)):
            if flag == False:
                if is_array_nan[i] == False:
                    start = i
                    flag = True
            else:
                if is_array_nan[i] == True or i == array.shape[0] - 1:
                    flag = False
                    if i - start <= self.seq_len + self.future:
                        continue
                    temp_x, temp_y = self.get_sequence_data(array[start:i])
                    if temp_x.shape[0] != 0:
                        x = np.concatenate((x, temp_x), axis=0)
                        y = np.concatenate((y, temp_y), axis=0)

        x_scaler = MinMaxScaler()
        x_scaler.fit(x)
        x = x_scaler.transform(x)
        y_scaler = MinMaxScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)
        return {
            'X': x,
            'Y': y,
            'x-scaler': x_scaler,
            'y-scaler': y_scaler
        }

    def get_number_sources(self):
        return len(self.data_sources)

    def get_number_targets(self):
        return len(self.data_targets)

if __name__ == "__main__":
    # print(1)
    with open("./config/dnc.yaml", "rb") as input:
        config = yaml.load(input, yaml.Loader)
    dataloader = Dataloader(config)
    train_loader = dataloader.get_loader_and_dataset('source')
    # for x,y in train_loader:
        
