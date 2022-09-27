import os
class BaseLoader():

    def __init__(self, config):
        self.seq_len = config['data']['seq-len']
        self.future = config['data']['future']

        self.source_paths = BaseLoader.create_file_path(
            config['data']["data-path"], config['data']["source"]
        )
        self.target_paths = BaseLoader.create_file_path(
            config['data']["data-path"], config['data']["target"]
        )
        self.source_time = config['data']["source-time"]
        self.test_time = config['data']["test-time"]
        self.target_time = config['data']["target-time"]

        self.batch_size = config['data']["batch-size"]

    def create_file_path(folder, file_names):
        file_paths = []
        for file_name in file_names:
            file_path = os.path.join(folder, file_name)
            file_paths.append(file_path)
        return file_paths
