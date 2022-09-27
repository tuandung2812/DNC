from utils.metrics import *
class BaseSupervisor:
    def __init__(self):
        self.__metrics_methods = metrics

    def compute_metrics(self, y_true, y_pred, metrics):
        eval_result = {}
        for metric in metrics:
            eval_result[metric] = self.__metrics_methods[metric](y_true, y_pred)
        return eval_result
