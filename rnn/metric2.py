import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn
import pandas as pd

def custom_loss(pred, targets, loss_weight):
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(pred[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(pred[:, 1:], targets[:, 2:])
    return bce_loss_1 + bce_loss_2*6
    # return bce_loss_1+0.5*bce_loss_2
    # return bce_loss_1

class JigsawEvaluator:

    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25, identity_columns=None):
        self.y = (y_true >= 0.5).astype(int)
        self.y_i = (y_identity >= 0.5).astype(int)
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight
        self.identity_columns = identity_columns

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = []
        # records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            record = dict()
            record['subgroup_auc'] = self._compute_subgroup_auc(i, y_pred)
            record['bpsn_auc'] = self._compute_bpsn_auc(i, y_pred)
            record['bnsp_auc'] = self._compute_bnsp_auc(i, y_pred)
            # records[0, i] = self._compute_subgroup_auc(i, y_pred)
            # records[1, i] = self._compute_bpsn_auc(i, y_pred)
            # records[2, i] = self._compute_bnsp_auc(i, y_pred)
            records.append(record)
        return pd.DataFrame(records, index=self.identity_columns)

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            # self._power_mean(bias_metrics[0]),
            # self._power_mean(bias_metrics[1]),
            # self._power_mean(bias_metrics[2])
            self._power_mean(bias_metrics['subgroup_auc']),
            self._power_mean(bias_metrics['bpsn_auc']),
            self._power_mean(bias_metrics['bnsp_auc'])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score, bias_metrics