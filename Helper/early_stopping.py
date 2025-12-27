import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity

from collections import deque

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_accuracy, model, checkpoint_path="agnews_faithfulness/best_model.pt"):
        score = val_accuracy

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.early_stop = True

        return self.early_stop