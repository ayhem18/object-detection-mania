import torch
import os

class EarlyStopping:
    def __init__(self, path: str, max_iterations: int = 5, improve_threshold: float = 1e-5):
        self.path = path
        self.max_iterations = max_iterations
        self.improve_threshold = improve_threshold
        self.best_val_loss = float('inf')
        self.counter = 0

    def initialize(self):
        self.best_val_loss = float('inf')
        self.counter = 0
        print(f"EarlyStopping initialized. Monitoring val_loss (threshold: {self.improve_threshold}). Best model will be saved to {self.path}")

    def check_early_stop(self, model: torch.nn.Module, current_val_loss: float) -> bool:
        if current_val_loss < self.best_val_loss - self.improve_threshold:
            self.best_val_loss = current_val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"Validation loss improved significantly to {self.best_val_loss:.7f}. Model saved.")
            return False

        self.counter += 1
        print(f"No significant improvement in validation loss ({current_val_loss:.7f}). Counter: {self.counter}/{self.max_iterations}")
        if self.counter >= self.max_iterations:
            print("Early stopping triggered.")
            return True
        return False
