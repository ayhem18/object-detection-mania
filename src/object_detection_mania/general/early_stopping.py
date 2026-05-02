import torch

class EarlyStopping:
    def __init__(self, path: str, max_iterations: int = 5, improve_threshold: float = 1e-5):
        """
        Args:
            path (str): Path to save the best model checkpoint.
            max_iterations (int): Number of epochs to wait for improvement before stopping.
            improve_threshold (float): Minimum improvement in val_loss to be considered significant.
        """
        self.path = path
        self.max_iterations = max_iterations
        self.improve_threshold = improve_threshold
        self.best_val_loss = float('inf')
        self.counter = 0

    def initialize(self):
        """
        Resets the internal state of the early stopping mechanism.
        """
        self.best_val_loss = float('inf')
        self.counter = 0
        print(f"EarlyStopping initialized. Monitoring val_loss (threshold: {self.improve_threshold}). Best model will be saved to {self.path}")

    def check_early_stop(self, model: torch.nn.Module, current_val_loss: float) -> bool:
        """
        Checks if training should stop based on the current validation loss.
        
        Args:
            model (torch.nn.Module): The model to save if it's the best so far.
            current_val_loss (float): The current validation loss.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
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
