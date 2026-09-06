import torch as torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self,
                calculate_accuracy,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 loss_steps: int = 100,
                 track_grad_norm: bool = True):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.calculate_accuracy = calculate_accuracy
        self.device = device
        self.loss_steps = loss_steps
        self.track_grad_norm = track_grad_norm

    def train(self, data_loader: torch.utils.data.DataLoader, epoch=None):
        #Training
        train_loss, train_acc = 0, 0
        #Put Data into training Mode
        self.model.train()
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            train_acc += self.calculate_accuracy(y_true=y,
                                                 y_pred=y_pred.argmax(dim=1)) #from logits -> prediction labels
            self.optimizer.zero_grad()
            loss.backward()

            ##### ==== Calculate natin dito ang L2 norm ng gradients ng model parameters =======#
            if self.track_grad_norm:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
            self.optimizer.step()
        train_loss = train_loss / len(data_loader)
        train_acc = train_acc / len(data_loader)
        total_norm = total_norm / len(data_loader) if self.track_grad_norm else None
        if epoch is not None and epoch % self.loss_steps == 0:
            print(f"Training Loss: {train_loss:.5f} | Training Accuracy: {train_acc:.5f}%")
        return train_loss, train_acc, total_norm

    def test(self, data_loader: torch.utils.data.DataLoader, epoch=None):
        #Testing
        test_loss, test_acc = 0, 0
        self.model.to(self.device)
        #Put Data into evaluation Mode
        self.model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                test_pred = self.model(X)
                loss = self.loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += self.calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss = test_loss / len(data_loader)
            test_acc = test_acc / len(data_loader)
            if epoch is not None and epoch % self.loss_steps == 0:
                print(f"Test Loss {test_loss:.5f} | Test Accuracy {test_acc:.5f}%")
            return test_loss, test_acc