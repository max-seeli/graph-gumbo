import torch
from torch_geometric.loader import DataLoader
from wandb_logger import WandBLogger
from tqdm import tqdm

class PerformanceMetric:

    def __init__(self, classes):
        """
        A structure to keep track of the performance metrics.

        Parameters
        ----------
        classes : list
            A list of class labels.
        """
        self.classes = classes
        self.reset()

    def reset(self):
        """
        Reset the metrics.
        """
        self.total_correct = 0
        self.total = 0
        self.total_correct_per_class = torch.zeros(len(self.classes), dtype=torch.int64)
        self.total_per_class = torch.zeros(len(self.classes), dtype=torch.int64)
        self.pred = []
        self.true = []

    def update(self, pred, true):
        """
        Update the metrics.

        Parameters
        ----------
        pred : torch.Tensor of shape (N,c)
            The predicted class labels.
        true : torch.Tensor of shape (N,)
            The true class labels.
        """
        predicted_classes = pred.argmax(dim=1)
        correct = predicted_classes == true
        self.total_correct += correct.sum().item()
        self.total += len(true)
        
        for c in range(len(self.classes)):
            mask = true == c
            self.total_correct_per_class[c] += correct[mask].sum()
            self.total_per_class[c] += mask.sum()

        self.pred.extend(predicted_classes.tolist())
        self.true.extend(true.tolist())

    def compute(self):
        """
        Compute the metrics.

        Returns
        -------
        dict
            A dictionary containing the computed metrics.
        """
        return {
            'accuracy': self.accuracy(),
            'mean_class_accuracy': self.mean_class_accuracy(),
        }
    
    def accuracy(self):
        """
        Compute the accuracy.

        Returns
        -------
        float
            The accuracy.
        """
        return self.total_correct / self.total
    
    def class_accuracy(self):
        """
        Compute the class accuracy.

        Returns
        -------
        dict
            A dictionary containing the class accuracy.
        """
        return {self.classes[i]: self.total_correct_per_class[i].item() / self.total_per_class[i].item()
                for i in range(len(self.classes))}
    
    def mean_class_accuracy(self):
        """
        Compute the mean class accuracy.

        Returns
        -------
        float
            The mean class accuracy.
        """
        return sum(self.class_accuracy().values()) / len(self.classes)
        
    def confusion_matrix(self):
        """
        Compute the confusion matrix.

        Returns
        -------
        torch.Tensor
            The confusion matrix.
        """
        cm = torch.zeros(len(self.classes), len(self.classes), dtype=torch.int64)
        for p, t in zip(self.pred, self.true):
            cm[t, p] += 1
        return cm


class GNNTrainer:
    
    def __init__(self, model, 
                 optimizer, lr_scheduler, loss_fn,
                 num_epochs, val_every,
                 train_metrics, val_metrics,
                 train_data, val_data, batch_size,
                 checkpoint_dir, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.val_every = val_every
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.checkpoint_dir = checkpoint_dir
        self.device = device 

        self.best_val_acc = 0

        self.wandb = WandBLogger(enabled=True, model=model)

    def train(self):
        for epoch in range(self.num_epochs):
            
            train_loss = self.train_epoch(epoch)
            train_metrics = self.train_metrics.compute()
            print(f'[@{epoch}] LR: {self.optimizer.param_groups[0]["lr"]:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_metrics["accuracy"]:.4f} | Train Mean Class Acc: {train_metrics["mean_class_accuracy"]:.4f}', end='')
            
            if epoch % self.val_every == 0:
                val_loss = self.val_epoch(epoch)
                val_metrics = self.val_metrics.compute()

                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(epoch)

                print(f' | Val Loss: {val_loss:.4f} | Val Acc: {val_metrics["accuracy"]:.4f} | Val Mean Class Acc: {val_metrics["mean_class_accuracy"]:.4f}')
                self.wandb.log({
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, commit=False, step=epoch)
            else:
                print()

            self.lr_scheduler.step()

            self.wandb.log({
                'train_loss': train_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()}
            }, commit=True, step=epoch)

        self.wandb.finish()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.train_metrics.reset()

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.loss_fn(out, data.y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.train_metrics.update(out, data.y)

        return total_loss / len(self.train_loader)
    
    def val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        self.val_metrics.reset()

        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.loss_fn(out, data.y)

                total_loss += loss.item()
                self.val_metrics.update(out, data.y)

        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f'{self.checkpoint_dir}/model_{epoch}.pt')
