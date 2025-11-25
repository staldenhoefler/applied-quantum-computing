import torch
import wandb

import os
from torch import utils
from medmnist.dataset import RetinaMNIST
from torchvision.transforms import ToTensor
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import torchmetrics
from torchmetrics import MetricCollection
import seaborn as sns

from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


class LitQuantumImageClassifier(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        inp_size: int = 28,
        n_qubits: int = 3,
        in_channels: int = 3,
        num_classes: int = 5,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.inp_size = inp_size
        self.lr = lr
        self.n_qubits = n_qubits
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.train_confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.val_confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.test_confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=self.num_classes)
        self.train_metrics = MetricCollection({
            'precision_micro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='micro'),
            'precision_macro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'precision_weighted': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes,
                                                         average='weighted'),
            'recall_micro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='micro'),
            'recall_macro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'recall_weighted': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'accuracy_micro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro'),
            'accuracy_macro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='macro'),
            'accuracy_weighted': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes,
                                                       average='weighted'),
            'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro'),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
        }, prefix="train_")

        self.val_metrics = MetricCollection({
            'precision_micro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='micro'),
            'precision_macro': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average='macro'),
            'precision_weighted': torchmetrics.Precision(task='multiclass', num_classes=self.num_classes,
                                                         average='weighted'),
            'recall_micro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='micro'),
            'recall_macro': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='macro'),
            'recall_weighted': torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average='weighted'),
            'accuracy_micro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro'),
            'accuracy_macro': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='macro'),
            'accuracy_weighted': torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes,
                                                       average='weighted'),
            'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='micro'),
            'f1_macro': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro'),
            'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='weighted'),
        }, prefix="val_")

        self.in_features = in_channels * inp_size * inp_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh(),
        )

        self.q_layer = self._build_quantum_layer(n_qubits)

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )



    def _build_quantum_layer(self, n_qubits: int) -> nn.Module:
        """
        EstimatorQNN + TorchConnector mapping R^n_qubits -> R^n_qubits
        via a parameterized circuit.
        """
        feature_map = ZZFeatureMap(feature_dimension=n_qubits)

        ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1, flatten=True)

        qc = feature_map.compose(ansatz)

        observables = []
        for i in range(n_qubits):
            pauli = ["I"] * n_qubits
            pauli[i] = "Z"
            pauli = "".join(pauli)
            observables.append(SparsePauliOp.from_list([(pauli, 1.0)]))

        estimator = AerEstimator(
            run_options={
                "method": "statevector",  # 'statevector' is much faster than shots for < 20 qubits
                "max_parallel_threads": 0,  # Use all CPU cores
            },
            transpile_options={"optimization_level": 0},  # 0 is faster to compile
            approximation=True  # Skips sampling, uses exact math (Faster & Lower Variance)
        )

        qnn = EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            observables=observables,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            input_gradients=True,
        )

        torch_qnn = TorchConnector(qnn)
        return torch_qnn

    def forward(self, x):
        z = self.encoder(x)

        z_q = self.q_layer(z)
        z_q = z_q.float()

        logits = self.classifier(z_q)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1).long()

        if self.current_epoch > 0 and self.current_epoch % 5 != 0:
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)
        # ------------------------

        z_q = self.q_layer(z)

        if z_q.dtype == torch.float64:
            z_q = z_q.float()

        logits = self.classifier(z_q)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1).long()

        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.val_metrics.update(logits, y)
        self.val_confmat.update(logits, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        cm = self.train_confmat.compute().cpu().numpy()

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title("Train Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        self.logger.experiment.log({"train_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        self.train_metrics.reset()
        self.train_confmat.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        cm = self.val_confmat.compute().cpu().numpy()
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_title("Validation Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        self.logger.experiment.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

        self.val_metrics.reset()
        self.val_confmat.reset()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def start_training():
    # init the classifier
    model = LitQuantumImageClassifier(
        lr=1e-3,
        inp_size=28,
        n_qubits=6,
        in_channels=3,
        num_classes=5,
    )

    batch_size = 8

    dataset = RetinaMNIST(
        root=os.getcwd(),
        split='train',
        transform=ToTensor(),
        download=True,
    )
    train_loader = utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    dataset_val = RetinaMNIST(
        root=os.getcwd(),
        split='val',
        transform=ToTensor(),
        download=True,
    )
    val_loader = utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
    )

    wandb_logger = WandbLogger(project='Applied-Quantum-Computer')
    wandb_logger.experiment.config["batch_size"] = batch_size

    trainer = pl.Trainer(
        devices=1,
        max_epochs=200,
        logger=wandb_logger,
        log_every_n_steps=5,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()