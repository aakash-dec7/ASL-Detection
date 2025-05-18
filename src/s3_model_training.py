import os
import json
import torch
from Logger import logger
from src.model import Model
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


class Training:
    def __init__(self):
        """
        Initializes the Training class.
        """
        self.model_dir = "artifacts/Model"
        self.metrics_dir = "artifacts/Metrics"

        # Define batch size and dataset path
        self.batch_size = 32
        self.dataset_path = "artifacts/Preprocessed/Train/dataset.pt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = Model().to(self.device)
        logger.info("Model initialized and moved to device.")

        # Load dataset
        self.dataset = torch.load(self.dataset_path, weights_only=False)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        logger.info(
            f"Dataset loaded from {self.dataset_path} | Batch size: {self.batch_size}"
        )

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-5, weight_decay=0.01
        )
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        logger.info("Optimizer and learning rate scheduler set up successfully.")

    def train(self, num_epochs=5):
        """
        Trains the model for the specified number of epochs.
        """
        self.model.train()
        epoch_loss_list = []

        logger.info(f"Training started for {num_epochs} epoch(s).")

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs} started.")
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(self.dataloader):
                # Zero out gradients
                self.optimizer.zero_grad()

                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                output = self.model.forward(images, labels=labels)
                loss = output.loss

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                logger.debug(
                    f"Epoch {epoch + 1} | Batch {batch_idx + 1} | Loss: {loss.item():.4f}"
                )

            # Compute average loss for the epoch
            avg_loss = epoch_loss / len(self.dataloader)
            epoch_loss_list.append(avg_loss)

            logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_loss:.4f}")

            # Update learning rate and free memory
            self.scheduler.step()
            torch.cuda.empty_cache()
            logger.debug("Scheduler stepped and GPU cache cleared.")

        # Save training metrics
        metrics = {"epoch_loss": epoch_loss_list}
        os.makedirs(self.metrics_dir, exist_ok=True)
        metrics_path = os.path.join(self.metrics_dir, "train_metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Training metrics saved at: {metrics_path}")
        logger.debug(f"Epoch Loss List: {epoch_loss_list}")

        # Save model state
        os.makedirs(self.model_dir, exist_ok=True)
        save_path = os.path.join(self.model_dir, "model.pth")
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Trained model saved at: {save_path}")

        return epoch_loss_list


if __name__ == "__main__":
    try:
        logger.info("Starting Training pipeline...")
        trainer = Training()
        trainer.train()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.error("Training pipeline failed.", exc_info=True)
        raise RuntimeError("Training pipeline failed.") from e
