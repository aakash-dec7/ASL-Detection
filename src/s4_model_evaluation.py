import json
import os
import torch
from Logger import logger
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from src.model import Model


class Evaluation:
    def __init__(self):
        """
        Initializes the Evaluation class.
        """
        self.metrics_dir = "artifacts/Metrics"
        self.model_path = "artifacts/Model/model.pth"

        # Define batch size and dataset path
        self.batch_size = 32
        self.dataset_path = "artifacts/Preprocessed/Test/dataset.pt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = Model().to(self.device)
        self.load_model_weights()

        # Load test dataset
        self.dataset = torch.load(self.dataset_path, weights_only=False)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        logger.info(
            f"Loaded test dataset from '{self.dataset_path}' "
            f"with batch size {self.batch_size} and {len(self.dataloader)} batches."
        )

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-5, weight_decay=0.01
        )
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        logger.info("Optimizer and scheduler initialized.")

    def load_model_weights(self):
        """
        Loads the model weights from the specified path.
        """
        try:
            logger.info(f"Loading model weights from '{self.model_path}'...")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model weights.", exc_info=True)
            raise RuntimeError("Could not load model weights.") from e

    def eval(self):
        """
        Evaluates the model on the test dataset.
        """
        self.model.eval()
        loss_list = []
        correct = 0
        total = 0

        logger.info("Starting evaluation of the model...")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.dataloader):
                logger.debug(f"Processing batch {batch_idx + 1}/{len(self.dataloader)}")

                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                output = self.model.forward(images, labels=labels)
                loss = output.loss
                logits = output.logits

                # Metrics calculation
                loss_list.append(loss.item())
                _, predicted = torch.max(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Calculate final metrics
        avg_loss = sum(loss_list) / len(loss_list)
        accuracy = 100 * correct / total
        logger.info(
            f"Evaluation complete. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )

        # Save metrics to JSON file
        os.makedirs(self.metrics_dir, exist_ok=True)
        eval_metrics_path = os.path.join(self.metrics_dir, "eval_metrics.json")
        metrics = {"accuracy": accuracy, "avg_loss": avg_loss}

        logger.info(f"Saving evaluation metrics to '{eval_metrics_path}'")
        with open(eval_metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Evaluation metrics saved successfully.")

        return avg_loss, accuracy


if __name__ == "__main__":
    try:
        logger.info("Starting Evaluation pipeline...")
        evaluator = Evaluation()
        evaluator.eval()
        logger.info("Evaluation pipeline completed successfully.")
    except Exception as e:
        logger.error("Evaluation pipeline failed.", exc_info=True)
        raise RuntimeError("Evaluation pipeline failed.") from e
