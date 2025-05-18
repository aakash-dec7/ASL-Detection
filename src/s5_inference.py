import json
import cv2
import torch
from PIL import Image
from Logger import logger
from src.model import Model
from torchvision import transforms


class Inference:
    def __init__(self):
        """
        Initializes Inference class.
        """
        self.model_path = "artifacts/Model/model.pth"
        self.labels_map_path = "artifacts/Preprocessed/labels_map.json"

        # Select the appropriate device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize and move the model to the selected device
        self.model = Model().to(self.device)
        self.load_model_weights()

        # Load the label mapping file
        self.load_labels_map()

        # Define image preprocessing steps
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # Resize input image
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Normalize
            ]
        )
        logger.info("Preprocessing pipeline initialized.")

    def load_labels_map(self):
        """
        Loads the label mapping from a JSON file and converts string keys to integers.
        """
        try:
            logger.debug("Loading label map...")
            with open(self.labels_map_path, "r") as f:
                original_map = json.load(f)
                self.labels_map = {int(v): k for k, v in original_map.items()}
            logger.info("Label map loaded successfully.")
        except Exception as e:
            logger.error("Failed to load label map.", exc_info=True)
            raise RuntimeError("Could not load label map.") from e

    def load_model_weights(self):
        """
        Loads the trained model weights from disk.
        """
        try:
            logger.debug(f"Loading model weights from {self.model_path}...")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model weights.", exc_info=True)
            raise RuntimeError("Could not load model weights.") from e

    def predict(self, image):
        """
        Performs inference on a given PIL image.
        """
        logger.debug("Starting inference...")

        self.model.eval()

        with torch.no_grad():
            try:
                # Apply preprocessing transformations and move image to device
                transformed_image = self.transform(image).unsqueeze(0).to(self.device)
                logger.debug("Image transformed and moved to device.")

                # Forward pass through the model
                prediction = self.model(transformed_image)
                predicted_index = torch.argmax(prediction, dim=-1).item()
                logger.debug(f"Raw prediction index: {predicted_index}")

                # Convert predicted index to class label
                predicted_label = self.labels_map.get(predicted_index, "!")
                logger.info(f"Prediction successful: {predicted_label}")
                return predicted_label
            except Exception as e:
                logger.error("Prediction failed.", exc_info=True)
                raise RuntimeError("Prediction failed.") from e
