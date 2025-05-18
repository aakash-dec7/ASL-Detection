import os
import pandas as pd
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from Logger import logger


class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion class.
        """
        self.dataset_name = "grassknoted/asl-alphabet"
        self.download_dir = "artifacts/ASL_Dataset"

        self.kaggle_json_path = Path.home() / ".kaggle"  # Path to Kaggle credentials
        self.set_kaggle_credentials()

    def set_kaggle_credentials(self):
        """
        Sets Kaggle API credentials from the user's .kaggle directory.
        """
        try:
            logger.info("Setting Kaggle API credentials...")
            os.environ["KAGGLE_CONFIG_DIR"] = str(self.kaggle_json_path)
            kaggle_json: Path = self.kaggle_json_path / "kaggle.json"

            # Check if kaggle.json exists
            if not kaggle_json.exists():
                logger.error("kaggle.json not found in the expected directory.")
                raise FileNotFoundError(
                    "kaggle.json file not found. Please upload your Kaggle API key."
                )

            logger.info("Kaggle API credentials found and set successfully.")
        except Exception as e:
            logger.exception("Failed to set Kaggle API credentials.")
            raise RuntimeError("Error setting Kaggle credentials") from e

    def download_dataset(self):
        """
        Downloads the dataset from Kaggle if it's not already present locally.
        """
        try:
            logger.info("Preparing to download dataset...")

            # Create the directory if it doesn't exist
            os.makedirs(self.download_dir, exist_ok=True)

            # Check if dataset is already present
            if os.path.exists(self.download_dir) and os.listdir(self.download_dir):
                logger.info(f"Dataset already exists at: {self.download_dir}")
                return

            logger.info(
                f"Dataset not found locally. Initiating download from Kaggle: {self.dataset_name}"
            )

            # Authenticate with Kaggle API
            api: KaggleApi = KaggleApi()
            api.authenticate()
            logger.info("Kaggle API authentication successful.")

            # Download and extract the dataset
            api.dataset_download_files(
                self.dataset_name, path=self.download_dir, unzip=True
            )
            logger.info(f"Dataset downloaded and extracted to: {self.download_dir}")

        except Exception as e:
            logger.exception("Dataset download failed.")
            raise RuntimeError("Dataset download failed") from e

    def run(self):
        """
        Executes the ingestion pipeline.
        """
        self.download_dataset()


if __name__ == "__main__":
    try:
        logger.info("Starting DataIngestion pipeline...")
        data_preprocessing = DataIngestion()
        data_preprocessing.run()
        logger.info("DataIngestion pipeline completed successfully.")
    except Exception as e:
        logger.error("DataIngestion pipeline failed.", exc_info=True)
        raise RuntimeError("DataIngestion pipeline failed.") from e
