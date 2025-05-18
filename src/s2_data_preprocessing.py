import os
import json
import torch
from PIL import Image
from Logger import logger
from torchvision import transforms
from torch.utils.data import TensorDataset


class DataPreprocessing:
    def __init__(self):
        """
        Initializes the DataPreprocessing class.
        """
        self.save_dataset_dir = "artifacts/Preprocessed"
        self.train_dataset_dir = (
            "artifacts/ASL_Dataset/asl_alphabet_train/asl_alphabet_train"
        )
        self.test_dataset_dir = (
            "artifacts/ASL_Dataset/asl_alphabet_test/asl_alphabet_test"
        )

        # Image transformation: resize, convert to tensor, normalize
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        os.makedirs(self.save_dataset_dir, exist_ok=True)

    def preprocess_train_images(self, max_images_per_class=10):
        """
        Preprocesses training images and save them as a TensorDataset.
        """
        logger.info("Preprocessing training images...")
        images, labels = [], []

        for label in sorted(os.listdir(self.train_dataset_dir)):
            label_path = os.path.join(self.train_dataset_dir, label)
            if not os.path.isdir(label_path):
                logger.debug(f"Skipping non-directory: {label_path}")
                continue

            count = 0
            for img_file in os.listdir(label_path):
                if count >= max_images_per_class:
                    break

                img_path = os.path.join(label_path, img_file)
                try:
                    image = Image.open(img_path).convert("RGB")
                    transformed = self.transform(image)
                    images.append(transformed)
                    labels.append(label)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
                finally:
                    torch.cuda.empty_cache()

            logger.info(f"Processed {count} images for label '{label}'.")

        logger.info(f"Total training images collected: {len(images)}")

        # Create label-to-index mapping
        labels = [label.upper() for label in labels]
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        logger.info(
            f"Generated label-to-index mapping for {len(label_to_idx)} classes."
        )

        # Convert image and label lists to tensors
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor([label_to_idx[label] for label in labels])

        # Save label mapping to JSON
        labels_map_path = os.path.join(self.save_dataset_dir, "labels_map.json")
        with open(labels_map_path, "w") as f:
            json.dump(label_to_idx, f, indent=4)
        logger.info(f"Label mapping saved to: {labels_map_path}")

        # Save TensorDataset
        train_dir = os.path.join(self.save_dataset_dir, "Train")
        os.makedirs(train_dir, exist_ok=True)

        dataset = TensorDataset(images_tensor, labels_tensor)
        dataset_path = os.path.join(train_dir, "dataset.pt")
        torch.save(dataset, dataset_path)
        logger.info(f"Training dataset saved at: {dataset_path}")

    def preprocess_test_images(self):
        """
        Preprocesses test images and save them as a TensorDataset using saved label map.
        """
        logger.info("Preprocessing test images...")
        images, labels = [], []

        labels_map_path = os.path.join(self.save_dataset_dir, "labels_map.json")
        if not os.path.exists(labels_map_path):
            logger.error(
                "Missing labels_map.json. Please run training preprocessing first."
            )
            raise FileNotFoundError(
                "Label map not found. Cannot preprocess test images."
            )

        with open(labels_map_path, "r") as f:
            label_to_idx = json.load(f)

        for img_file in os.listdir(self.test_dataset_dir):
            img_path = os.path.join(self.test_dataset_dir, img_file)
            label = img_file.split("_")[0].upper()

            if label not in label_to_idx:
                logger.warning(f"Unknown label '{label}' in {img_file}. Skipping.")
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                transformed = self.transform(image)
                images.append(transformed)
                labels.append(label_to_idx[label])
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
            finally:
                torch.cuda.empty_cache()

        logger.info(f"Total test images processed: {len(images)}")

        # Convert to tensors
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels)

        # Save test dataset
        test_dir = os.path.join(self.save_dataset_dir, "Test")
        os.makedirs(test_dir, exist_ok=True)

        dataset = TensorDataset(images_tensor, labels_tensor)
        dataset_path = os.path.join(test_dir, "dataset.pt")
        torch.save(dataset, dataset_path)
        logger.info(f"Test dataset saved at: {dataset_path}")

    def run(self):
        """
        Executes the preprocessing pipeline.
        """
        self.preprocess_train_images()
        self.preprocess_test_images()


if __name__ == "__main__":
    try:
        logger.info("Starting DataPreprocessing pipeline...")
        data_preprocessing = DataPreprocessing()
        data_preprocessing.run()
        logger.info("DataPreprocessing pipeline completed successfully.")
    except Exception as e:
        logger.error("DataPreprocessing pipeline failed.", exc_info=True)
        raise RuntimeError("DataPreprocessing pipeline failed.") from e
