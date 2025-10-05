import json
import argparse
import numpy as np
import time
import torch
from torch import load as torch_load, cuda as torch_cuda, device as torch_device
from PIL import Image
import os
import logging
import traceback
import yaml
from pathlib import Path

# Placeholder imports - replace with your actual imports
try:
    from modeldata.data.module import SceneTextDataModule
    from modeldata.models.utils import load_from_checkpoint
except ImportError:
    print("Warning: modeldata modules not found. Please ensure they are installed or adjust the imports.")
    # You might need to add the path to your modeldata modules
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), 'modeldata'))
    from data.module import SceneTextDataModule
    from models.utils import load_from_checkpoint

class BhaashaOCR:
    def __init__(self, checkpoint_path, batch_size=20):
        """Initialize the OCR model with checkpoint path"""
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.setup_logging()
        self.initialize_model()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.getLogger("PIL").propagate = False
        self.logger = logging.getLogger(__name__)

    def initialize_model(self):
        """Initialize the model with enhanced logging"""
        try:
            init_start_time = time.time()
            self.logger.info("Starting model initialization...")

            # Validate checkpoint file exists
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

            # Initialize CUDA and device
            self.using_cuda = torch_cuda.is_available()
            self.device = torch_device("cuda" if self.using_cuda else "cpu")
            self.logger.info(f"Device initialization complete. Using CUDA: {self.using_cuda}")

            # Load model from checkpoint
            self.logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
            self.model = load_from_checkpoint(self.checkpoint_path).eval()
            self.model = self.model.to(self.device)

            # Setup transform and batch size from config
            self.img_transform = SceneTextDataModule.get_transform(
                getattr(self.model.hparams, "img_size", (32, 128))  # height=32, width=128
            )

            # config = self.load_config()
            self.batch_size = self.batch_size
            
            init_end_time = time.time()
            self.logger.info(f"Model initialization completed in {init_end_time - init_start_time:.2f}s")
            self.logger.info(f"Batch size set to: {self.batch_size}")

        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def load_image(self, image_path):
        """Load and validate image"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

    def process_batch(self, batch_images):
        """Process a batch of images"""
        try:
            # Transform images
            self.logger.info(f"Transforming batch of {len(batch_images)} images...")
            transformed_images = torch.stack([
                self.img_transform(img) for img in batch_images
            ]).to(self.device)

            # Model inference
            with torch.no_grad():
                logits = self.model(transformed_images)
                logits = logits.detach().cpu()

            # Decode predictions
            pred = logits.softmax(-1)
            labels, _ = self.model.tokenizer.decode(pred)
            
            # Calculate token IDs and per-character confidences
            token_ids = logits.argmax(-1)
            all_confs = []
            for i in range(pred.size(0)):
                confs = pred[i].gather(1, token_ids[i].unsqueeze(-1)).squeeze(-1).tolist()
                all_confs.append(confs)

            # Compute average confidence per prediction
            avg_confidences = [
                round(sum(conf_list) / len(conf_list), 4) if conf_list else 0.0
                for conf_list in all_confs
            ]

            # Return list of (label, avg_confidence)
            return list(zip(labels, avg_confidences))
        
        except Exception as e:
            self.logger.error(f"Error in process_batch: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def infer_image(self, image_path):
        """Process a single image"""
        try:
            self.logger.info(f"Starting inference for image: {image_path}")
            
            # Load image
            image = self.load_image(image_path)
            
            # Process as single batch
            result = self.process_batch([image])
            
            if result:
                recognized_text, confidence = result[0]
                self.logger.info(f"Recognition complete. Text: '{recognized_text}', Confidence: {confidence}")
                return recognized_text, confidence
            else:
                self.logger.warning("No text recognized")
                return "", 0.0

        except Exception as e:
            self.logger.error(f"Error in infer_image: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def infer_multiple_images(self, image_paths, out_dir):
        """Process multiple images in batches and save individual layout.txt files"""
        try:
            self.logger.info(f"Starting inference for {len(image_paths)} images")
            processed_count = 0
            
            # Load all images first
            images = []
            valid_paths = []
            for img_path in image_paths:
                try:
                    image = self.load_image(img_path)
                    images.append(image)
                    valid_paths.append(img_path)
                except Exception as e:
                    self.logger.error(f"Failed to load image {img_path}: {str(e)}")
        
            # Process in batches
            for i in range(0, len(images), self.batch_size):
                batch_end = min(i + self.batch_size, len(images))
                batch_images = images[i:batch_end]
                batch_paths = valid_paths[i:batch_end]
                
                self.logger.info(f"Processing batch {i//self.batch_size + 1}, images {i} to {batch_end}")
                
                if batch_images:
                    batch_results = self.process_batch(batch_images)
                    
                    # Save results for each image in the batch
                    for idx, (img_path, (recognized_text, confidence)) in enumerate(zip(batch_paths, batch_results)):
                        image_name = Path(img_path).stem
                        
                        # Format output as requested
                        output_text = f"IITD_Sanskrit_{recognized_text} {confidence}"
                        
                        # Save layout.txt file with image name
                        layout_output_file = os.path.join(out_dir, f"{image_name}.txt")
                        with open(layout_output_file, 'w', encoding='utf-8') as f:
                            f.write(output_text)
                        
                        processed_count += 1
                        self.logger.info(f"Processed {image_name}: '{recognized_text}' (confidence: {confidence})")
                
                self.logger.info(f"Batch {i//self.batch_size + 1} complete.")

            self.logger.info(f"All processing complete. {processed_count} images processed successfully.")
            return processed_count

        except Exception as e:
            self.logger.error(f"Error in infer_multiple_images: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def get_image_files(self, input_dir):
        """Get all image files from input directory"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in Path(input_dir).iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(str(file_path))
        
        image_files.sort()  # Sort for consistent processing order
        return image_files

def main():
    parser = argparse.ArgumentParser(description="Bhaasha OCR Inference")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained checkpoint file (.ckpt, .pth, or .pt)")
    
    # Make image_path and input_dir mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", help="Path to a single input image for inference")
    group.add_argument("--input_dir", help="Path to directory containing multiple images for batch inference")
    
    parser.add_argument("--out_dir", required=True, help="Path to folder where OCR output is saved")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for processing (default: 20)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained checkpoint file does not exist: {args.pretrained}")
    
    # Validate checkpoint file extension
    valid_extensions = ['.ckpt', '.pth', '.pt']
    if not any(args.pretrained.endswith(ext) for ext in valid_extensions):
        raise ValueError(f"Pretrained file must have one of these extensions: {valid_extensions}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        # Initialize OCR model with checkpoint file
        print(f"Initializing OCR model from checkpoint: {args.pretrained}")
        ocr = BhaashaOCR(args.pretrained, batch_size=args.batch_size)
        
        if args.image_path:
            # Single image processing
            if not os.path.exists(args.image_path):
                raise FileNotFoundError(f"Image path does not exist: {args.image_path}")
            
            print(f"Processing single image: {args.image_path}")
            recognized_text, confidence = ocr.infer_image(args.image_path)
            
            # Save result with image filename
            image_name = Path(args.image_path).stem
            output_text = f"IITD_Sanskrit_{recognized_text} {confidence}"
            
            layout_output_file = os.path.join(args.out_dir, f"{image_name}.txt")
            with open(layout_output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)
            
            print(f"Recognition Result: {recognized_text}")
            print(f"Confidence: {confidence}")
            print(f"Layout output saved to: {layout_output_file}")
            
        elif args.input_dir:
            # Multiple images processing
            if not os.path.exists(args.input_dir):
                raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
            
            print(f"Processing all images in directory: {args.input_dir}")
            image_files = ocr.get_image_files(args.input_dir)
            
            if not image_files:
                print("No supported image files found in the input directory.")
                print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
                return
            
            print(f"Found {len(image_files)} images to process")
            processed_count = ocr.infer_multiple_images(image_files, args.out_dir)
            print(f"Successfully processed {processed_count} images")
            print(f"Layout files saved to: {args.out_dir}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()







