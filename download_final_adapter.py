#!/usr/bin/env python3
"""
Download Final Adapter Script for EC2 GRPO Inference

Downloads the trained GRPO adapter from Google Drive for matrix multiplication DSL inference.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FINAL_ADAPTER_PATH = "./final_adapter"
GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1bnQEqN-ZvRCeaJ--heMOUliFZL87N9eD?usp=drive_link"

def check_gdown():
    """Check if gdown is installed, install if not"""
    try:
        import gdown
        logger.info("✅ gdown is already installed")
        return True
    except ImportError:
        logger.info("⚠️  gdown not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown>=4.6.0"])
            logger.info("✅ gdown installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install gdown: {e}")
            return False

def download_adapter():
    """Download the final adapter from Google Drive"""
    if os.path.exists(FINAL_ADAPTER_PATH):
        logger.info(f"✅ Final adapter already exists at: {FINAL_ADAPTER_PATH}")
        
        # Check if it contains expected files
        adapter_files = list(Path(FINAL_ADAPTER_PATH).rglob("*.json")) + list(Path(FINAL_ADAPTER_PATH).rglob("*.safetensors"))
        if adapter_files:
            logger.info(f"   Found {len(adapter_files)} adapter files")
            return True
        else:
            logger.warning("   Directory exists but appears empty, re-downloading...")
    
    # Create directory
    os.makedirs(FINAL_ADAPTER_PATH, exist_ok=True)
    
    try:
        # Import gdown after ensuring it's installed
        import gdown
        
        logger.info(f"📦 Downloading final adapter to: {FINAL_ADAPTER_PATH}")
        logger.info(f"   Source: {GOOGLE_DRIVE_URL}")
        
        # Download the folder
        gdown.download_folder(GOOGLE_DRIVE_URL, output=FINAL_ADAPTER_PATH, quiet=False)
        
        # Verify download
        adapter_files = list(Path(FINAL_ADAPTER_PATH).rglob("*.json")) + list(Path(FINAL_ADAPTER_PATH).rglob("*.safetensors"))
        
        if adapter_files:
            logger.info(f"✅ Download successful! Found {len(adapter_files)} adapter files:")
            for file in adapter_files[:5]:  # Show first 5 files
                logger.info(f"   - {file.relative_to(FINAL_ADAPTER_PATH)}")
            if len(adapter_files) > 5:
                logger.info(f"   ... and {len(adapter_files) - 5} more files")
            return True
        else:
            logger.error("❌ Download completed but no adapter files found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to download adapter: {e}")
        return False

def verify_adapter():
    """Verify the downloaded adapter has required files"""
    if not os.path.exists(FINAL_ADAPTER_PATH):
        return False
    
    required_files = ["adapter_config.json"]
    adapter_files = ["adapter_model.safetensors", "adapter_model.bin"]  # Either is fine
    
    # Check for required config
    config_found = any(
        os.path.exists(os.path.join(root, file))
        for root, dirs, files in os.walk(FINAL_ADAPTER_PATH)
        for file in files
        if file in required_files
    )
    
    # Check for adapter weights
    weights_found = any(
        os.path.exists(os.path.join(root, file))
        for root, dirs, files in os.walk(FINAL_ADAPTER_PATH)
        for file in files
        if file in adapter_files
    )
    
    if config_found and weights_found:
        logger.info("✅ Adapter verification passed")
        return True
    else:
        logger.warning("⚠️  Adapter verification failed:")
        if not config_found:
            logger.warning("   - Missing adapter_config.json")
        if not weights_found:
            logger.warning("   - Missing adapter weights (.safetensors or .bin)")
        return False

def main():
    """Main function"""
    logger.info("🚀 GRPO Final Adapter Download Script")
    logger.info("=" * 50)
    
    # Check and install gdown
    if not check_gdown():
        logger.error("❌ Failed to install gdown. Please install manually:")
        logger.error("   pip install gdown>=4.6.0")
        sys.exit(1)
    
    # Download adapter
    if not download_adapter():
        logger.error("❌ Failed to download adapter")
        sys.exit(1)
    
    # Verify adapter
    if not verify_adapter():
        logger.error("❌ Adapter verification failed")
        sys.exit(1)
    
    logger.info("🎉 Final adapter ready for inference!")
    logger.info(f"   Location: {os.path.abspath(FINAL_ADAPTER_PATH)}")
    logger.info("\nYou can now run inference with:")
    logger.info("   python3 inference_ec2.py")
    logger.info("   or")
    logger.info("   ./launch_inference_ec2.sh")

if __name__ == "__main__":
    main() 