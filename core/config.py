import os
import logging
from dotenv import load_dotenv

def setup_config():
    """Load environment variables based on the environment."""
    if not os.getenv('WEBSITE_INSTANCE_ID'):
        load_dotenv()
        logging.info("Running in development mode - loaded .env file")
    else:
        logging.info("Running in Azure - using Application Settings")
