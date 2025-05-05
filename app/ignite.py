import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from core.openfabric_wrapper import Starter

if __name__ == '__main__':
    PORT = 8889
    logger.info("Starting application on port %d", PORT)
    Starter.ignite(debug=True, host="0.0.0.0", port=PORT)
