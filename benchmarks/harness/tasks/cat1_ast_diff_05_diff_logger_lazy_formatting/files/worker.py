import logging

logger = logging.getLogger(__name__)


def process(job_id: int, user_id: int) -> None:
    logger.info(f'Processing job {job_id} for user {user_id}')
