import logging

def create_logger():
    global logger

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # if not args.debug:
    #     logger.setLevel(logging.CRITICAL)

    if not len(logger.handlers) == 0:
        logger.handlers.clear()

    fh = logging.FileHandler('main.log', mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
