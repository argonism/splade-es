from datetime import datetime
import logging

import gokart

import splade_es
from splade_es.tasks.splade import *
from splade_es.tasks.evaluate import *

if __name__ == "__main__":
    gokart.add_config("./conf/param.ini")
    gokart.add_config('./conf/logging.ini')

    logger = logging.getLogger('root')
    fh = logging.FileHandler(f"log/{datetime.now():%Y-%m-%d_%H%M%S}.log")
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    gokart.run()