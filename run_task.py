import logging
from datetime import datetime

import gokart

import splade_es
from splade_es.tasks.evaluate import *
from splade_es.tasks.splade import *

if __name__ == "__main__":
    gokart.add_config("./conf/param.ini")
    gokart.add_config('./conf/logging.ini')

    Path(__file__).parent.joinpath("log").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"log/{datetime.now():%Y-%m-%d_%H%M%S}.log")
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s')
    fh.setFormatter(formatter)
    logging.root.addHandler(fh)

    gokart.run()
