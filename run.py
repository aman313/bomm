#!/media/aman/8a3ffbda-8a36-45f9-b426-d146a65d9ece/data1/research/BagOfModels/venv/bin/ python
import logging
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
sys.path.insert(0,script_dir)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position

def run():
    main(prog="allennlp")

if __name__ == "__main__":
    run()
