import logging
import paddle.fluid as fluid

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def check_version():
    """
     Log error and exit when the installed version of paddlepaddle is
     not satisfied.
     """
    err = "PaddlePaddle version 1.7 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code." \

    try:
        fluid.require_version('1.7.0')
    except Exception as e:
        logger.error(err)
        sys.exit(1)
