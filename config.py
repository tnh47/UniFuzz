import loguru

loguru.logger.add("log/DEBUG.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="DEBUG")
loguru.logger.add("log/INFO.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="INFO")
loguru.logger.add("log/ERROR.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="ERROR")
loguru.logger.add("log/WARNING.log", encoding="utf-8", enqueue=True, backtrace=True, diagnose=True, level="WARNING")

SOLC_BIN_PATH = "/home/ngonhat/Desktop/UniFuzz/.venv/bin/solc"  # set to your solc path
#SOLC_BIN_PATH = "/usr/bin/solc-0.4.26"
################


def get_logger() -> loguru.logger:
    return loguru.logger