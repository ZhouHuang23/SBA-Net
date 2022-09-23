# -*- coding: utf-8 -*-

import logging
import colorlog

log_colors_config = {
    'DEBUG': 'white',
    'INFO': 'purple',
    'WARNING': 'blue',
    'ERROR': 'yellow',
    'CRITICAL': 'bold_red',
}


def set_logger(logs_filename=None):
    global file_handler
    file_handler = None

    logger = logging.getLogger('logger_name')
    # Output to console
    console_handler = logging.StreamHandler()

    # -----------------------------------------------------------------------
    # Output to file
    if logs_filename is not None:
        file_handler = logging.FileHandler(filename=logs_filename, mode='a', encoding='utf8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            fmt='[%(asctime)s.%(msecs)03d]%(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

    # 日志级别，logger 和 handler以最高级别为准，不同handler之间可以不一样，不相互影响
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # 日志输出格式
    console_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s# %(message)s',
        log_colors=log_colors_config
    )
    console_handler.setFormatter(console_formatter)

    # 重复日志问题：
    # 1、防止多次addHandler；
    # 2、loggername 保证每次添加的时候不一样；
    # 3、显示完log之后调用removeHandler
    if not logger.handlers:
        logger.addHandler(console_handler)
        if logs_filename is not None:
            logger.addHandler(file_handler)

    console_handler.close()
    if logs_filename is not None:
        file_handler.close()

    return logger


if __name__ == '__main__':
    logger = set_logger()
    logger.debug('hello')
    logger.info('hello')
    logger.warning('hello')
    logger.error('hello')
