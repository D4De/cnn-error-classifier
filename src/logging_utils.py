import sys
import logging as log

def setup_logging(log_path: str):
    root = log.getLogger()
    root.setLevel(log.INFO)

    console_handler = log.StreamHandler(sys.stdout)
    console_handler.setLevel(log.DEBUG)
    formatter = log.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    root.addHandler(console_handler)

    file_handler = log.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(log.INFO)
    file_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    
def set_console_logging_level(level):
    root = log.getLogger()
    for handler in root.handlers:
        if isinstance(handler, type(log.StreamHandler)):
            handler.setLevel(level)
