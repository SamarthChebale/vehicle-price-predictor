import sys

def error_message_detail(error, error_detail: sys = sys):
    if error_detail is not None and hasattr(error_detail, 'exc_info'):
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return f"Error message: [{str(error)}]"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
