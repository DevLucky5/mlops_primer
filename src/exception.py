import sys

def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    message = str(error)

    error_message = f"Error: {message} occured.\n[{filename}] script - at [{lineno}] line."
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message
