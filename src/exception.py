import sys

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script: {0}, line number: {1}, error message: {2}".format(
        file_name, exc_tb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)  # Pass 'error_message' instead of 'self'
    def __str__(self):
        return self.error_message

try:
    # Simulate an error
    x = 1 / 0
except Exception as e:
    # Wrap the error with CustomException
    custom_error = CustomException("Division by zero", sys.exc_info())
    # Raise the custom error
    raise custom_error
