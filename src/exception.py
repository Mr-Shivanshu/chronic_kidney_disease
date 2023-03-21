import os
import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exec_tb=error_detail.exc_info()
    line_number=exec_tb.tb_frame.f_lineno
    file_name=exec_tb.tb_frame.f_code.co_filename
    error_message=f"Error occured in python script name[{file_name}] line_number[{line_number}] error message[{error}]"

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
        
        def __str__(self):
            self.error_message
            