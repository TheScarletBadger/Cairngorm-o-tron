# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:18:09 2026

@author: Barry
"""
from langchain.tools import tool
from datetime import datetime

@tool
def Systool_Current_Time():
    """
    Get the current date and time

    """
    now = datetime.now()
    str1=now.strftime("%H:%M on %A %d %B")
    return(f'The current date and time is {str1}')