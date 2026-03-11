# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:37:44 2026

@author: Barry
"""

import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFacePipeline

#Spoof headers for use with requests
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"}

#Load text generation model
genmodel = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-4B-Thinking-2507",
    task="text-generation",
    device_map="auto"
    )

#Get current date and time
def get_datetime_string():
    now = datetime.now()
    str1=now.strftime("%H:%M %A %d %B")
    str2=now.strftime("%d/%m/%y")
    return(f'---\n\nTime\nThe current date and time is {str1} ({str2})')

#Get forecast from Mountain Weather Informtion Service (MWIS)
def get_mwis_forecast():
    response = requests.get('https://www.mwis.org.uk/forecasts/scottish/cairngorms-np-and-monadhliath/text',headers=headers)
    mwis_raw = BeautifulSoup(response.text, 'html.parser').find_all('div', class_='forecast')
    mwis='---\n\nMountain Weather Information Service (MWIS) Forecast\n\n'
    for forecast in mwis_raw:
        cf = forecast.get_text(separator=" ", strip=True)
        cf = re.sub(r'\s+', ' ', cf).strip()
        mwis = mwis + cf + '\n\n' 
    return(mwis)

#Get avalanche forecast from Scottish Avalanche Information Service (SAIS)
def get_sais_forecast():
    response = requests.get('https://www.sais.gov.uk/northern-cairngorms',headers=headers)
    raw = BeautifulSoup(response.text, 'html.parser').find_all('div', id='forecast-info')
    sais ='---\n\nScottish Avalanche Information Service (SAIS) Forecast - Northern Cairngorms\n\n'
    for forecast in raw:
        cf = forecast.get_text(separator=" ", strip=True)
        cf = re.sub(r'\s+', ' ', cf).strip()
        sais = sais + cf + '\n\n' 
    response = requests.get('https://www.sais.gov.uk/southern-cairngorms',headers=headers)
    raw = BeautifulSoup(response.text, 'html.parser').find_all('div', id='forecast-info') 
    sais = sais + '---\n\nScottish Avalanche Information Service (SAIS) Forecast - Southern Cairngorms\n\n'    
    for forecast in raw:
        cf = forecast.get_text(separator=" ", strip=True)
        cf = re.sub(r'\s+', ' ', cf).strip()
        sais = sais + cf + '\n\n' 
    return(sais)

#print(get_datetime_string())
#print(get_mwis_forecast())
#print(get_sais_forecast())

prompt = 'Please provide a concise summary of the following reports\n'

X = genmodel.invoke(prompt + get_datetime_string() + get_mwis_forecast() + get_sais_forecast())