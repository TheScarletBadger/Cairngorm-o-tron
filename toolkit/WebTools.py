# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:40:34 2026

@author: Barry
"""
from langchain.tools import tool
from bs4 import BeautifulSoup
import requests
import re

#Spoof headers for use with requests
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"}

@tool
def Webtool_MWIS():
    """
    Get mountain weather forecast for the Cairngorm national park from the Mountain Weather Informtion Service (MWIS)
    """
    try:
        response = requests.get('https://www.mwis.org.uk/forecasts/scottish/cairngorms-np-and-monadhliath/text',headers=headers)
        mwis_raw = BeautifulSoup(response.text, 'html.parser').find_all('div', class_='forecast')
        mwis='---\n\nMountain Weather Information Service (MWIS) Forecast\n\n'
        for forecast in mwis_raw:
            cf = forecast.get_text(separator=" ", strip=True)
            cf = re.sub(r'\s+', ' ', cf).strip()
            mwis = mwis + cf + '\n\n' 
        return(mwis)
    except:
        return('Error - Tool Unavailable - Could read MWIS website')
    
@tool
def Webtool_SAIS():
    """
    Get avalanche risk forecast for the Cairngorms from Scottish Avalanche Information Service (SAIS).
    If response indicates that SAIS has finished reporting for the winter Avalanch risk can be assumed to be negligable.
    """
    try:
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
    except:
        return('Error - Tool unavailable - Could not read SAIS website')