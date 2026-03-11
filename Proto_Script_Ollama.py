from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup
from langchain.tools import tool

#Spoof headers for use with requests
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"}

@tool
def get_datetime_string():
    """
    Get the current date and time

    """
    now = datetime.now()
    str1=now.strftime("%H:%M %A %d %B")
    str2=now.strftime("%d/%m/%y")
    return(f'---\n\nTime\nThe current date and time is {str1} ({str2})')

@tool
def get_mwis_forecast():
    """
    Get mountain weather forecast for the Cairngorms from the Mountain Weather Informtion Service (MWIS)

    """
    response = requests.get('https://www.mwis.org.uk/forecasts/scottish/cairngorms-np-and-monadhliath/text',headers=headers)
    mwis_raw = BeautifulSoup(response.text, 'html.parser').find_all('div', class_='forecast')
    mwis='---\n\nMountain Weather Information Service (MWIS) Forecast\n\n'
    for forecast in mwis_raw:
        cf = forecast.get_text(separator=" ", strip=True)
        cf = re.sub(r'\s+', ' ', cf).strip()
        mwis = mwis + cf + '\n\n' 
    return(mwis)

@tool
def get_sais_forecast():
    """
    Get avalanche risk forecast for the Cairngorms from Scottish Avalanche Information Service (SAIS)
    """
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

@tool
def get_cairngorms():
    """
    Get list of mountains in the cairngorm national park and their heights and location mountains not listed here can be assumed to not be in the cairngorms.

    """
    print('ping')
    response = requests.get('https://www.cairngormpark.co.uk/mountains.htm',headers=headers)
    gorms_raw = BeautifulSoup(response.text, 'html.parser').find_all("p")
    return(gorms_raw)

print("Connecting to Ollama")
genmodel = ChatOllama(model="hf.co/unsloth/Qwen3-4B-Thinking-2507-GGUF:Q8_0")
print("Creating agent's toolkit")
tools = [get_datetime_string, get_mwis_forecast, get_sais_forecast, get_cairngorms]
tools_by_name = {t.name: t for t in tools}
genmodel = genmodel.bind_tools(tools)

messages = [
    SystemMessage("""
                  Your role is to provide answers to questions in relation to the cairngorms national park.
                  When specific mountains are discussed, your answer must consider their height and location which can be obtained by using the get_cairngorms tool.
                  It is critical you do not make up information.
                  """),
    HumanMessage("What peaks would be the safest to climb tomorrow")
]

#VIBE CODED GARBAGE FOLLOWS
response = genmodel.invoke(messages)
step = 0
# Agentic loop
while True:
    step += 1
    print(f"\n--- Step {step}: Invoking model ---")
    
    response = genmodel.invoke(messages)
    messages.append(response)
    if not response.tool_calls:
        break

    # Execute each tool call and append results
    for tool_call in response.tool_calls:
        print(f"\n  >> Calling tool: {tool_call['name']}")
        tool_fn = tools_by_name[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

print("\n=== Final Response ===")
print(response.content)