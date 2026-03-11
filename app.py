# -*- coding: utf-8 -*-
"""
Gradio App
"""
import gradio as gr
from langchain.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain.tools import tool
import requests
import re
from datetime import datetime
from bs4 import BeautifulSoup

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
def get_peak_details():
    """
    Get list of mountain peaks in the cairngorm national park, their heights and location.

    """
    response = requests.get('https://www.cairngormpark.co.uk/mountains.htm',headers=headers)
    gorms_raw = BeautifulSoup(response.text, 'html.parser').find_all("p")
    return(gorms_raw)

def gen_response(prompt, messages, history):
    messages.append(HumanMessage(prompt))
    tools_called = []
    response = genmodel.invoke(messages)
    while True:
        response = genmodel.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        # Execute each tool call and append results
        for call in response.tool_calls:
            print(call['name'])
            tools_called.append(call['name'])
            print(call['name'])
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    print(response.content)
    print(''.join([tool for tool in tools_called]))
    
    if len(tools_called) > 0:
        history.append({"role": "assistant", "content": response.content, "metadata":{"title":"Tools Called: " + ', '.join([tool for tool in tools_called])}})
    else:
        history.append({"role": "assistant", "content": response.content})
    
    
    print(history)
    return(history,history,messages)

def append_to_history(usrtxt,history):
    history.append({"role": "user", "content": usrtxt})
    return(history,history)

print("Connecting to Ollama")
genmodel = ChatOllama(model="hf.co/unsloth/Qwen3-4B-Thinking-2507-GGUF:Q8_0",think=True,stream=False)
print("Creating agent's toolkit")
tools = [get_datetime_string, get_mwis_forecast, get_sais_forecast, get_peak_details]
tools_by_name = {t.name: t for t in tools}
genmodel = genmodel.bind_tools(tools)

starting_history = [{"role": "assistant", "content": "Mighty Cairngorm-O-Tron will hear your puny questions now!"}]
starting_messages = [
    SystemMessage("""
                  You are a mighty computer.
                  Your mission is to provide answers to questions from hikers in relation to the cairngorms national park.
                  When specific mountains are discussed, your answer must consider their height and location which can be obtained by using the get_cairngorms tool.
                  It is critical for safety that you do not make up information or guess when you have insufficient data from your tools to render a response.
                  """),
    AIMessage("Mighty Cairngorm-O-Tron will hear your questions now!")        
                  ]
                  
with gr.Blocks() as app:
    history = gr.State(starting_history) 
    messages = gr.State(starting_messages)  
    gr.HTML('''<h1>Cairngorm-O-Tron</h1>\n<a href="https://github.com/TheScarletBadger/Cairngorm-o-tron">https://github.com/TheScarletBadger/Cairngorm-o-tron</a>''')
    with gr.Row():
        text_output = gr.Chatbot(value=starting_history,height="80vh",label='Chat history')
    with gr.Row():
        text_input = gr.Textbox(label='Input')
        text_input.submit(append_to_history, inputs=[text_input,history], outputs=[text_output,history]).then(gen_response, inputs=[text_input,messages,history], outputs=[text_output,history,messages])

gr.close_all()
app.launch()


