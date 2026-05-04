# -*- coding: utf-8 -*-
"""
Gradio App for interaction with the Cairngorm-O-Tron agent
"""
import gradio as gr
from langchain.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from toolkit.PeakTools import Peaktool_Query_Name, Peaktool_Query_HeightM, Peaktool_Query_HeightFt, Peaktool_List_Peaks
from toolkit.WebTools import Webtool_MWIS, Webtool_SAIS
from toolkit.SysTools import Systool_Current_Time

#Define functions for use by gradio app
def gen_response(prompt, messages, history):
    '''
    Generate responses using tool bound model
    '''
    messages.append(HumanMessage(prompt))
    tools_called = []
    #response = genmodel.invoke(messages)
    while True:
        response = genmodel.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            break
        # Execute each tool call and append results to messages
        for call in response.tool_calls:
            tools_called.append(call['name'])
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    #append response and tools called list to gradio chatbot history
    if len(tools_called) > 0:
        history.append({"role": "assistant", "content": '', "metadata": {"title": "Tools Called: " + ', '.join(tools_called)}})
        history.append({"role": "assistant", "content": response.content})
    else:
        history.append({"role": "assistant", "content": response.content})

    return(history,history,messages)

def append_to_history(usrtxt,history):
    '''
    Append user generated text to gradio chatbot history
    '''
    history.append({"role": "user", "content": usrtxt})
    return(history,history)


#connect to local llm server and load Qwen3-4b-Thinking model
print("Initializing model")
genmodel = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="xxxxxx",
    model="qwen/qwen3-4b-thinking-2507",
    temperature=0.3
)

#bind tools to model
print("Creating agent's toolkit")
tools = [Peaktool_Query_Name, Peaktool_Query_HeightM, Peaktool_Query_HeightFt, Peaktool_List_Peaks, Webtool_MWIS, Webtool_SAIS, Systool_Current_Time]
tools_by_name = {t.name: t for t in tools}
genmodel = genmodel.bind_tools(tools)

#Spoof headers for use with requests
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"}

#initilize messages and gradio chatbot history
starting_history = [{"role": "assistant", "content": "Mighty Cairngorm-O-Tron will hear your puny questions now!"}]
starting_messages = [
    SystemMessage("""
                  You are Cairngorm-O-Tron a mighty computer.
                  Your mission is to provide answers to questions from puny human hikers in relation to the cairngorms national park.
                  It is critical for safety that you do not make up information or guess at unknowns always use the tools available to you.
                  In all your responses maintain the persona of the Mighty Cairngorm-O-Tron
                  """),
    AIMessage("Mighty Cairngorm-O-Tron will hear your puny human questions now!")        
                  ]
                  
with gr.Blocks() as app:
    history = gr.State(starting_history) 
    messages = gr.State(starting_messages)  
    gr.HTML('''<h1>Cairngorm-O-Tron</h1>\n<a href="https://github.com/TheScarletBadger/Cairngorm-o-tron">https://github.com/TheScarletBadger/Cairngorm-o-tron</a>''')
    with gr.Row():
        text_output = gr.Chatbot(value=starting_history,height="65vh",label='Chat history')
    with gr.Row():
        text_input = gr.Textbox(label='Input')
        text_input.submit(append_to_history, inputs=[text_input,history], outputs=[text_output,history]).then(gen_response, inputs=[text_input,messages,history], outputs=[text_output,history,messages])

gr.close_all()
app.launch()


