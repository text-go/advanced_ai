import os
import psutil
import platform
import subprocess
import pyautogui
import socket
import shutil
import webbrowser
import pyperclip
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, List, TypedDict

load_dotenv()

# --- SYSTEM TOOLS ---

@tool
def get_system_metrics() -> str:
    """Returns real-time CPU, RAM, Disk, and Battery usage."""
    cpu = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    bat = psutil.sensors_battery()
    bat_s = f"{bat.percent}%" if bat else "N/A"
    return f"CPU: {cpu}% | RAM: {ram}% | Disk: {disk}% | Battery: {bat_s}"

@tool
def process_manager(action: str, name: str = "") -> str:
    """Actions: 'list' (top 5 memory), 'kill' (provide process name)."""
    if action == "list":
        procs = sorted(psutil.process_iter(['name', 'memory_percent']), 
                       key=lambda x: x.info['memory_percent'], reverse=True)[:5]
        return "\n".join([f"{p.info['name']}: {p.info['memory_percent']:.1f}%" for p in procs])
    elif action == "kill" and name:
        for p in psutil.process_iter(['name']):
            if name.lower() in p.info['name'].lower():
                p.kill()
        return f"Terminated processes matching: {name}"
    return "Invalid action."

# --- WEB & RESEARCH ---

@tool
def search_web(query: str) -> str:
    """Search the internet for news, code, or general info."""
    try:
        return DuckDuckGoSearchRun().run(query)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def open_url(url: str) -> str:
    """Opens a specific URL in the default web browser."""
    if not url.startswith("http"): url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url}"

# --- FILE OPERATIONS ---

@tool
def file_control(action: str, path: str, content: str = "") -> str:
    """Actions: 'write', 'read', 'delete', 'list_dir'."""
    try:
        if action == "write":
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            return f"File written: {path}"
        elif action == "read":
            with open(path, "r", encoding="utf-8") as f: return f.read()[:500]
        elif action == "delete":
            os.remove(path)
            return f"Deleted {path}"
        elif action == "list_dir":
            return ", ".join(os.listdir(path or "."))
    except Exception as e:
        return f"File error: {str(e)}"

# --- UTILITIES ---

@tool
def screenshot() -> str:
    """Captures and saves a screenshot to the current directory."""
    path = f"luka_snap_{datetime.now().strftime('%H%M%S')}.png"
    pyautogui.screenshot().save(path)
    return f"Screenshot saved: {path}"

@tool
def clipboard(action: str, text: str = "") -> str:
    """Actions: 'copy' (text to clipboard), 'paste' (return clipboard content)."""
    if action == "copy":
        pyperclip.copy(text)
        return "Text copied to clipboard."
    return f"Clipboard content: {pyperclip.paste()}"

@tool
def system_power(action: str) -> str:
    """Actions: 'lock', 'shutdown', 'restart'."""
    sys = platform.system()
    if action == "lock":
        if sys == "Windows": os.system("rundll32.exe user32.dll,LockWorkStation")
        else: return "Lock only supported on Windows."
    elif action == "shutdown": os.system("shutdown /s /t 1")
    return f"System {action} triggered."

@tool
def media_control(action: str) -> str:
    """Actions: 'vol_up', 'vol_down', 'mute'."""
    if action == "vol_up": pyautogui.press("volumeup", presses=5)
    elif action == "vol_down": pyautogui.press("volumedown", presses=5)
    elif action == "mute": pyautogui.press("volumemute")
    return f"Media action {action} performed."

@tool
def get_ip_info() -> str:
    """Returns local and public IP addresses."""
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return f"Hostname: {hostname} | Local IP: {local_ip}"

@tool
def run_cmd(command: str) -> str:
    """Executes a shell command and returns output."""
    try:
        res = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        return res.decode()[:1000]
    except Exception as e:
        return str(e)

# --- AGENT CORE LOGIC ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The history of messages"]

def get_llm():
    key = os.getenv("GOOGLE_API_KEY")
    if not key: raise ValueError("GOOGLE_API_KEY missing.")
    # Binding 10+ tools to the model
    all_tools = [
        get_system_metrics, process_manager, search_web, open_url, 
        file_control, screenshot, clipboard, system_power, 
        media_control, get_ip_info, run_cmd
    ]
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=key).bind_tools(all_tools)

def call_model(state: AgentState):
    user = os.getenv("USER_NAME", "Lalit")
    sys_msg = SystemMessage(content=f"You are LUKA, a god-level agent for {user}. Be efficient, use tools to control the PC, and search the web for missing info.")
    model = get_llm()
    response = model.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}

def router(state: AgentState):
    last_msg = state["messages"][-1]
    return "action" if last_msg.tool_calls else "end"

# Graph Construction
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode([
    get_system_metrics, process_manager, search_web, open_url, 
    file_control, screenshot, clipboard, system_power, 
    media_control, get_ip_info, run_cmd
]))

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"action": "action", "end": END})
workflow.add_edge("action", "agent")
brain = workflow.compile()

def query_luka(user_input: str, history: list):
    inputs = {"messages": history + [HumanMessage(content=user_input)]}
    config = {"recursion_limit": 20}
    result = brain.invoke(inputs, config)
    return result["messages"][-1].content