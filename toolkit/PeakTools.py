# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:04:21 2026
@author: Barry
""" 
import json
from importlib.resources import files
from langchain.tools import tool

@tool
def Peaktool_Query_Name(name: str):
    '''
    Fetch info about peaks in the cairngorms national park by name.
    Accepts name of a peak as input.
    Returns basic information about the mountain including height and grid reference.
    Note only Munro peaks are listed by this tool.
    There are lower peaks in the park but no information is available from the tool.
    Returns JSON containing below fields for peak queried.
        name: name of peak
        heightM: height in meters
        heightFt: height in feet
        gridRef: UK national grid reference of the peak
        info: some additional information about the peak
    If no matching peak found returns empty array '[]'
    '''
    try:
        with files('CairngormPeaks').joinpath('cairngorm_mountains.json').open() as f:
            gorms = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Error - Tool unavailable - Failed to load cairngorm_mountains.json"
    return(json.dumps([g for g in gorms if g["name"].lower() == name.lower()]))

@tool
def Peaktool_Query_HeightM(lolim: int, hilim: int):
    '''
    Fetch info about peaks in the cairngorms national park by range of heights in meters.
    Accepts two integers as input.
    The first integer is the lower bound of the height range to query in meters.
    The second integer is the upper bound of the height range to query in meters.
    Note only Munro peaks are listed by this tool and fall between approximately 900m and 1320m in height.
    There are lower peaks in the park but no information is available from the tool.
    Returns JSON containing below fields for each peak in the height range queried.
        name: name of peak
        heightM: height in meters
        heightFt: height in feet
        gridRef: UK national grid reference of the peak
        info: some additional information about the peak
    If no matching peak found returns empty array '[]'
    '''
    try:
        with files('CairngormPeaks').joinpath('cairngorm_mountains.json').open() as f:
            gorms = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Error - Tool unavailable - Failed to load cairngorm_mountains.json"
    return(json.dumps([g for g in gorms if g["heightM"] >= lolim and g["heightM"] <= hilim]))

@tool
def Peaktool_Query_HeightFt(lolim: int, hilim: int):
    '''
    Fetch info about peaks in the cairngorms national park by range of heights in feet.
    Accepts two integers as input.
    The first integer is the lower bound of the height range to query in feet.
    The second integer is the upper bound of the height range to query in feet.
    Note only Munro peaks are listed by this tool and fall between approximately 3000ft and 4300ft in height.
    There are lower peaks in the park but no information is available from the tool.
    Returns JSON containing below fields for each peak in the height range queried.
        name: name of peak
        heightM: height in meters
        heightFt: height in feet
        gridRef: UK national grid reference of the peak
        info: some additional information about the peak
    If no matching peak found returns empty array '[]'
    '''
    try:
        with files('CairngormPeaks').joinpath('cairngorm_mountains.json').open() as f:
            gorms = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Error - Tool unavailable - Failed to load cairngorm_mountains.json"
    return(json.dumps([g for g in gorms if g["heightFt"] >= lolim and g["heightFt"] <= hilim]))

@tool
def Peaktool_List_Peaks():
    '''
    Fetch the names of all peaks in the cairngorms national park.
    Note only Munro peaks are listed by this tool and fall between approximately 900m and 1320m in height.
    There are lower peaks in the park but no information is available from the tool.
    '''
    try:
        with files('CairngormPeaks').joinpath('cairngorm_mountains.json').open() as f:
            gorms = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Error - Tool unavailable - Failed to load cairngorm_mountains.json"
    return(json.dumps([g["name"] for g in gorms]))