import win32com.client
import socket
from alicat import FlowController
import asyncio

def who_am_i():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))  
        address = s.getsockname()[0]
    if address == '10.123.1.121':
        print('You are running on the microscope computer')
        return address
    else:
        print('you are not running on the microscope computer')
        return
    
conditions = {
    'basal': {
        'N2': 0.79,
        'O2': 0.21,
        'CO2': 0.00
        },
    'hypercarbonia': {
        'N2': 0.74,
        'O2': 0.21,
        'CO2': 0.05
        },
    'hypoxia': {
        'N2': 0.90,
        'O2': 0.10,
        'CO2': 0.00
        },
    'test': {
        'N2': 1,
        'O2': 0.00,
        'CO2': 0.00
        },
    'zero': {
        'N2': 0,
        'O2': 0.00,
        'CO2': 0.00
        }
}

async def connect_alicats(port, units: list=['A', 'B', 'C']):
    co2 = FlowController(port, unit=units[2])
    await co2.set_gas('CO2')
    o2 = FlowController(port, unit=units[1])
    await o2.set_gas('O2')
    n2 = FlowController(port, unit=units[0])
    await n2.set_gas('N2')
    return co2, o2, n2

async def get_alicat_info(co2, o2, n2):
    co2result = await co2.get()
    o2result = await o2.get()
    n2result = await n2.get()
    return co2result, o2result, n2result


async def close_alicats(co2, o2, n2):
    await co2.close()
    await o2.close()
    await n2.close()

async def set_gas_flow_composition(co2, o2, n2, conditionsDict, maxFlow: float=0.1):
    n2Flow =  float(conditionsDict['N2'])* float(maxFlow)
    o2Flow =  float(conditionsDict['O2'])* float(maxFlow)
    co2Flow = float(conditionsDict['CO2'])*float(maxFlow)
    await co2.set_flow_rate(co2Flow)
    await o2.set_flow_rate(o2Flow)
    await n2.set_flow_rate(n2Flow)

def connect_to_prairie_view(address):
    pl = win32com.client.Dispatch('PrairieLink.Application')
    pl.Connect(address,'77B2')
    if pl.Connected():
        print('Successfully Connected to Prairie View')
        return pl
    else:
        print('Did not connect to prairie view - try opening prairie view')
        return

def disconnect_from_prairie_view(pl):
    pl.Disconnect()

def run_single_trial(pl):
    pl.sendScriptCommands('-ts')