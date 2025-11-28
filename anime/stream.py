from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from calculation import calculation_function

app = FastAPI()

data_event = asyncio.Event()
shutdown_event = asyncio.Event()

fire_data = None
background_tasks = []

#---------------------neuron positions on page---------------------------------------------
neuron_positions = []
spacing = 40
left_gap = 100
top_gap = 100
layer_id = 0


# Layer 1: 10x10 grid with 8 neurons arranged inside each cell (2x4 layout)
grid_W, grid_H, neurons_per_cell = 10,10,8

inner_layout = [(i % 2, i // 2) for i in range(neurons_per_cell)]  # 2 columns x 4 rows

for i in range(grid_H):
    for j in range(grid_W):
        cell_origin_x = j * spacing * 2
        cell_origin_y = i * spacing * 3
        for offset_x, offset_y in inner_layout:
            neuron_positions.append({
                'x': left_gap + cell_origin_x + offset_x*25,
                'y': top_gap + cell_origin_y + offset_y*25,
                'layer': layer_id
            })
layer_id += 1

# # Layer 2: 20x20 grid
# grid_W, grid_H = 20,20
# for i in range(grid_H):
#     for j in range(grid_W):
#         neuron_positions.append({
#             'x': left_gap + j * spacing + 1500,  # Shift to the right to separate layers visually
#             'y': top_gap + i * spacing,
#             'layer': layer_id
#         })
# layer_id += 1

grid_W, grid_H = 40,40
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 1000,  # Shift to the right to separate layers visually
            'y': top_gap + i * spacing,
            'layer': layer_id
        })
layer_id += 1


# Layer 3: 4x4 grid
grid_W, grid_H = 4,4
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 1500,
            # 'y': top_gap + i * spacing + 1000,  # Shift downward
            'y': top_gap + i * spacing + 1500,
            'layer': layer_id
        })
layer_id += 1

# Layer 4: 5x5 grid
grid_W, grid_H = 5,5
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 2000,
            # 'y': top_gap + i * spacing + 1000,
            'y': top_gap + i * spacing + 1500,
            'layer': layer_id
        })
layer_id += 1

#-------------------------------------------------------------------------------------



@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

class Params(BaseModel):
    u_se_ampa: float
    u_se_nmda: float
    u_se_gaba: float
    tau_rec_ampa: float
    tau_rec_nmda: float
    tau_rec_gaba: float
    tau_rise_ampa: float
    tau_rise_nmda: float
    tau_rise_gaba: float
    learning_rate: float
    weight_scale: float

@app.post("/input_params")
async def get_params(data: Params):
    global fire_data

    print(data)
    fire_data = await run_in_threadpool(calculation_function, data)

    data_event.set()
    

    return "finish computing"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()


    while True:
        # await asyncio.wait(
        #     [data_event.wait(), shutdown_event.wait()],
        #     return_when=asyncio.FIRST_COMPLETED,
        # )

        # # If shutdown triggered before data_event, exit early
        # if shutdown_event.is_set():
        #     await websocket.close()
        #     break
        await asyncio.wait(
        [asyncio.create_task(data_event.wait()), asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_event.is_set():
            await websocket.close()
            break

        In_fires = fire_data["In_fires"]
        E_fires = fire_data["E_fires"]
        I_fires = fire_data["I_fires"]
        Out_fires = fire_data["Out_fires"]
        await websocket.send_json({"neurons": neuron_positions})
        for t in range(In_fires.shape[0]):
            In_states = In_fires[t].view(-1).tolist()
            E_states = E_fires[t].view(-1).tolist()
            I_states = I_fires[t].view(-1).tolist()
            Out_states = Out_fires[t].view(-1).tolist()
            states = In_states + E_states + I_states + Out_states

            await websocket.send_json({"frame": t, "states": states})
            await asyncio.sleep(0.05)  # 20 FPS
        
        data_event.clear()


@app.post("/shutdown")
async def shutdown():
    shutdown_event.set()
    return {"status": "shutdown triggered"}