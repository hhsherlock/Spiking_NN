from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import numpy as np
import pickle
import torch

app = FastAPI()

data = None
# In_fires = torch.zeros((6000,10,10,8))
# E_fires = torch.zeros((6000,20,20))
# I_fires = torch.zeros((6000,4,4))
# Out_fires = torch.zeros((6000,5,5))

In_fires = torch.ones((6000,10,10,8))
E_fires = torch.ones((6000,20,20))
I_fires = torch.ones((6000,4,4))
Out_fires = torch.ones((6000,5,5))

@app.on_event("startup")
async def load_data():
    global data, In_fires, E_fires, I_fires, Out_fires

    # simulate loading large dataset asynchronously
    import asyncio
    await asyncio.sleep(0)  # let event loop run (non-blocking)
    
    # your actual loading code here
    with open('/home/yaning/Documents/large_files/fires.pkl', 'rb') as f:
        import pickle
        data = pickle.load(f)

        In_fires = data["In_fires"]
        E_fires = data["E_fires"]
        I_fires = data["I_fires"]
        Out_fires = data["Out_fires"]







neuron_positions = []
spacing = 40
left_gap = 100
top_gap = 100
layer_id = 0


# Layer 1: 10x10 grid with 8 neurons arranged inside each cell (2x4 layout)
grid_W, grid_H, neurons_per_cell = In_fires.shape[1:]
# grid_W, grid_H, neurons_per_cell = 10,10,8

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

# Layer 2: 20x20 grid
grid_W, grid_H = E_fires.shape[1:]
# grid_W, grid_H = 20,20
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 1500,  # Shift to the right to separate layers visually
            'y': top_gap + i * spacing,
            'layer': layer_id
        })
layer_id += 1

# Layer 3: 4x4 grid
grid_W, grid_H = I_fires.shape[1:]
# grid_W, grid_H = 4,4
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 1500,
            'y': top_gap + i * spacing + 1000,  # Shift downward
            'layer': layer_id
        })
layer_id += 1

# Layer 4: 5x5 grid
grid_W, grid_H = Out_fires.shape[1:]
# grid_W, grid_H = 5,5
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 2000,
            'y': top_gap + i * spacing + 1000,
            'layer': layer_id
        })
layer_id += 1

# firing states
# T = E_fires.shape[0]

T = 10
N = 100
firing_states = np.random.randint(0, 2, (T, N))

@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"neurons": neuron_positions})
    for t in range(T):
        In_states = In_fires[t].view(-1).tolist()
        E_states = E_fires[t].view(-1).tolist()
        I_states = I_fires[t].view(-1).tolist()
        Out_states = Out_fires[t].view(-1).tolist()
        states = In_states + E_states + I_states + Out_states
        # states = firing_states[t].tolist()
        await websocket.send_json({"frame": t, "states": states})
        await asyncio.sleep(0.05)  # 20 FPS
    await websocket.close()
