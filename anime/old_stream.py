from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio
import numpy as np
import pickle

app = FastAPI()

# # get the firing arrays
# with open('/home/yaning/Documents/large_files/fires.pkl', 'rb') as f:
#     data = pickle.load(f)

# In_fires = data["In_fires"]
# E_fires = data["E_fires"]
# I_fires = data["I_fires"]
# Out_fires = data["Out_fires"]



neuron_positions = []
spacing = 40
left_gap = 100
top_gap = 100
layer_id = 0


# Layer 1: 10x10 grid with 8 neurons arranged inside each cell (2x4 layout)
grid_W, grid_H, neurons_per_cell = 10, 10, 8
cell_width, cell_height = 1.0 / grid_W, 1.0 / grid_H

inner_layout = [(i % 2, i // 2) for i in range(neurons_per_cell)]  # 2 columns x 4 rows

for i in range(grid_H):
    for j in range(grid_W):
        cell_origin_x = j * cell_width
        cell_origin_y = i * cell_height
        for offset_x, offset_y in inner_layout:
            neuron_positions.append({
                'x': cell_origin_x + (offset_x + 0.5) * (cell_width / 2),
                'y': cell_origin_y + (offset_y + 0.5) * (cell_height / 4),
                'layer': layer_id,
                # 'cell_i': i,
                # 'cell_j': j
            })
layer_id += 1

# Layer 2: 20x20 grid
grid_W, grid_H = 20, 20
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 500,  # Shift to the right to separate layers visually
            'y': top_gap + i * spacing,
            'layer': layer_id
        })
layer_id += 1

# Layer 3: 4x4 grid
grid_W, grid_H = 4, 4
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing,
            'y': top_gap + i * spacing + 1000,  # Shift downward
            'layer': layer_id
        })
layer_id += 1

# Layer 4: 5x5 grid
grid_W, grid_H = 5, 5
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 500,
            'y': top_gap + i * spacing + 1000,
            'layer': layer_id
        })
layer_id += 1

# Example: Precomputed firing states
N = 100  # neurons
T = 1000  # timesteps
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
        states = firing_states[t].tolist()
        await websocket.send_json({"frame": t, "states": states})
        await asyncio.sleep(0.05)  # 20 FPS
    await websocket.close()