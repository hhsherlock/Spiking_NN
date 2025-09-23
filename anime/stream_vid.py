# stream.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import io
from PIL import Image, ImageDraw
import numpy as np
import time

from calculation import calculation_function  # your existing computation

app = FastAPI()

data_event = asyncio.Event()
shutdown_event = asyncio.Event()

fire_data = None

# ------------------ neuron_positions (keep your existing layout) ---------------------
neuron_positions = []
spacing = 40
left_gap = 100
top_gap = 100
layer_id = 0

# Layer 1: 10x10 grid with 8 neurons arranged inside each cell (2x4 layout)
grid_W, grid_H, neurons_per_cell = 10, 10, 8
inner_layout = [(i % 2, i // 2) for i in range(neurons_per_cell)]
for i in range(grid_H):
    for j in range(grid_W):
        cell_origin_x = j * spacing * 2
        cell_origin_y = i * spacing * 3
        for offset_x, offset_y in inner_layout:
            neuron_positions.append({
                'x': left_gap + cell_origin_x + offset_x * 25,
                'y': top_gap + cell_origin_y + offset_y * 25,
                'layer': layer_id
            })
layer_id += 1

# Layer 2: 20x20 grid
grid_W, grid_H = 20, 20
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 1500,
            'y': top_gap + i * spacing,
            'layer': layer_id
        })
layer_id += 1

# Layer 3: 4x4 grid
grid_W, grid_H = 4, 4
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 1500,
            'y': top_gap + i * spacing + 1000,
            'layer': layer_id
        })
layer_id += 1

# Layer 4: 5x5 grid
grid_W, grid_H = 5, 5
for i in range(grid_H):
    for j in range(grid_W):
        neuron_positions.append({
            'x': left_gap + j * spacing + 2000,
            'y': top_gap + i * spacing + 1000,
            'layer': layer_id
        })
layer_id += 1
# -----------------------------------------------------------------------------------

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
    """
    Run your calculation in a threadpool (so FastAPI async loop is not blocked)
    and set `fire_data` and data_event when done.
    """
    global fire_data
    print("Received params:", data)
    # run calculation_function in threadpool - it should return dict with In_fires, E_fires, I_fires, Out_fires (numpy/torch)
    fire_data = await run_in_threadpool(calculation_function, data)
    data_event.set()
    return {"status": "finish computing"}

@app.post("/shutdown")
async def shutdown():
    shutdown_event.set()
    return {"status": "shutdown triggered"}

# ---------------------- MJPEG streaming endpoint -----------------------------------
def _render_frame_to_jpeg_bytes(states, width=3000, height=1500, neuron_positions=None):
    """
    Render one frame (states: 1D array/list of length == len(neuron_positions))
    to JPEG bytes. Uses PIL for drawing.
    """
    # Create white background
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Parameters for drawing
    r_fire = 8      # radius for firing neurons
    r_silent = 5    # radius for silent neurons

    # Colors (RGB)
    color_fire = (220, 20, 60)       # red-ish for firing
    color_silent = (236, 234, 234)   # light gray for silent
    color_bad = (0, 0, 0)            # black for out-of-range

    # Draw neurons
    for idx, neuron in enumerate(neuron_positions):
        x = int(neuron["x"])
        y = int(neuron["y"])
        try:
            val = float(states[idx])
        except Exception:
            val = 0.0

        if val != val or val is None:  # NaN guard
            col = color_bad
            r = r_silent
        elif val > 0:
            col = color_fire
            r = r_fire
        else:
            col = color_silent
            r = r_silent

        bbox = (x - r, y - r, x + r, y + r)
        draw.ellipse(bbox, fill=col, outline=(0, 0, 0))

    # Encode to JPEG in-memory
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)  # quality tradeoff: lower => less bandwidth
    jpeg_bytes = buf.getvalue()
    buf.close()
    return jpeg_bytes

async def mjpeg_generator(fire_data, neuron_positions, fps=20):
    """
    Async generator that yields multipart MJPEG frames. Yields bytes ready
    to stream in a 'multipart/x-mixed-replace; boundary=frame' response.
    """
    boundary = b"--frame\r\n"
    content_type_line = b"Content-Type: image/jpeg\r\n"
    # Extract arrays from fire_data. They might be numpy arrays or torch tensors.
    In_fires = fire_data["In_fires"]
    E_fires = fire_data["E_fires"]
    I_fires = fire_data["I_fires"]
    Out_fires = fire_data["Out_fires"]

    # convert helper: ensure numpy arrays on CPU and plain dtype
    def to_numpy(x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    In_fires = to_numpy(In_fires)
    E_fires = to_numpy(E_fires)
    I_fires = to_numpy(I_fires)
    Out_fires = to_numpy(Out_fires)

    n_frames = In_fires.shape[0]
    # Optionally: guard if shapes mismatch
    total_neurons = len(neuron_positions)

    for t in range(n_frames):
        # Build flat states array in same order your client used
        In_states = In_fires[t].reshape(-1)
        E_states = E_fires[t].reshape(-1)
        I_states = I_fires[t].reshape(-1)
        Out_states = Out_fires[t].reshape(-1)
        states = np.concatenate([In_states, E_states, I_states, Out_states])
        # If mismatch in length, pad/truncate safely
        if states.shape[0] < total_neurons:
            pad = np.zeros(total_neurons - states.shape[0])
            states = np.concatenate([states, pad])
        elif states.shape[0] > total_neurons:
            states = states[:total_neurons]

        jpeg = await run_in_threadpool(_render_frame_to_jpeg_bytes, states, 3000, 1500, neuron_positions)
        header = boundary + content_type_line + (f"Content-Length: {len(jpeg)}\r\n\r\n").encode("utf-8")
        yield header + jpeg + b"\r\n"
        # throttle to approximate desired fps
        await asyncio.sleep(1.0 / fps)

    # After streaming all frames, end -- optionally repeat or close.
    # Here we just stop the generator; the connection will close on client side.
    return

@app.get("/stream")
async def stream_endpoint(request: Request):
    """
    Wait for data_event (computation done), then stream MJPEG frames to client.
    If you want the /stream endpoint to block until next computation, it waits for data_event.
    """
    # Wait until calculation finishes (or shutdown)
    await data_event.wait()

    if fire_data is None:
        return HTMLResponse("No data available", status_code=404)

    # Create the streaming response. Browser will treat it as MJPEG.
    generator = mjpeg_generator(fire_data, neuron_positions, fps=20)
    # Content type for MJPEG
    return StreamingResponse(generator, media_type='multipart/x-mixed-replace; boundary=frame')
