const svg = d3.select("#canvas");
const socket = new WebSocket("ws://localhost:8000/ws");
let neurons = [];
let currentStates = [];

// Slider element and value display
const slider = document.getElementById("slider");
const sliderValue = document.getElementById("slider-value");

// Send slider value to Python via HTTP POST
async function sendValueToPython(value) {
    const response = await fetch("/update_param", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value: parseFloat(value) })
    });
    console.log(await response.json());
}

// Event listener for slider input
slider.addEventListener("input", () => {
    const value = slider.value;
    sliderValue.textContent = value;
    sendValueToPython(value);  // Send to Python backend
});


// get the every t step neuron fire states
socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.neurons) {
        neurons = data.neurons;
        currentStates = Array(neurons.length).fill(0);  // Init all to silent
        svg.selectAll("circle")
            .data(neurons)
            .enter()
            .append("circle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", 10)
            .attr("class", d => `neuron silent layer${d.layer}`);
    } else if (data.states) {
        // Smart diff update:
        data.states.forEach((state, i) => {
            if (state !== currentStates[i]) {
                // State changed, update class
                d3.select(svg.selectAll("circle").nodes()[i])
                    .attr("class", `neuron ${state ? 'firing' : 'silent'} layer${neurons[i].layer}`);
                currentStates[i] = state;  // Update local state
            }
        });
    }
    // Update frame number if 'frame' is present
    if (data.frame !== undefined) {
        document.getElementById("frame-number").textContent = data.frame;
    }
};

