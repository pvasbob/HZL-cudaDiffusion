const get = (id) => document.getElementById(id);
const fields = ["concentration", "target", "forecast", "temperature", "baseline", "reconstructed"];
const palette = [[20, 20, 52], [53, 56, 120], [42, 129, 144], [83, 198, 119], [249, 230, 92]];
let snapshot;
let busy = false;
let running = false;
let timer;

function paintField(field) {
  const canvas = get(field);
  const {width, height, lower, upper} = snapshot;
  canvas.width = width;
  canvas.height = height;
  canvas.style.aspectRatio = width + "/" + height;
  const context = canvas.getContext("2d");
  const values = snapshot[field];
  if (!values.length) return;
  const pixels = context.createImageData(width, height);
  for (let i = 0; i < values.length; i++) {
    const value = field === "temperature" ? (values[i] - lower) / (upper - lower) : values[i];
    const index = Math.max(0, Math.min(1, value)) * (palette.length - 1);
    const left = Math.floor(index);
    const right = Math.min(left + 1, palette.length - 1);
    for (let channel = 0; channel < 3; channel++) {
      pixels.data[4*i + channel] = palette[left][channel] +
        (palette[right][channel] - palette[left][channel]) * (index - left);
    }
    pixels.data[4*i + 3] = 255;
  }
  context.putImageData(pixels, 0, 0);
}

const seconds = (value) => value.toLocaleString(undefined, {maximumFractionDigits: 3}) + " s";

function render() {
  for (const field of fields) paintField(field);
  get("clock").textContent = seconds(snapshot.time);
  get("endpoint").textContent = seconds(snapshot.endpoint);
  get("error").textContent = snapshot.targetError.toExponential(3);
  get("baseError").textContent = snapshot.baselineError.toExponential(3);
  get("forecastTime").textContent = snapshot.forecast.length
    ? seconds(snapshot.forecastStart) + " to " + seconds(snapshot.forecastEnd)
    : "Press Preview to look ahead " + seconds(snapshot.horizon);
  get("tempRange").textContent = snapshot.lower + "–" + snapshot.upper +
    " K · " + snapshot.tilesX + " × " + snapshot.tilesY + " control tiles";
}

function controls() {
  get("run").textContent = running ? "Pause" : "Run";
  get("run").disabled = !snapshot || (!running && (busy || snapshot.finished));
  get("step").disabled = !snapshot || busy || running || snapshot.finished;
  for (const id of ["preview", "reset", "save", "mode"]) {
    get(id).disabled = !snapshot || busy || running;
  }
}

async function request(path, body) {
  busy = true;
  controls();
  try {
    const options = body === undefined ? {} : {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
    };
    const response = await fetch("/api/" + path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    snapshot = result;
    if (snapshot.finished) running = false;
    render();
    get("status").textContent = path === "save" ? "Results saved on the Linux server."
      : snapshot.finished ? "Endpoint reached" : running ? "Running" : "Paused";
  } catch (error) {
    running = false;
    get("status").textContent = error.message;
  } finally {
    busy = false;
    controls();
  }
}

const step = () => request("step", {automatic: get("mode").value === "auto"});
async function play() {
  if (!running) return;
  await step();
  if (running) timer = setTimeout(play, 100);
}
get("run").onclick = () => {
  running = !running;
  controls();
  get("status").textContent = running ? "Running" : "Paused";
  if (running) play();
  else clearTimeout(timer);
};
get("step").onclick = step;
for (const action of ["preview", "reset", "save"]) {
  get(action).onclick = () => request(action, {});
}
for (const [field, brush] of [["temperature", "tempBrush"], ["target", "targetBrush"]]) {
  get(field).addEventListener("pointerdown", (event) => {
    if (!snapshot || running || busy || event.button !== 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = Math.min(snapshot.width - 1, Math.floor((event.clientX - bounds.left) * snapshot.width / bounds.width));
    const y = Math.min(snapshot.height - 1, Math.floor((event.clientY - bounds.top) * snapshot.height / bounds.height));
    const value = get(brush).valueAsNumber;
    request("paint", {field, x, y, value});
  });
}
await request("state");
if (snapshot) {
  get("tempBrush").min = snapshot.lower;
  get("tempBrush").max = snapshot.upper;
  get("tempBrush").value = Math.min(snapshot.upper, Math.max(snapshot.lower, 850));
}
