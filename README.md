# HZL-cudaDiffusion

## Browser dashboard

Build the web engine and start the dashboard on the Linux computer that will run the experiment. The web server uses Python 3 and its standard library.

```sh
cmake --build --preset cuda --target hzl-web-engine
python3 web/server.py --host 0.0.0.0 -- --backend cuda --endpoint 300 --horizon 300
```

On the Windows computer, open `http://<linux-ip>:8080/` in Edge, Chrome, or Firefox. Both computers need network access to each other. Use `hostname -I` on Linux to find its addresses. The dashboard shares one experiment across visitors. Use `--host 0.0.0.0` on a trusted local network. The default host is loopback for use on Linux itself.

For a CPU build, build the same target in your CPU build directory and pass `--engine build/cpu/hzl-web-engine -- --backend cpu` to the Python server. Options after `--` go to the experiment. These include `--interval`, `--tiles-x`, `--tiles-y`, `--pattern`, and `--output`.

- **Run** or **Step** advances the current experiment. Select **Manual** to apply temperatures you paint while paused.
- **Preview** forecasts with the current map held fixed over the configured horizon. It leaves the simulation clock where it is.
- **Reset** starts a fresh run with the launch settings. **Save on server** writes the experiment CSV files and metadata to the configured output directory on Linux.

### HTTP API

`GET /api/state` returns the simulation time, grid dimensions, metrics, and row-major fields. The following POST routes accept JSON and return the updated state.

| Route | JSON body |
|---|---|
| `/api/step` | `{"automatic":true}` or `{"automatic":false}` |
| `/api/preview` | `{}` |
| `/api/reset` | `{}` |
| `/api/save` | `{}` |
| `/api/paint` | `{"field":"temperature","x":10,"y":12,"value":850}`. Use `"target"` and a value from 0 to 1 to paint concentration. |
