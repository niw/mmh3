# Server

`mmh3 server` takes a generation over HTTP and keeps its models loaded between the generations it
runs, so a machine asked for one video after another reads the checkpoints once.

```sh
export MMH3_MODELS=$PWD/models

target/release/mmh3 server
```

It waits on `127.0.0.1:8833` and writes each generation into a directory of its own under
`$XDG_CACHE_HOME/mmh3/jobs`. `--listen ADDR` and `--jobs DIR` change either.

There is no authentication. A machine that should answer anybody but itself wants something in
front of this that decides who may ask, which is why the default is loopback.

## Asking for a generation

A form field is the option of the same name, so what `mmh3 generate` takes on the command line
this takes as a field. A field that carries a file is written beside the generation and stands for
the path it was written to.

```sh
curl -X POST http://localhost:8833/v1/generations \
  -F prompt="A red panda sips tea on a sunny wooden porch while birds chirp in the garden." \
  -F steps=4 -F seed=2
```

```json
{
  "id": "6ab1cf3b0000",
  "state": "queued",
  "queued": 1790037819,
  "seconds": null,
  "step": null,
  "steps": null,
  "error": null
}
```

`state` is `queued`, `running`, `done` or `failed`, `seconds` is how long it took once it is one of
the last two, and `error` says why it is `failed`.

The answer is the generation rather than the video, which is not there yet. There is one generation
at a time, which takes every GPU the machine has, and the rest wait their turn.

Reference pictures, sounds and clips go the same way, as repeated fields in the order the prompt
refers to them:

```sh
curl -X POST http://localhost:8833/v1/generations \
  -F prompt="<Picture 1> walks past <Picture 2>." \
  -F reference=@cat.png -F reference=@street.jpg
```

A field that is not an option of a generation is refused by name, and so is a file longer than the
server takes. See [Usage](usage.md) for what the options mean.

## Watching one

```sh
curl -N http://localhost:8833/v1/generations/6ab1cf3b0000/events
```

One event a step, then one for the end, and then the stream closes.

```
data: {"error":null,"id":"6ab1cf3b0000","queued":1790037819,"seconds":null,"state":"running","step":3,"steps":4}

data: {"error":null,"id":"6ab1cf3b0000","queued":1790037819,"seconds":10.9,"state":"done","step":null,"steps":null}
```

Asking again works as well, at `GET /v1/generations/{id}`, and says the same thing.

## Taking the video

```sh
curl http://localhost:8833/v1/generations/6ab1cf3b0000/video -o out.mp4
```

Until the generation is done this answers 409 rather than a video.

## What a server holds

```sh
curl http://localhost:8833/v1/status
```

```json
{
  "generating": "6ab1cf3b0000",
  "generations": 7,
  "memory": {"free": 57481990144, "total": 130660909056},
  "models": ["text_encoder", "video_vae", "audio_vae", "dit"],
  "queued": 1
}
```

`models` is `null` rather than a list while a generation has them, since a machine that is busy is
not a machine holding nothing. `memory` is what the GPUs have between them, and `models` is what
any of them holds.

The models are held by a worker in the server's own process, which is the same worker
`mmh3 worker` runs and holds its models the same way. A generation is a leader and reads no model
itself, so without one there would be nothing to keep: the server lends its cards through that
worker even when a run could have taken every step on one card by itself. A server told to use
machines of its own with `--worker` lends none of its cards unless `--local-worker` says so, as a
run does. So `--vram-budget GB` and `--idle-unload SECONDS` mean here what they mean there. See
[Distributed generation](distributed.md). `--consistent` given to the server holds for every
generation it runs.

On a DGX Spark the models take about 6.4 s to read, and that is what the first generation pays and
the ones after it do not, whatever they generate. Three generations of the same request against a
server just started:

| Request | First | Then |
| --- | ---: | ---: |
| 448×256, one step | 12.3 s | 5.3 s, 5.0 s |
| 1344×768, four steps, [FastH3](fasth3.md) | 88.3 s | 77.0 s, 80.5 s |

So it is the short generation that gains most: reading the models is more than half of the first
one and less than a tenth of the second, where the four steps take 58 s and the video VAE 19 to
22 s however the models arrived.

## Every request

| | |
| --- | --- |
| `POST /v1/generations` | Take a generation, as `multipart/form-data`. Answers 202 and the generation. |
| `GET /v1/generations` | Every generation this server was asked for, the most recent first. |
| `GET /v1/generations/{id}` | What one is doing, with the step it is on while it runs. |
| `GET /v1/generations/{id}/events` | The same as it happens, as server-sent events. |
| `GET /v1/generations/{id}/video` | The video, once there is one. |
| `DELETE /v1/generations/{id}` | Take one away with whatever it wrote. |
| `GET /v1/status` | What this machine is doing and what it is holding. |

A generation that is running is not taken away: `DELETE` answers 409 and leaves it. Stopping one
half way is not something this does yet.

## Building it

The server is the `server` Cargo feature, which `make` includes. A build that leaves it out has no
`mmh3 server` and none of the crates behind it. See [Build](build.md).
