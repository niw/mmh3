# Distributed generation

One `mmh3 generate` can spread a generation across other machines running `mmh3 worker`. There
are three things it hands out, and they are worth very different amounts:

| | what moves | what it saves |
| --- | --- | --- |
| the prompt | kilobytes | the 27 GB text encoder never loads on this machine |
| the video and audio decode | hundreds of MB per chunk | about 8 s of a 79 s run |
| every DiT step | tens of GB per step | about a third of a run, and the only piece a slow link cannot have |

Sharing a step is what a worker is for on a machine that could generate by itself: the other two
are cheap enough to take on any link and not worth a second machine on their own. On a machine with
no GPU, which mmh3 builds for, all three together are the run.

## Getting started

On the machine that will help, with its own models directory:

```sh
target/release/mmh3 worker --models /path/to/models
```

It listens on `0.0.0.0:7833`, stays idle until asked, and loads a checkpoint on the first request
that needs it. It keeps what it has loaded for the runs that follow, and lets go of the checkpoint
it has used least when the device has no memory left for the next one, so a second run against the
same worker finds the models already there. `--idle-unload SECONDS` gives the memory back on a machine
that is also somebody's desktop. Then on the machine the user runs:

```sh
target/release/mmh3 generate --out out.mp4 --prompt "..." \
  --worker other-machine.local
```

Repeat `--worker` for several. `--worker HOST` uses port 7833, and `--worker HOST:PORT` names
another.

To see what a worker offers before running a generation:

```sh
target/release/mmh3 worker --probe other-machine.local
```

That prints its backend, device, memory, capabilities and transports, the checkpoints it holds, the
round trip, and the bandwidth at three sizes. It is the quickest way to tell a link that will carry
a shared step from one that will not.

## What a worker does

A worker answers requests and holds nothing between them except the weights it has loaded. It never
talks to the user. Each unit is offered separately, and a worker advertises only what its backend
can actually do.

**The prompt.** A worker holding the text encoder encodes the prompt and answers with the hidden
states, so the leader never loads those 27 GB. This matters more on a 64 GB Mac than on a machine
that had the memory anyway.

**The video and the audio.** The leader splits a decode by temporal chunk and hands every chunk
out, evenly between the workers. The chunks are still asked for in order and the temporal tail
still carries from one to the next, so the seams are what a single machine produces. What the
leader does is put the canvases together and encode them, which wants the plan and the buffers and
none of the weights that made them. A canvas that comes back the wrong length for the latent is
refused and that chunk decodes locally.

**A DiT step.** This is the one that needs a session and a fast link. The sequence is split across
the ranks, Ulysses style: everything in a block except attention works a token at a time and needs
no communication, and attention is handled by exchanging the block's inputs so that each rank holds
the whole sequence for its own heads, attends, and exchanges the result back. Two exchanges per
block, fifty blocks. At 768p across two machines that is about 28 GB a step.

A rank projects only the heads it will attend, so the work that does not shrink with its share stays
small. Ranks trade directly with each other rather than through the leader.

**The leader is not one of the ranks.** It relays the rendezvous, hands each step out and puts the
parts back together, and runs no block: the machines that do the work are the workers, and nothing
else.

So a run that shares its steps **lends this machine a rank of its own**. A worker starts here on
loopback, takes a session like any other, and nothing in the run knows the difference, so naming
one machine still splits a step in two. It goes last in the list, so `--shard-dit N` still means
the first N machines named, and `--shard-dit 0` starts none.

**A leader reads no weights at all.** For the DiT it takes the shape from the checkpoint header,
which costs the header pages and nothing, and assembles the parts from that. For the video VAE it
takes the tiling plan from the header the same way, and putting the canvases together needs no
weight either. A worker holding the text encoder spares it those 27 GB, and one that decodes the
soundtrack the rest. So what a leader holds is the plan, the schedule, the noise, the canvas
buffers and the file, and the models are read by the machines that compute with them.

A run with no worker is not a leader: it reads what it needs and generates by itself, which is what
`mmh3 generate` does on one machine.

**So a machine with no GPU at all can run `mmh3 generate`**, built with neither backend. It lends
itself nothing, and the prompt, every step, the video and the soundtrack happen on the workers
while this process holds the plan, the schedule and the file. Blending a decode is the one thing
such a machine cannot do, so it asks for the video rather than its chunks and the worker that
decodes it blends it too. Either backend can be the machine on the other end.

## Choosing how much each machine takes

An even split is right only when the machines match. The barrier at each exchange is a hard
rendezvous, so a rank given more than it can carry holds every other rank at every block.

A machine measures itself once at the two kinds of work a block is made of, the products and the
attention, and reports both in the handshake. The result is kept in
`$XDG_CACHE_HOME/mmh3/speed.txt` (or `~/.cache/mmh3/speed.txt`) with the device and the precision it
was measured at, so a restart does not pay for it again, and a run on another precision measures
again rather than reusing a number that means something else.

The leader cuts the two axes by their own numbers: the tokens follow the products, since the
projections and the MLP work a token at a time, and the heads follow the attention, since a rank
attends its own heads over the whole sequence however few tokens it carries. A machine can be
unevenly good at the two, which is the usual case between a discrete GPU and an integrated one, and
would otherwise be given a share that suits neither.

A machine that measures no attention is priced by its products on both axes. A machine that measures
nothing at all takes no share of a step while others did measure, since an equal share on no
evidence is the case that stalls everybody. It is still asked for the prompt and for chunks. Where
no machine measures anything, the shares stay even.

## Transports

The exchange follows whatever path there is, and RDMA is an accelerator rather than a requirement.

- **RoCE**, when both ends have a port and a route. A region is registered on every active port of
  the device and a transfer is cut into a piece per path, which on hardware whose NIC hangs off two
  PCIe links is most of the available bandwidth rather than half of it.
- **TCP sockets** otherwise. Every pair of ranks opens a socket of its own, reported through the
  rendezvous the leader runs, so no payload passes through the leader. The lower rank of a pair
  calls and the higher answers, and a rank takes every call before it makes any.

A machine with no RoCE port, which is every Mac and most of what else could take a share, uses the
socket path and is otherwise an ordinary rank. `mmh3 worker --transport socket` keeps a port to
itself so that the socket path can be exercised between two machines that both have one.

The barriers always go over the socket. A connection carries payloads and says nothing about when
they arrive, so it cannot stand in for the thing that orders two ranks.

## Whom it is worth it to

Measured at 640×384, dense attention, the same prompt and seed throughout, on a DGX Spark and an
M4 Max Mac:

| | a step |
| --- | --- |
| the Spark alone | 2.0 s, flat |
| the Mac alone | 58 to 92 s across four steps, climbing |
| the Mac as a rank beside the Spark | 23 s |
| **the Mac handing every step to the Spark** | **2.2 s** |

**A machine does best by the run when it takes none of it.** The third row is the Mac as a rank: a
barrier is a hard rendezvous, so a pair runs at the slower rank plus the wire, and the Mac drags the
Spark down to thirteen times its own time while gaining only three or four times its own. The fourth
row is the Mac running no block at all, and it is twenty-five times faster than the Mac alone while
costing the Spark 0.2 s a step, which is what handing out a step and putting it back together takes.

Neither of those last two numbers can be read from one machine. 2.2 s beside 2.0 s is what says the
coordination costs 0.2 s. On its own it says only that a step took 2.2 s.

The Mac also stops reading 45 GB it never needed: 20 GB of DiT, which a machine that runs no block
does not want, and 25 GB of text encoder, which the Spark already held. That is a larger change than
the clock shows, since the wall time of the same run moved only from 53 s to 51 s.

Between two DGX Sparks at 200 GbE, splitting a 768p step takes about a third off a run. A worker
that only encodes the prompt and decodes chunks saves 9% of the same run.

## What falls back, and what does not

**What nobody offers, the leader does.** A worker that cannot be reached, that does not serve a
capability, or that holds no checkpoint for it is passed over, and the leader reads what it needs
and does that piece itself. Naming a worker is asking, not telling.

**What somebody drops ends the run.** A worker that answered the handshake and then timed out,
disconnected or returned an error stops the generation, and the run names the machine and the
reason. It is not worked around. A rank that fails during the steps already ends a run, since
there is no halfway through a step to fall back from, so falling back elsewhere only ever covered
the short stretches either side of them, at the price of a leader that had to be able to do
everything. A canvas that arrives and does not fit the latent is a different thing: it was
answered, it cannot be used, and that chunk decodes locally.

A build with no backend has nothing to fall back to at all, so it says which option would have
avoided what it reached instead of answering with an empty tensor.

A run refuses to share a step, and keeps the DiT locally, when no worker offers `dit_shard`, when
the DiT file cannot be read, when an adapter this run uses cannot be found on a worker, or when a
rank cannot reach another rank by either a socket or a connection. Refusing names both ends.

Both machines must speak the same protocol version, which they check in the handshake and which both
ends report on a mismatch. Two builds that disagree refuse each other rather than waiting at the
first exchange, which is what they did before and which looks exactly like one slow machine.

## Determinism

Within one backend, a shared run is reproducible: the same machines with the same cut produce the
same video every time, and the same cut reached two different ways produces the same video. Ulysses
concatenates per-head outputs and never reduces across heads, so which machine computed which head
is invisible, and the exchange carries a block's input in the INT8 its layers already consume.

It is not always what one machine produces on its own, though. Splitting the sequence changes the
shape of every GEMM in a block, and the algorithm kept for a shape is the fastest one timed for it,
so two shapes can round differently. Split in two, a 448×256 run over two steps came out different
from the same run unsplit. A step handed to one rank, which splits nothing, came out identical to
it.

Across backends it does not hold, and cannot: a Metal rank packs as halves what a CUDA rank carries
as INT8, and one value crossing a rounding edge moves most of the velocity fifty blocks later. A
mixed run is a different video rather than a noisier one, and is reproducible only against itself.

## Limits

- A Metal rank refuses an attention precision other than its own, a merged LoRA patch, video
  sparse attention, and a DiT that carries VSA gate projections. All four are answered when the
  session opens and each says which, so the leader keeps the run and shares nothing: it costs a
  machine rather than a run.
- The VSA gates are a property of the checkpoint rather than of the run. A DiT carrying them,
  which is any run with the FastH3 patch, cannot be shared with a Metal rank whatever
  `--attention` names.
- A Metal build refuses sparse attention, a merged LoRA and an attention precision other than its
  own only when it will run the steps itself. Handing every step out, it can ask for any of them,
  which is how a Mac runs FastH3 on a machine that has the kernels for it.
- A share is cut once when the session opens and does not move between steps.
- A run that meant to share and could not reads the DiT after the refusal rather than before it,
  so falling back costs the load it had skipped.
- The leader holds the DiT's shape but a leader that hands out every step never uploads its weights.

## Options

On the worker:

- `--listen ADDR` (default `0.0.0.0:7833`): Where to wait.
- `--models DIR`: The models directory, or `MMH3_MODELS`.
- `--token FILE`: A shared secret, matched against the leader's.
- `--transport auto|socket` (default `auto`): `socket` keeps this machine's RoCE port to itself.
- `--vram-budget GB` (default: as much as the device gives): Hold the worker to that much device
  memory, failing an allocation past it as a device that small would.
- `--idle-unload SECONDS` (default off): Let go of every model after that long with nothing to do.
- `--probe HOST[:PORT]`: Report on another worker instead of serving.

On the leader:

- `--worker HOST[:PORT]`: Name a machine to spread the run across, repeated for several.
- `--shard-dit N` (default: every worker that can): Share every step across at most N workers,
  counting the one this machine lends itself, which comes after the ones `--worker` names.
  `--shard-dit 0` keeps the DiT here and lends nothing.
- `--token FILE`: The shared secret.
- `--vram-budget GB`: As on the worker, for what this machine reads itself.

## Security

The link between these machines is assumed to be private, so the default is no authentication.
`--token FILE` sends a shared secret in the handshake and a worker whose token does not match
refuses everything. There is no TLS: a link that needs encryption needs a tunnel, not a hand-rolled
handshake.
