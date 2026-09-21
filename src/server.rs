//! `mmh3 server`, which takes a generation over HTTP and hands back the video it wrote.
//!
//! A form field stands for the option of the same name, so what a request asks for is turned into
//! the arguments the command would have had and handed to the same generation. There is one
//! device, so there is one generation at a time: requests queue, and the answer to a request is
//! the job it made rather than the video, which is not there yet.
//!
//! There is no authentication here. That is somebody else's to add in front, which is why this
//! listens on loopback unless it is told otherwise.

use crate::cli::parse_options;
use axum::extract::{Multipart, Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;

/// Where a server waits when it is not told.
const DEFAULT_LISTEN: &str = "127.0.0.1:8833";
/// The most a field that stands for an option may hold. An option is a number or a sentence, and
/// a prompt is the longest of them by far.
const FIELD_BYTES: u64 = 1 << 20;
/// The most an uploaded file may hold. A reference clip is the largest thing a request carries.
const FILE_BYTES: u64 = 4 << 30;

const USAGE: &str = "usage: mmh3 server [--listen ADDR] [--jobs DIR] [--models DIR] [--worker HOST[:PORT]]... \
                     [--local-worker] [--token FILE] [--vram-budget GB] [--idle-unload SECONDS]";

/// What a generation the server was asked for is doing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Queued,
    Running,
    Done,
    Failed,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Queued => "queued",
            Stage::Running => "running",
            Stage::Done => "done",
            Stage::Failed => "failed",
        }
    }
}

/// One generation the server was asked for.
struct Job {
    state: Stage,
    /// What the generation was told, in the form the command would have been told it.
    arguments: Vec<String>,
    /// Where this job's video was written.
    video: PathBuf,
    /// Why it failed, for a client that asks after it did.
    failure: Option<String>,
    queued: SystemTime,
    started: Option<Instant>,
    seconds: Option<f64>,
}

impl Job {
    fn describe(&self, id: &str) -> serde_json::Value {
        json!({
            "id": id,
            "state": self.state.name(),
            "queued": self.queued.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
            "seconds": self.seconds,
            "error": self.failure,
        })
    }
}

/// Everything the handlers share.
struct Server {
    jobs: Mutex<HashMap<String, Job>>,
    /// Ids in the order they were asked for, which is the order they run in.
    queue: Sender<String>,
    /// What every generation is told before what its own request said, so that a machine's models
    /// directory and its workers are the server's business rather than each request's.
    defaults: Vec<String>,
    directory: PathBuf,
}

/// `mmh3 server [--listen ADDR] [--jobs DIR] ...`, which serves until it is stopped.
pub fn serve(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let allowed = [
        "listen",
        "jobs",
        "models",
        "worker",
        "worker-units",
        "token",
        "vram-budget",
        "idle-unload",
    ];
    let options = parse_options(arguments, &allowed, crate::generation::FLAGS, USAGE)?;
    let listen = options
        .get("listen")
        .map_or(DEFAULT_LISTEN, |listen| *listen)
        .to_owned();
    let directory = match options.get("jobs") {
        Some(directory) => PathBuf::from(directory),
        None => {
            crate::models::cache_file("jobs").ok_or("pass a directory for the jobs with --jobs")?
        }
    };
    std::fs::create_dir_all(&directory)?;

    // Everything but --listen and --jobs is the generation's, and every generation is told it.
    let mut defaults = Vec::new();
    let mut remaining = arguments.iter();
    while let Some(name) = remaining.next() {
        // An option that stands alone carries nothing after it to pair with.
        if crate::generation::FLAGS.contains(&&name[2..]) {
            defaults.push(name.clone());
            continue;
        }
        let Some(value) = remaining.next() else { break };
        if !matches!(name.as_str(), "--listen" | "--jobs") {
            defaults.push(name.clone());
            defaults.push(value.clone());
        }
    }
    keep_the_models(&mut defaults);

    let (queue, waiting) = channel();
    let server = Arc::new(Server {
        jobs: Mutex::new(HashMap::new()),
        queue,
        defaults,
        directory: directory.clone(),
    });

    // The generations run here, one after another, off the runtime that answers the requests. A
    // generation holds the device for minutes and answers nothing while it does.
    let generating = Arc::clone(&server);
    std::thread::Builder::new()
        .name("generate".to_owned())
        .spawn(move || {
            for id in waiting {
                generate(&generating, &id);
            }
        })?;

    let application = Router::new()
        .route("/v1/generations", post(create).get(list))
        .route("/v1/status", get(status))
        .route("/v1/generations/{id}", get(describe))
        .route("/v1/generations/{id}/video", get(video))
        .with_state(server);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(&listen).await?;
        println!("serving on {listen}, jobs in {}", directory.display());
        axum::serve(listener, application).await
    })?;
    Ok(())
}

/// Asks for a worker in this process, so that the models stay loaded between the generations
/// rather than being read for each one.
///
/// A leader reads no model: it hands out the prompt, every step and every chunk of the decode. A
/// run with nothing to hand them to reads them all itself and lets go of them when it ends, which
/// is right for a command and wrong for a server. A worker is what holds models between the runs
/// that use them, with the budget it is held to and the letting go it does when the device has no
/// memory left, so the server asks for one of its own.
///
/// A server told to use machines of its own is left alone: whoever named them meant them.
fn keep_the_models(defaults: &mut Vec<String>) {
    if defaults
        .iter()
        .any(|argument| argument == "--worker" || argument == "--local-worker")
    {
        return;
    }
    defaults.push("--local-worker".to_owned());
}

/// Runs one job to its end, whichever end that is.
fn generate(server: &Server, id: &str) {
    let arguments = {
        let mut jobs = server.jobs.lock().expect("the jobs");
        let Some(job) = jobs.get_mut(id) else {
            return;
        };
        job.state = Stage::Running;
        job.started = Some(Instant::now());
        job.arguments.clone()
    };
    println!("generation {id} begins");
    let generated = crate::generate::run(&arguments, USAGE);
    let mut jobs = server.jobs.lock().expect("the jobs");
    let Some(job) = jobs.get_mut(id) else {
        return;
    };
    job.seconds = job.started.map(|started| started.elapsed().as_secs_f64());
    match generated {
        Ok(()) => {
            job.state = Stage::Done;
            println!("generation {id} wrote {}", job.video.display());
        }
        Err(error) => {
            job.state = Stage::Failed;
            println!("generation {id} failed: {error}");
            job.failure = Some(error.to_string());
        }
    }
}

/// Takes a generation, and answers with the job rather than with the video, which is not there
/// yet. The form's fields are the options of `mmh3 generate`, and a field with a file name is
/// written beside the job and stands for the path it was written to.
async fn create(
    State(server): State<Arc<Server>>,
    mut form: Multipart,
) -> Result<Response, Failure> {
    let id = next_id();
    let directory = server.directory.join(&id);
    tokio::fs::create_dir_all(&directory).await?;
    let video = directory.join("video.mp4");
    let mut arguments = server.defaults.clone();
    arguments.push("--out".to_owned());
    arguments.push(video.to_string_lossy().into_owned());

    while let Some(mut field) = form.next_field().await.map_err(Failure::form)? {
        let Some(name) = field.name().map(str::to_owned) else {
            return Err(Failure::asked("a form field with no name"));
        };
        if !crate::generation::OPTIONS.contains(&name.as_str()) {
            return Err(Failure::asked(&format!(
                "{name} is not an option of a generation"
            )));
        }
        let value = match field.file_name().map(str::to_owned) {
            // A file stands for the path it is written to, which is this job's own directory.
            Some(filename) => {
                let path = directory.join(named(&filename)?);
                let mut file = tokio::fs::File::create(&path).await?;
                let mut written = 0;
                while let Some(chunk) = field.chunk().await.map_err(Failure::form)? {
                    written += chunk.len() as u64;
                    if written > FILE_BYTES {
                        return Err(Failure::asked("a file longer than this takes"));
                    }
                    file.write_all(&chunk).await?;
                }
                file.flush().await?;
                path.to_string_lossy().into_owned()
            }
            None => {
                let bytes = field.bytes().await.map_err(Failure::form)?;
                if bytes.len() as u64 > FIELD_BYTES {
                    return Err(Failure::asked("a field longer than this takes"));
                }
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| Failure::asked("a form field that is not text"))?
            }
        };
        arguments.push(format!("--{name}"));
        arguments.push(value);
    }

    let job = Job {
        state: Stage::Queued,
        arguments,
        video,
        failure: None,
        queued: SystemTime::now(),
        started: None,
        seconds: None,
    };
    let described = job.describe(&id);
    server
        .jobs
        .lock()
        .expect("the jobs")
        .insert(id.clone(), job);
    if server.queue.send(id.clone()).is_err() {
        return Err(Failure::gone("nothing is generating any more"));
    }
    Ok((
        StatusCode::ACCEPTED,
        [(header::LOCATION, format!("/v1/generations/{id}"))],
        Json(described),
    )
        .into_response())
}

/// Every generation this server has been asked for, the most recent first.
async fn list(State(server): State<Arc<Server>>) -> Response {
    let jobs = server.jobs.lock().expect("the jobs");
    let mut described: Vec<(SystemTime, serde_json::Value)> = jobs
        .iter()
        .map(|(id, job)| (job.queued, job.describe(id)))
        .collect();
    described.sort_by(|(one, _), (other, _)| other.cmp(one));
    let generations: Vec<serde_json::Value> =
        described.into_iter().map(|(_, value)| value).collect();
    Json(json!({ "generations": generations })).into_response()
}

/// What this machine is doing and what it is holding, for a client deciding whether to ask it for
/// anything.
async fn status(State(server): State<Arc<Server>>) -> Response {
    let (generating, queued, generations) = {
        let jobs = server.jobs.lock().expect("the jobs");
        let generating = jobs
            .iter()
            .find(|(_, job)| job.state == Stage::Running)
            .map(|(id, _)| id.clone());
        let queued = jobs
            .values()
            .filter(|job| job.state == Stage::Queued)
            .count();
        (generating, queued, jobs.len())
    };
    Json(json!({
        "generating": generating,
        "queued": queued,
        "generations": generations,
        // None rather than an empty list: a generation has the models, and saying it holds
        // nothing would be saying something else.
        "models": kept(),
        "memory": memory().map(|(free, total)| json!({ "free": free, "total": total })),
    }))
    .into_response()
}

/// What this machine is holding on to, or None while a generation has the models or on a build
/// that holds none of its own.
fn kept() -> Option<Vec<&'static str>> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        crate::resident::kept()
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        None
    }
}

/// What the device says it has left and what it has in all, or None on a build with neither
/// backend, which has no device to ask.
fn memory() -> Option<(u64, u64)> {
    #[cfg(feature = "cuda")]
    {
        let (free, total) = mmh3_cuda::memory_info().ok()?;
        Some((free as u64, total as u64))
    }
    #[cfg(feature = "metal")]
    {
        let (free, total) = mmh3_metal::memory_info().ok()?;
        Some((free as u64, total as u64))
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        None
    }
}

/// What a generation is doing.
async fn describe(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let jobs = server.jobs.lock().expect("the jobs");
    let job = jobs.get(&id).ok_or_else(Failure::unknown)?;
    Ok(Json(job.describe(&id)).into_response())
}

/// The video a generation wrote, once it has written one.
async fn video(
    State(server): State<Arc<Server>>,
    Path(id): Path<String>,
) -> Result<Response, Failure> {
    let (state, path) = {
        let jobs = server.jobs.lock().expect("the jobs");
        let job = jobs.get(&id).ok_or_else(Failure::unknown)?;
        (job.state, job.video.clone())
    };
    match state {
        Stage::Done => Ok((
            [(header::CONTENT_TYPE, "video/mp4")],
            tokio::fs::read(&path).await?,
        )
            .into_response()),
        Stage::Failed => Err(Failure::gone("that generation failed")),
        _ => Err(Failure::waiting()),
    }
}

/// The file name a client gave, taken down to a name. A name with a directory in it is a name
/// meant to land somewhere other than where it was sent.
fn named(filename: &str) -> Result<String, Failure> {
    let name = std::path::Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    match name.is_empty() || name.starts_with('.') {
        true => Err(Failure::asked("a file name that names no file")),
        false => Ok(name.to_owned()),
    }
}

/// An id no other job of this server has. The clock says which run of the server it belongs to
/// and the count says which job of that run, so a client holding an id from a server that has
/// been restarted is told there is no such generation rather than handed somebody else's.
fn next_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNT: AtomicU64 = AtomicU64::new(0);
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:04x}",
        started.as_secs(),
        COUNT.fetch_add(1, Ordering::Relaxed) & 0xffff
    )
}

/// What a request is answered with when it is not answered with what it asked for.
struct Failure(StatusCode, String);

impl Failure {
    fn asked(message: &str) -> Self {
        Failure(StatusCode::BAD_REQUEST, message.to_owned())
    }

    fn form(error: axum::extract::multipart::MultipartError) -> Self {
        Failure(StatusCode::BAD_REQUEST, error.to_string())
    }

    fn unknown() -> Self {
        Failure(StatusCode::NOT_FOUND, "no such generation".to_owned())
    }

    fn waiting() -> Self {
        Failure(
            StatusCode::CONFLICT,
            "that generation has not written a video yet".to_owned(),
        )
    }

    fn gone(message: &str) -> Self {
        Failure(StatusCode::CONFLICT, message.to_owned())
    }
}

impl From<std::io::Error> for Failure {
    fn from(error: std::io::Error) -> Self {
        Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}
