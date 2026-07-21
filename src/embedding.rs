use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::db::Database;
use crate::embed_remote::RemoteEmbedder;
use crate::error::MemoryError;
use crate::memory::MemoryType;

/// Target embedding dimension (MRL truncation from 1024)
const EMBED_DIM: usize = 256;

const QUERY_PREFIX: &str = "search_query: ";
const DOCUMENT_PREFIX: &str = "search_document: ";

const HF_REPO_ID: &str = "onnx-community/mdbr-leaf-ir-ONNX";

const ENV_BACKEND: &str = "ENGRAM_EMBED_BACKEND";
const ENV_URL: &str = "ENGRAM_EMBED_URL";
const ENV_MODEL: &str = "ENGRAM_EMBED_MODEL";
const ENV_MODEL_VERSION: &str = "ENGRAM_EMBED_MODEL_VERSION";

const DEFAULT_URL: &str = "http://127.0.0.1:8080";
const DEFAULT_MODEL: &str = "mdbr-leaf-ir";
const DEFAULT_MODEL_VERSION: &str = "1";

/// `model_version` stamped on embeddings produced in-process via `fastembed`.
const IN_PROCESS_MODEL_VERSION: &str = "mdbr-leaf-ir-q8-d256";
/// `model_version` stamped on embeddings produced via the `onnxruntime-server` HTTP
/// backend. Deliberately distinct from `IN_PROCESS_MODEL_VERSION`: a different ONNX
/// Runtime build (this crate's bundled `ort`/pyke.io static lib vs. whatever build
/// `onnxruntime-server` links against) can produce slightly different numeric output
/// for the same quantized model, so vectors from the two backends are not safe to mix
/// in the same similarity search. The distinct version string is what lets
/// `check_model_version_guard` detect and refuse that mix, and what `engram-cli
/// reembed` migrates between.
const ONNXRUNTIME_SERVER_MODEL_VERSION: &str = "mdbr-leaf-ir-q8-d256-ortserver";

enum Backend {
    InProcess(Arc<Mutex<TextEmbedding>>),
    OnnxRuntimeServer(Arc<RemoteEmbedder>),
}

pub struct EmbeddingService {
    backend: Backend,
    model_version: String,
}

impl Clone for EmbeddingService {
    /// Clone shares the underlying ONNX model (or remote client) via `Arc`, so
    /// no model reload / re-handshake occurs.
    fn clone(&self) -> Self {
        let backend = match &self.backend {
            Backend::InProcess(m) => Backend::InProcess(Arc::clone(m)),
            Backend::OnnxRuntimeServer(c) => Backend::OnnxRuntimeServer(Arc::clone(c)),
        };
        Self {
            backend,
            model_version: self.model_version.clone(),
        }
    }
}

struct ModelFiles {
    onnx_model: PathBuf,
    onnx_data: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
    special_tokens_map: PathBuf,
    tokenizer_config: PathBuf,
}

/// Download model files from HuggingFace and return paths to each cached file.
fn download_model_files() -> Result<ModelFiles, MemoryError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| MemoryError::Embedding(format!("Failed to init HF API: {}", e)))?;
    let repo = api.model("onnx-community/mdbr-leaf-ir-ONNX".to_string());

    let map_err = |file: &str| {
        let file = file.to_string();
        move |e: hf_hub::api::sync::ApiError| {
            MemoryError::Embedding(format!("Failed to download {}: {}", file, e))
        }
    };

    Ok(ModelFiles {
        onnx_model: repo
            .get("onnx/model_quantized.onnx")
            .map_err(map_err("onnx/model_quantized.onnx"))?,
        onnx_data: repo
            .get("onnx/model_quantized.onnx_data")
            .map_err(map_err("onnx/model_quantized.onnx_data"))?,
        tokenizer: repo
            .get("tokenizer.json")
            .map_err(map_err("tokenizer.json"))?,
        config: repo.get("config.json").map_err(map_err("config.json"))?,
        special_tokens_map: repo
            .get("special_tokens_map.json")
            .map_err(map_err("special_tokens_map.json"))?,
        tokenizer_config: repo
            .get("tokenizer_config.json")
            .map_err(map_err("tokenizer_config.json"))?,
    })
}

fn load_model(files: &ModelFiles) -> Result<TextEmbedding, MemoryError> {
    let onnx_file = std::fs::read(&files.onnx_model)
        .map_err(|e| MemoryError::Embedding(format!("Failed to read ONNX model: {}", e)))?;
    let onnx_data = std::fs::read(&files.onnx_data)
        .map_err(|e| MemoryError::Embedding(format!("Failed to read ONNX data: {}", e)))?;

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: std::fs::read(&files.tokenizer)
            .map_err(|e| MemoryError::Embedding(format!("Failed to read tokenizer: {}", e)))?,
        config_file: std::fs::read(&files.config)
            .map_err(|e| MemoryError::Embedding(format!("Failed to read config: {}", e)))?,
        special_tokens_map_file: std::fs::read(&files.special_tokens_map)
            .map_err(|e| MemoryError::Embedding(format!("Failed to read special tokens: {}", e)))?,
        tokenizer_config_file: std::fs::read(&files.tokenizer_config).map_err(|e| {
            MemoryError::Embedding(format!("Failed to read tokenizer config: {}", e))
        })?,
    };

    let user_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
        .with_pooling(Pooling::Cls)
        .with_external_initializer("model_quantized.onnx_data".to_string(), onnx_data);

    let options = InitOptionsUserDefined::new();

    TextEmbedding::try_new_from_user_defined(user_model, options)
        .map_err(|e| MemoryError::Embedding(format!("Failed to load model: {}", e)))
}

/// Truncate embedding to target dimension and L2-normalize.
fn truncate_and_normalize(embedding: &mut Vec<f32>) {
    embedding.truncate(EMBED_DIM);
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }
}

impl EmbeddingService {
    pub fn new() -> Result<Self, MemoryError> {
        match std::env::var(ENV_BACKEND).ok().as_deref() {
            Some("onnxruntime-server") => Self::new_onnxruntime_server(),
            _ => Self::new_in_process(),
        }
    }

    fn new_in_process() -> Result<Self, MemoryError> {
        let files = download_model_files()?;
        let model = load_model(&files)?;

        Ok(Self {
            backend: Backend::InProcess(Arc::new(Mutex::new(model))),
            model_version: IN_PROCESS_MODEL_VERSION.to_string(),
        })
    }

    fn new_onnxruntime_server() -> Result<Self, MemoryError> {
        let url = std::env::var(ENV_URL).unwrap_or_else(|_| DEFAULT_URL.to_string());
        let model = std::env::var(ENV_MODEL).unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let version =
            std::env::var(ENV_MODEL_VERSION).unwrap_or_else(|_| DEFAULT_MODEL_VERSION.to_string());

        let tokenizer_files = crate::embed_remote::download_tokenizer_files(HF_REPO_ID)?;
        let tokenizer = crate::embed_remote::build_tokenizer(&tokenizer_files)?;
        let client = RemoteEmbedder::connect(&url, &model, &version, tokenizer)?;

        Ok(Self {
            backend: Backend::OnnxRuntimeServer(Arc::new(client)),
            model_version: ONNXRUNTIME_SERVER_MODEL_VERSION.to_string(),
        })
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// Embed raw text without any prefix.
    fn embed_raw(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let mut embedding = match &self.backend {
            Backend::InProcess(model) => {
                let mut model = model.lock().map_err(|e| {
                    MemoryError::Embedding(format!("Failed to acquire lock: {}", e))
                })?;

                let embeddings = model
                    .embed(vec![text], None)
                    .map_err(|e| MemoryError::Embedding(e.to_string()))?;

                embeddings
                    .into_iter()
                    .next()
                    .ok_or_else(|| MemoryError::Embedding("No embedding generated".to_string()))?
            }
            Backend::OnnxRuntimeServer(client) => client.embed_raw(text)?,
        };

        truncate_and_normalize(&mut embedding);
        Ok(embedding)
    }

    /// Embed a search query (adds query prefix for asymmetric retrieval).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let prefixed = format!("{}{}", QUERY_PREFIX, text);
        self.embed_raw(&prefixed)
    }

    /// Embed a memory for storage (adds document prefix for asymmetric retrieval).
    pub fn embed_memory(
        &self,
        memory_type: MemoryType,
        content: &str,
    ) -> Result<Vec<f32>, MemoryError> {
        let text = format!("{}{}: {}", DOCUMENT_PREFIX, memory_type.as_str(), content);
        self.embed_raw(&text)
    }

    /// Embed text using the same document prefix as `embed_memory` for a given type.
    ///
    /// Used by `auto_link_handoff_sections` to compare section text against target-type
    /// memory embeddings in a shared vector space.  The prefix matches the one applied
    /// during storage so similarity scores are meaningful across types.
    pub fn embed_memory_text(
        &self,
        memory_type: MemoryType,
        text: &str,
    ) -> Result<Vec<f32>, MemoryError> {
        self.embed_memory(memory_type, text)
    }

    #[allow(dead_code)] // Used by MCP server batch tools
    pub fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, MemoryError> {
        let prefixed: Vec<String> = texts
            .into_iter()
            .map(|t| format!("{}{}", DOCUMENT_PREFIX, t))
            .collect();

        let mut embeddings = match &self.backend {
            Backend::InProcess(model) => {
                let mut model = model.lock().map_err(|e| {
                    MemoryError::Embedding(format!("Failed to acquire lock: {}", e))
                })?;
                model
                    .embed(prefixed, None)
                    .map_err(|e| MemoryError::Embedding(e.to_string()))?
            }
            Backend::OnnxRuntimeServer(client) => prefixed
                .iter()
                .map(|t| client.embed_raw(t))
                .collect::<Result<Vec<Vec<f32>>, MemoryError>>(
            )?,
        };

        for embedding in &mut embeddings {
            truncate_and_normalize(embedding);
        }

        Ok(embeddings)
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Refuse to proceed if `project_id` already has embeddings stored under a
/// `model_version` other than the one this `EmbeddingService` would produce.
///
/// Different embedding backends (or different builds of the same backend, see
/// `ONNXRUNTIME_SERVER_MODEL_VERSION`) are not guaranteed to produce vectors in a
/// comparable space, so silently mixing them in the same project's similarity
/// search would corrupt results without any visible error. This is a hard
/// failure, not a warning: callers must run `engram-cli reembed --confirm` to
/// migrate the project onto the current backend's `model_version` before
/// retrying.
///
/// No-op for empty projects and for projects already fully on the current
/// `model_version`.
pub fn check_model_version_guard(
    db: &Database,
    embedding: &EmbeddingService,
    project_id: &str,
) -> Result<(), MemoryError> {
    let expected = embedding.model_version();
    let stored_versions = db.get_model_versions_for_project(project_id)?;
    let mismatched: Vec<&String> = stored_versions
        .iter()
        .filter(|v| v.as_str() != expected)
        .collect();

    if mismatched.is_empty() {
        return Ok(());
    }

    Err(MemoryError::Embedding(format!(
        "project '{project_id}' has stored embeddings under model_version(s) {:?}, but the \
         configured embedding backend produces '{expected}'. Mixing vector spaces in the same \
         similarity search would silently corrupt results, so refusing to proceed. Run \
         `engram-cli reembed --confirm` to migrate this project's embeddings to '{expected}' first.",
        mismatched
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
    }

    #[test]
    fn test_truncate_and_normalize() {
        let mut v2 = vec![3.0, 4.0];
        let norm = (3.0_f32 * 3.0 + 4.0 * 4.0).sqrt(); // 5.0
        truncate_and_normalize(&mut v2); // won't truncate since len < EMBED_DIM
        // After normalize: [0.6, 0.8]
        assert!((v2[0] - 3.0 / norm).abs() < 0.001);
        assert!((v2[1] - 4.0 / norm).abs() < 0.001);
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "vector length mismatch");
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Self-consistency gate for the `onnxruntime-server` HTTP-backed path.
    ///
    /// This deployment uses `onnxruntime-server` exclusively rather than
    /// switching between it and in-process `fastembed` for the same database,
    /// so matching `fastembed`'s in-process output bit-for-bit is no longer a
    /// requirement (different ONNX Runtime builds -- this crate's bundled
    /// `ort`/pyke.io static lib vs. whatever build `onnxruntime-server` links
    /// against -- are expected to produce slightly different numbers for the
    /// same quantized model; that's now an accepted, moot fact rather than a
    /// blocker; see `ONNXRUNTIME_SERVER_MODEL_VERSION`).
    ///
    /// What must still hold: embedding the same input twice against the same
    /// running server produces the same vector, so that repeated
    /// `memory_store`/`memory_query` calls against one server instance stay
    /// internally comparable. This allows a tiny epsilon for any
    /// non-determinism in the server's own execution (e.g. thread-scheduling
    /// float summation order), but is otherwise a hard equality check.
    ///
    /// Requires a real, already-running `onnxruntime-server` instance serving
    /// the `mdbr-leaf-ir` model. Not run by default -- build and start the
    /// server, then run manually:
    ///
    /// ```sh
    /// git clone https://github.com/kibae/onnxruntime-server.git
    /// cd onnxruntime-server
    /// cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
    /// mkdir -p /tmp/onnx-models/mdbr-leaf-ir/1
    /// cp /path/to/model_quantized.onnx /tmp/onnx-models/mdbr-leaf-ir/1/model.onnx
    /// cp /path/to/model_quantized.onnx_data /tmp/onnx-models/mdbr-leaf-ir/1/
    /// # NOTE: onnxruntime-server resolves the external-data file relative to
    /// # its own current working directory, not the model directory, so start
    /// # it from inside that directory:
    /// cd /tmp/onnx-models/mdbr-leaf-ir/1
    /// /path/to/build/src/standalone/onnxruntime_server --model-dir=/tmp/onnx-models --http-port=8080 &
    /// cd -
    /// cargo test --lib embedding::tests::onnxruntime_server_is_internally_deterministic -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn onnxruntime_server_is_internally_deterministic() {
        // SAFETY: test-only, single-threaded w.r.t. this env var within this test.
        unsafe {
            std::env::set_var(ENV_BACKEND, "onnxruntime-server");
        }

        let service = EmbeddingService::new().expect("failed to connect to onnxruntime-server");

        let short_text = "hello world, this is a short memory";
        // Well over the model's 512-token max_length, forcing truncation.
        let long_text = "the quick brown fox jumps over the lazy dog. ".repeat(200);

        struct Case {
            name: &'static str,
            first: Vec<f32>,
            second: Vec<f32>,
        }

        let cases = vec![
            Case {
                name: "short string, query prefix",
                first: service.embed(short_text).expect("embed short query (1)"),
                second: service.embed(short_text).expect("embed short query (2)"),
            },
            Case {
                name: "short string, document prefix",
                first: service
                    .embed_memory(MemoryType::Fact, short_text)
                    .expect("embed short document (1)"),
                second: service
                    .embed_memory(MemoryType::Fact, short_text)
                    .expect("embed short document (2)"),
            },
            Case {
                name: "long string requiring truncation, query prefix",
                first: service.embed(&long_text).expect("embed long query (1)"),
                second: service.embed(&long_text).expect("embed long query (2)"),
            },
            Case {
                name: "long string requiring truncation, document prefix",
                first: service
                    .embed_memory(MemoryType::Fact, &long_text)
                    .expect("embed long document (1)"),
                second: service
                    .embed_memory(MemoryType::Fact, &long_text)
                    .expect("embed long document (2)"),
            },
        ];

        for case in cases {
            assert_eq!(
                case.first.len(),
                EMBED_DIM,
                "unexpected dim for {}",
                case.name
            );
            let diff = max_abs_diff(&case.first, &case.second);
            let cos = cosine_similarity(&case.first, &case.second);
            eprintln!("{}: max abs diff = {diff}, cosine = {cos}", case.name);
            assert!(
                diff < 1e-5,
                "{}: same input embedded twice against the same server should be \
                 (near-)identical, but max abs diff {diff} exceeds 1e-5\nfirst:  {:?}\nsecond: {:?}",
                case.name,
                &case.first[..8.min(case.first.len())],
                &case.second[..8.min(case.second.len())]
            );
        }
    }
}
