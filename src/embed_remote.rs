//! HTTP client for an externally-run `onnxruntime-server` instance
//! (https://github.com/kibae/onnxruntime-server), used when
//! `ENGRAM_EMBED_BACKEND=onnxruntime-server` is set.
//!
//! `onnxruntime-server` runs the raw ONNX forward pass only -- no tokenization
//! or pooling. All of that stays client-side here, replicating exactly what
//! `fastembed` does in-process for this model (tokenize -> run -> CLS-pool ->
//! full-width L2 normalize -> truncate_and_normalize), so output is
//! bit-identical to the in-process path.

use serde_json::{Value, json};
use std::time::Duration;
use tokenizers::{AddedToken, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::error::MemoryError;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_RETRIES: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(150);

pub struct TokenizerFiles {
    tokenizer: std::path::PathBuf,
    config: std::path::PathBuf,
    special_tokens_map: std::path::PathBuf,
    tokenizer_config: std::path::PathBuf,
}

/// Download only the tokenizer-related files (not the ONNX model itself --
/// that's loaded server-side by `onnxruntime-server`).
pub fn download_tokenizer_files(repo_id: &str) -> Result<TokenizerFiles, MemoryError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| MemoryError::Embedding(format!("Failed to init HF API: {}", e)))?;
    let repo = api.model(repo_id.to_string());

    let map_err = |file: &str| {
        let file = file.to_string();
        move |e: hf_hub::api::sync::ApiError| {
            MemoryError::Embedding(format!("Failed to download {}: {}", file, e))
        }
    };

    Ok(TokenizerFiles {
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

/// Replicates `fastembed::common::load_tokenizer` exactly (padding strategy,
/// truncation length, special tokens) so tokenization matches the in-process path.
pub fn build_tokenizer(files: &TokenizerFiles) -> Result<Tokenizer, MemoryError> {
    let embed_err = |e: std::io::Error| MemoryError::Embedding(e.to_string());

    let tokenizer_bytes = std::fs::read(&files.tokenizer).map_err(embed_err)?;
    let config_bytes = std::fs::read(&files.config).map_err(embed_err)?;
    let special_tokens_bytes = std::fs::read(&files.special_tokens_map).map_err(embed_err)?;
    let tokenizer_config_bytes = std::fs::read(&files.tokenizer_config).map_err(embed_err)?;

    let config: Value = serde_json::from_slice(&config_bytes)?;
    let special_tokens_map: Value = serde_json::from_slice(&special_tokens_bytes)?;
    let tokenizer_config: Value = serde_json::from_slice(&tokenizer_config_bytes)?;

    let mut tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|e| MemoryError::Embedding(format!("failed to parse tokenizer.json: {e}")))?;

    let model_max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .ok_or_else(|| {
            MemoryError::Embedding("missing model_max_length in tokenizer_config.json".to_string())
        })? as f32;
    const DEFAULT_MAX_LENGTH: usize = 512;
    let max_length = DEFAULT_MAX_LENGTH.min(model_max_length as usize);
    let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
    let pad_token = tokenizer_config["pad_token"]
        .as_str()
        .ok_or_else(|| {
            MemoryError::Embedding("missing pad_token in tokenizer_config.json".to_string())
        })?
        .into();

    let mut tokenizer = tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| MemoryError::Embedding(format!("failed to configure tokenizer: {e}")))?
        .clone();

    if let Value::Object(root_object) = special_tokens_map {
        for (_, value) in root_object.iter() {
            if let Some(content) = value.as_str() {
                tokenizer.add_special_tokens(&[AddedToken {
                    content: content.into(),
                    special: true,
                    ..Default::default()
                }]);
            } else if value.is_object()
                && let (
                    Some(content),
                    Some(single_word),
                    Some(lstrip),
                    Some(rstrip),
                    Some(normalized),
                ) = (
                    value["content"].as_str(),
                    value["single_word"].as_bool(),
                    value["lstrip"].as_bool(),
                    value["rstrip"].as_bool(),
                    value["normalized"].as_bool(),
                )
            {
                tokenizer.add_special_tokens(&[AddedToken {
                    content: content.into(),
                    special: true,
                    single_word,
                    lstrip,
                    rstrip,
                    normalized,
                }]);
            }
        }
    }

    Ok(tokenizer.into())
}

/// Mirrors `fastembed::common::normalize` verbatim (full-width L2 normalize with
/// the same `1e-12` epsilon). `fastembed`'s `TextEmbedding::embed` normalizes at
/// full width internally before our client-side `truncate_and_normalize` runs;
/// this replicates that step for the HTTP-backed path.
fn fastembed_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|val| val * val).sum::<f32>().sqrt();
    let epsilon = 1e-12;
    v.iter().map(|&val| val / (norm + epsilon)).collect()
}

pub struct RemoteEmbedder {
    agent: ureq::Agent,
    url: String,
    model: String,
    version: String,
    tokenizer: Tokenizer,
    needs_token_type_ids: bool,
}

fn session_path(url: &str, model: &str, version: &str) -> String {
    format!("{url}/api/sessions/{model}/{version}")
}

impl RemoteEmbedder {
    pub fn connect(
        url: &str,
        model: &str,
        version: &str,
        tokenizer: Tokenizer,
    ) -> Result<Self, MemoryError> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(CONNECT_TIMEOUT)
            .timeout_read(READ_TIMEOUT)
            .timeout_write(READ_TIMEOUT)
            .build();

        let inputs = ensure_session(&agent, url, model, version)?;
        let needs_token_type_ids = inputs.contains_key("token_type_ids");

        Ok(Self {
            agent,
            url: url.to_string(),
            model: model.to_string(),
            version: version.to_string(),
            tokenizer,
            needs_token_type_ids,
        })
    }

    pub fn embed_raw(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| MemoryError::Embedding(format!("tokenize failed: {e}")))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|x| *x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|x| *x as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|x| *x as i64).collect();

        let mut body = json!({
            "input_ids": [ids],
            "attention_mask": [mask],
        });
        if self.needs_token_type_ids {
            body["token_type_ids"] = json!([type_ids]);
        }

        let response =
            execute_with_retry(&self.agent, &self.url, &self.model, &self.version, &body)?;

        let last_hidden_state = response.get("last_hidden_state").ok_or_else(|| {
            MemoryError::Embedding(
                "onnxruntime-server response missing 'last_hidden_state' tensor".to_string(),
            )
        })?;

        // Shape: [batch, seq_len, hidden]. CLS-pool = batch 0, sequence position 0.
        let pooled: Vec<f32> = last_hidden_state
            .as_array()
            .and_then(|batches| batches.first())
            .and_then(|seq| seq.as_array())
            .and_then(|seq| seq.first())
            .and_then(|cls| cls.as_array())
            .ok_or_else(|| {
                MemoryError::Embedding(
                    "onnxruntime-server 'last_hidden_state' has unexpected shape".to_string(),
                )
            })?
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| {
                        MemoryError::Embedding("non-numeric value in last_hidden_state".to_string())
                    })
                    .map(|f| f as f32)
            })
            .collect::<Result<Vec<f32>, MemoryError>>()?;

        Ok(fastembed_normalize(&pooled))
    }
}

/// GET the session to check it exists; if not, create it. A 409 from create
/// means another caller raced us to create it -- also fine.
fn ensure_session(
    agent: &ureq::Agent,
    url: &str,
    model: &str,
    version: &str,
) -> Result<serde_json::Map<String, Value>, MemoryError> {
    let get_session = || {
        agent
            .get(&session_path(url, model, version))
            .call()
            .map_err(Box::new)
    };
    let get_result = with_retry(get_session);

    let info = match get_result {
        Ok(response) => response
            .into_json::<Value>()
            .map_err(|e| MemoryError::Embedding(format!("invalid session info JSON: {e}")))?,
        Err(_) => {
            let create_result = with_retry(|| {
                agent
                    .post(&format!("{url}/api/sessions"))
                    .send_json(json!({ "model": model, "version": version }))
                    .map_err(Box::new)
            });
            match create_result {
                Ok(response) => response.into_json::<Value>().map_err(|e| {
                    MemoryError::Embedding(format!("invalid session-create JSON: {e}"))
                })?,
                Err(e) if matches!(*e, ureq::Error::Status(409, _)) => with_retry(get_session)
                    .map_err(|e| {
                        MemoryError::Embedding(format!(
                            "onnxruntime-server session exists but GET failed: {e}"
                        ))
                    })?
                    .into_json::<Value>()
                    .map_err(|e| {
                        MemoryError::Embedding(format!("invalid session info JSON: {e}"))
                    })?,
                Err(e) => {
                    return Err(MemoryError::Embedding(format!(
                        "failed to create onnxruntime-server session for {model}/{version} at {url}: {e}"
                    )));
                }
            }
        }
    };

    info.get("inputs")
        .and_then(|v| v.as_object())
        .cloned()
        .ok_or_else(|| {
            MemoryError::Embedding("onnxruntime-server session info missing 'inputs'".to_string())
        })
}

fn execute_with_retry(
    agent: &ureq::Agent,
    url: &str,
    model: &str,
    version: &str,
    body: &Value,
) -> Result<Value, MemoryError> {
    let result = with_retry(|| {
        agent
            .post(&session_path(url, model, version))
            .send_json(body.clone())
            .map_err(Box::new)
    });
    match result {
        Ok(response) => response.into_json::<Value>().map_err(|e| {
            MemoryError::Embedding(format!("invalid onnxruntime-server response JSON: {e}"))
        }),
        Err(e) => Err(MemoryError::Embedding(format!(
            "onnxruntime-server execute request failed for {model}/{version} at {url}: {e}"
        ))),
    }
}

/// Bounded retries for transient connection issues only (e.g. server mid-restart).
/// Fails fast and loudly after `CONNECT_RETRIES` attempts -- no infinite retry.
/// `ureq::Error` is boxed by callers to keep this function's error type small
/// (clippy `result_large_err`); callers that need to match on the inner
/// variant (e.g. a 409 conflict) deref the box.
fn with_retry<T>(
    mut f: impl FnMut() -> Result<T, Box<ureq::Error>>,
) -> Result<T, Box<ureq::Error>> {
    let mut attempt = 0;
    loop {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) if matches!(*e, ureq::Error::Transport(_)) && attempt + 1 < CONNECT_RETRIES => {
                attempt += 1;
                std::thread::sleep(RETRY_BASE_DELAY * attempt);
            }
            Err(e) => return Err(e),
        }
    }
}
