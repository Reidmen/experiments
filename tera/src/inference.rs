// Adopted from https://github.com/huggingface/candle/blob/96f1a28e390fceeaa12b3272c8ac5dcccc8eb5fa/candle-examples/examples/phi/main.rs
use crate::database::VectorIndex;
use crate::utils::device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_mixformer::Config;
use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
use hf_hub::{api::sync::Api, Repo};
use lazy_static::lazy_static;
use serde_json::json;
use tokenizers::Tokenizer;
use tracing::debug;

lazy_static! {
    pub static ref PHI: (QMixFormer, Tokenizer) = load_model().expect("Unable to load model");
}

pub fn load_model() -> Result<(QMixFormer, Tokenizer)> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Unable to get cache directory"))?
        .join("tera");
    std::fs::create_dir_all(&cache_dir)?;

    let tokenizer_path = cache_dir.join("tokenizer.json");
    let weights_path = cache_dir.join("model-q4k.gguf");

    // Download files if they don't exist in cache directory
    if !tokenizer_path.exists() || !weights_path.exists() {
        let api = Api::new()?.repo(Repo::model("Demonthos/dolphin-2_6-phi-2-candle".to_string()));

        if !tokenizer_path.exists() {
            debug!("Downloading tokenizer...");
            let tokenizer_filename = api.get("tokenizer.json")?;
            std::fs::copy(tokenizer_filename, &tokenizer_path)?;
        }

        if !weights_path.exists() {
            debug!("Downloading model weights...");
            let weights_filename = api.get("model-q4k.gguf")?;
            std::fs::copy(weights_filename, &weights_path)?;
        }
    }

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;
    let config = Config::v2();
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&weights_path, &device(false)?)?;
    let model = QMixFormer::new_v2(&config, vb)?;

    Ok((model, tokenizer))
}

struct TextGeneration {
    model: QMixFormer,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: QMixFormer,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        debug!(prompt = prompt, "starting the inference loop");
        let tokens = self.tokenizer.encode(prompt, true).map_err(E::msg)?;
        if tokens.is_empty() {
            anyhow::bail!("Empty prompts are not supported in the phi model.")
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => anyhow::bail!("cannot find the endoftext token"),
        };
        let start_gen = std::time::Instant::now();

        let mut response = String::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[start_at..])?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == 198 {
                break;
            }
            let token = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            response += &token;
        }
        let dt = start_gen.elapsed();
        debug!(
            generated_tokens = generated_tokens,
            speed = format!("{:.2} token/s", generated_tokens as f64 / dt.as_secs_f64()),
            "inference loop finished"
        );
        Ok(response.trim().to_string())
    }
}

pub async fn answer_with_context(query: &str, references: Vec<VectorIndex>) -> Result<String> {
    if references.is_empty() {
        return Ok(
            "Non of your saved content is relevant to this question. I can only answer based on your saved content."
                .to_string(),
        );
    }

    let mut context = Vec::new();
    for reference in references.clone() {
        context.push(json!(
            {
                "content": reference.content_chunk,
                "metadata": reference.metadata,
            }
        ))
    }

    let context = json!(context).to_string();

    let prompt = format!(
        "<|im_start|>system\nAs a helpful AI assistant named Tera. Your answer should be very concise. Do not repeat question or references. Today is {date}<|im_end|>\n<|im_start|>user\nquestion: \"{question}\"\nreferences: \"{context}\"\n<|im_end|>\n<|im_start|>assistant\n",
        context = context,
        question = query,
        date = chrono::Local::now().format("%A, %B %e, %Y")
    );

    debug!(prompt =? prompt, "Synthesizing answer with context");

    let (model, tokenizer) = &*PHI;

    let mut pipeline = TextGeneration::new(
        model.clone(),
        tokenizer.clone(),
        398752958,
        Some(0.3),
        None,
        1.1,
        64,
        &device(false)?,
    );
    let response = pipeline.run(&prompt, 400)?;

    Ok(response)
}

#[cfg(test)]
mod tests {
    use surrealdb::sql::{Datetime, Thing};

    use super::*;

    fn ensure_model_loaded() {
        let _ = load_model().unwrap();
    }

    #[test]
    fn test_load_model() {
        let result = load_model();
        assert!(result.is_ok(), "Model failed to load: {}", result.err().unwrap());

        let (_, tokenizer) = result.unwrap();
        // Test tokenizer
        let tokens = tokenizer.encode("Hello, world!", true).unwrap();
        assert!(!tokens.get_ids().is_empty(), "Tokenizer returned empty tokens");
    }

    #[tokio::test]
    async fn test_answer_with_context_empty_references() {
        ensure_model_loaded();
        let result = answer_with_context("What is the capital of France?", Vec::new()).await;
        assert!(
            result.is_ok(),
            "Failed to answer with empty references: {}",
            result.err().unwrap()
        );
        assert_eq!(
            result.unwrap(),
            "Non of your saved content is relevant to this question. I can only answer based on your saved content."
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_answer_with_context() {
        ensure_model_loaded();
        let references = vec![VectorIndex {
            id: Thing::from(("test", "1")),
            content_id: Thing::from(("content", "1")).to_string(),
            chunk_number: 1,
            content_chunk: "France is a country in Europe.".to_string(),
            metadata: json!({}),
            vector: vec![0.1; 384],
            created_at: Datetime::default(),
        }];
        let result = answer_with_context("What is the capital of France?", references).await;
        assert!(
            result.is_ok(),
            "Failed to answer with context: {}",
            result.err().unwrap()
        );

        let answer = result.unwrap();
        assert!(!answer.is_empty(), "Answer is empty");
    }

    #[tokio::test]
    #[ignore]
    async fn test_text_generation() {
        ensure_model_loaded();

        let (model, tokenizer) = load_model().unwrap();
        let mut pipeline = TextGeneration::new(
            model,
            tokenizer,
            398752958,
            Some(0.3), // Some temperature
            Some(0.9), // Some top_p
            1.1,       // repeat_penalty
            64,        // context size
            &device(false).unwrap(),
        );
        let response = pipeline.run("Hello, world!, how are you?", 100);
        assert!(response.is_ok(), "Failed to generate text: {}", response.err().unwrap());
        assert!(!response.unwrap().is_empty(), "Response is empty");
    }

    #[test]
    fn test_model_cache() {
        // First load should create cache
        let first_load = load_model();
        assert!(
            first_load.is_ok(),
            "First model load failed: {}",
            first_load.err().unwrap()
        );

        // Second load should load from cache
        let second_load = load_model();
        assert!(
            second_load.is_ok(),
            "Second model load failed: {}",
            second_load.err().unwrap()
        );

        // Verify cache directory exists
        let cache_dir = dirs::cache_dir().unwrap().join("tera");
        assert!(cache_dir.exists(), "Cache directory does not exist");
        assert!(
            cache_dir.join("tokenizer.json").exists(),
            "Tokenizer file does not exist"
        );
        assert!(cache_dir.join("model-q4k.gguf").exists(), "Model file does not exist");
    }
}
