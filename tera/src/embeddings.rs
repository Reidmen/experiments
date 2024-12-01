use std::path::PathBuf;

use crate::utils::device;
use anyhow::{Context, Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo};
use lazy_static::lazy_static;
use tokenizers::{PaddingParams, Tokenizer};
use tracing::debug;

lazy_static! {
    pub static ref AI: (BertModel, Tokenizer) = load_model().expect("Unable to load model");
}

fn get_model_path() -> Result<PathBuf> {
    let path = dirs::cache_dir()
        .context("Unable to get cache directory")?
        .join("tera")
        .join("models")
        .join("bge-small-en-v1.5");

    Ok(path)
}

pub fn clear_model_cache() -> Result<()> {
    let model_path = get_model_path()?;
    if model_path.exists() {
        std::fs::remove_dir_all(&model_path)?;
    }
    Ok(())
}

pub fn load_model() -> Result<(BertModel, Tokenizer)> {
    let model_path = get_model_path()?;
    std::fs::create_dir_all(&model_path).context("Unable to create model directory")?;

    let config_path = model_path.join("config.json");
    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("pytorch_model.bin");

    // check if the files exist
    let file_exists = config_path.exists() && tokenizer_path.exists() && weights_path.exists();
    if !file_exists {
        debug!("Downloading model files...");
        let api = Api::new()?.repo(Repo::model("BAAI/bge-small-en-v1.5".to_string()));

        if !config_path.exists() {
            let config_filename = api.get("config.json")?;
            std::fs::copy(config_filename, &config_path)?;
        }

        if !tokenizer_path.exists() {
            let tokenizer_filename = api.get("tokenizer.json")?;
            std::fs::copy(tokenizer_filename, &tokenizer_path)?;
        }

        if !weights_path.exists() {
            let weights_filename = api.get("pytorch_model.bin")?;
            std::fs::copy(weights_filename, &weights_path)?;
        }
        debug!("Model files downloaded");
    } else {
        debug!("Loading models from cache");
    }

    // Load models
    let config = std::fs::read_to_string(config_path)?;
    let config: Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(E::msg)
        .context("Unable to load tokenizer")?;

    let vb = VarBuilder::from_pth(&weights_path, DTYPE, &device(false)?)?;
    let model = BertModel::load(vb, &config)?;

    // Configure the tokenizer padding
    if let Some(pp) = tokenizer.get_padding_mut() {
        pp.strategy = tokenizers::PaddingStrategy::BatchLongest
    } else {
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
    }

    Ok((model, tokenizer))
}

pub fn get_embeddings(sentence: &str) -> Result<Tensor> {
    let (model, tokenizer) = &*AI;

    // drop any non-ascii characters
    let sentence = sentence.chars().filter(|c| c.is_ascii()).collect::<String>();

    let tokens = tokenizer
        .encode_batch(vec![sentence], true)
        .map_err(E::msg)
        .context("Unable to encode sentence")?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &device(false)?)?)
        })
        .collect::<Result<Vec<_>>>()
        .context("Unable to get token ids")?;

    let token_ids = Tensor::stack(&token_ids, 0).context("Unable to stack token ids")?;
    let token_type_ids = token_ids.zeros_like().context("Unable to get token type ids")?;

    let embeddings = model
        .forward(&token_ids, &token_type_ids, None)
        .context("Unable to get embeddings")?;

    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().context("Unable to get embeddings dimensions")?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64)).context("Unable to get embeddings sum")?;
    let embeddings = embeddings
        .broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)
        .context("Unable to get embeddings broadcast div")?;

    Ok(embeddings)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_clear_model_cache() {
        // Clear cache first
        let _ = clear_model_cache();

        // First load -- it will download the files
        let results = load_model();
        assert!(results.is_ok());

        // Verify the cache directory exists
        let cache_path = get_model_path().unwrap();
        assert!(cache_path.exists());
        assert!(cache_path.join("config.json").exists());
        assert!(cache_path.join("tokenizer.json").exists());
        assert!(cache_path.join("pytorch_model.bin").exists());

        // Second load -- it will succeed from cache
        let results_from_cache = load_model();
        assert!(results_from_cache.is_ok());
    }

    #[test]
    fn test_load_model() {
        let result = load_model();
        assert!(result.is_ok());

        let (_, tokenizer) = result.unwrap();

        // Check the tokenizer setup
        let padding = tokenizer.get_padding();
        assert!(padding.is_some());
        // check the model configuration
        // assert_eq!(model.config().hidden_size, 384);
        // assert_eq!(model.config().vocab_size, 30522);
    }

    #[test]
    fn test_get_embeddings_basic() {
        let _ = load_model();
        let results = get_embeddings("Hello, world!");
        assert!(results.is_ok());

        let embeddings = results.unwrap();
        assert_eq!(embeddings.dims(), [1, 384]); // [batch_size, hidden_size]
    }

    #[test]
    fn test_get_embeddings_empty() {
        let _ = load_model();
        let results = get_embeddings("");
        assert!(results.is_ok());
    }

    #[test]
    fn test_get_embeddings_long() {
        let _ = load_model();
        let results = get_embeddings(&"a".repeat(1000));
        assert!(results.is_ok());
    }

    #[test]
    fn test_get_embeddings_non_ascii() {
        let _ = load_model();
        let results = get_embeddings("Hello こんにちは");
        assert!(results.is_ok());

        // It should only process non-ascii characters
        let embeddings = results.unwrap();
        assert_eq!(embeddings.dims(), [1, 384]);
    }

    fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
        let dot_product = (a * b)?.sum_all()?;
        let norm_a = a.sqr()?.sum_all()?.sqrt()?;
        let norm_b = b.sqr()?.sum_all()?.sqrt()?;
        let similarity = (dot_product / (norm_a * norm_b))?;
        Ok(similarity.to_scalar::<f32>()?)
    }

    #[test]
    fn test_embeddings_similarity() {
        let _ = load_model();
        let embedding_base = get_embeddings("I love gaming").unwrap();
        let embedding_similar = get_embeddings("I love video games").unwrap();
        let embedding_dissimilar = get_embeddings("I love hiking").unwrap();

        // compute the cosine similarity between the base and similar
        let comparison_similar = cosine_similarity(&embedding_base, &embedding_similar).unwrap();
        let comparison_dissimilar = cosine_similarity(&embedding_base, &embedding_dissimilar).unwrap();

        // similar sentences should have higher similarity
        assert!(comparison_similar > comparison_dissimilar);
    }

    #[test]
    fn test_lazy_static() {
        // Test the lazy static is initialized
        let ai = &*AI;
        assert!(ai.0.device.is_cpu());

        // Test the subsequent access is fast
        let start = std::time::Instant::now();
        let _ai = &*AI;
        assert!(start.elapsed() < std::time::Duration::from_millis(10));
    }
}
