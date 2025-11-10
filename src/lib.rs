use anyhow::anyhow;
use ort::value::Tensor;
use std::array::TryFromSliceError;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

const MODEL_INPUT_LIMIT: usize = 512;
const DEFAULT_POOL_SIZE: usize = 3;

#[derive(Debug, thiserror::Error)]
pub enum BgeError {
    #[error(
        "Number of tokens in the input exceed the model limit. Limit: {}, got: {}",
        MODEL_INPUT_LIMIT,
        0
    )]
    LargeInput(usize),
    #[error(transparent)]
    OnnxRuntimeError(#[from] ort::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub struct Bge {
    tokenizer: Tokenizer,
    model: ort::session::Session,
}

impl Bge {
    /// Creates a new instance of `Bge` by loading a tokenizer and a model from the specified file paths.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_file_path` - A path to the file containing the tokenizer configuration.
    /// * `model_file_path` - A path to the ONNX model file.
    ///
    /// # Returns
    ///
    /// If successful, returns an `Ok(Self)` containing a new instance of `Bge`. On failure, returns an `Err(anyhow::Error)`
    /// detailing the error encountered during the loading process.
    ///
    /// # Errors
    ///
    /// This function can fail if:
    /// - The paths provided do not point to valid files.
    /// - The tokenizer or model file cannot be correctly parsed or loaded, possibly due to format issues or
    ///   compatibility problems.
    ///
    /// # Examples
    ///
    /// ```
    /// let bge = bge::Bge::from_files("path/to/tokenizer.json", "path/to/model.onnx");
    /// match bge {
    ///     Ok(instance) => println!("Bge instance created successfully."),
    ///     Err(e) => eprintln!("Failed to create Bge instance: {}", e),
    /// }
    /// ```
    pub fn from_files<P>(tokenizer_file_path: P, model_file_path: P) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        let tokenizer = Tokenizer::from_file(tokenizer_file_path.as_ref().to_str().unwrap())
            .map_err(|e| anyhow!(e))?;
        let model = ort::session::Session::builder()?.commit_from_file(model_file_path)?;
        Ok(Self { tokenizer, model })
    }

    /// Generates embeddings for a given input text using the model.
    ///
    /// This method tokenizes the input text, performs necessary preprocessing,
    /// and then runs the model to produce embeddings. The embeddings are normalized
    /// before being returned.
    ///
    /// # Arguments
    ///
    /// * `input` - The text input for which embeddings should be generated.
    ///
    /// # Returns
    ///
    /// If successful, returns a `Result` containing a fixed-size array of `f32` elements representing
    /// the generated embeddings. On failure, returns a `BgeError` detailing the nature of the error.
    ///
    /// # Errors
    ///
    /// This method can return an error in several cases:
    /// - `BgeError::LargeInput` if the input text produces more tokens than the model can accept.
    /// - `BgeError::OnnxRuntimeError` for errors related to running the ONNX model.
    /// - `BgeError::Other` for all other errors, including issues with tokenization or tensor extraction.
    ///
    /// # Examples
    ///
    /// ```
    /// # let bge = bge::Bge::from_files("path/to/tokenizer.json", "path/to/model.onnx").unwrap();
    /// let embeddings = bge.create_embeddings("This is a sample text.");
    /// match embeddings {
    ///     Ok(embeds) => println!("Embeddings: {:?}", embeds),
    ///     Err(e) => eprintln!("Error generating embeddings: {}", e),
    /// }
    /// ```
    pub fn create_embeddings(&mut self, input: &str) -> Result<[f32; 384], BgeError> {
        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(|e| BgeError::Other(anyhow!(e)))?;
        let encoding_ids = encoding.get_ids();
        let tokens_count = encoding_ids.len();

        if tokens_count > MODEL_INPUT_LIMIT {
            return Err(BgeError::LargeInput(tokens_count));
        }

        let input_ids: Vec<i64> = encoding_ids.iter().map(|v| *v as i64).collect();
        let attention_mask: Vec<i64> = vec![1; tokens_count];
        let token_type_ids: Vec<i64> = vec![0; tokens_count];
        let seq_shape = [1usize, tokens_count];

        let input_ids_tensor =
            Tensor::from_array((seq_shape, input_ids)).map_err(BgeError::OnnxRuntimeError)?;
        let attention_mask_tensor =
            Tensor::from_array((seq_shape, attention_mask)).map_err(BgeError::OnnxRuntimeError)?;
        let token_type_ids_tensor =
            Tensor::from_array((seq_shape, token_type_ids)).map_err(BgeError::OnnxRuntimeError)?;

        let inputs = ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        };

        let outputs = self.model.run(inputs).map_err(BgeError::OnnxRuntimeError)?;

        let output = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(BgeError::OnnxRuntimeError)?;
        let view = output.1;

        let vec: Vec<f32> = view.to_vec();
        let slice = vec.as_slice();
        let mut res: [f32; 384] = slice
            .try_into()
            .map_err(|e: TryFromSliceError| BgeError::Other(e.into()))?;
        normalize(&mut res);
        Ok(res)
    }
}

fn normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
    if norm != 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// A session pool for managing multiple BGE inference sessions.
///
/// This pool maintains a fixed number of `Bge` instances that can be used concurrently
/// to generate embeddings. When a session is requested via `create_embeddings`, the pool
/// will wait until a session becomes available if all sessions are currently in use.
///
/// The pool uses a channel-based approach to manage session acquisition and release,
/// ensuring that sessions are properly returned after use through RAII guards.
pub struct BgeSessionPool {
    /// Channel sender for returning sessions to the pool
    sender: mpsc::Sender<Bge>,
    /// Channel receiver for acquiring sessions from the pool
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<Bge>>>,
    /// Number of sessions in the pool
    pool_size: usize,
}

impl BgeSessionPool {
    /// Creates a new session pool with the default pool size.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_file_path` - Path to the tokenizer configuration file.
    /// * `model_file_path` - Path to the ONNX model file.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `BgeSessionPool` or an error if any session
    /// fails to initialize.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The tokenizer or model files cannot be loaded
    /// - Any of the `Bge` instances fail to initialize
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio_test::block_on(async {
    /// let pool = bge::BgeSessionPool::new("path/to/tokenizer.json", "path/to/model.onnx").unwrap();
    /// # });
    /// ```
    pub fn new<P>(tokenizer_file_path: P, model_file_path: P) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        Self::with_pool_size(tokenizer_file_path, model_file_path, DEFAULT_POOL_SIZE)
    }

    /// Creates a new session pool with a specified pool size.
    ///
    /// # Arguments
    ///
    /// * `tokenizer_file_path` - Path to the tokenizer configuration file.
    /// * `model_file_path` - Path to the ONNX model file.
    /// * `pool_size` - Number of concurrent sessions to maintain in the pool.
    ///
    /// # Returns
    ///
    /// A `Result` containing the initialized `BgeSessionPool` or an error if any session
    /// fails to initialize.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The tokenizer or model files cannot be loaded
    /// - Any of the `Bge` instances fail to initialize
    /// - The pool size is 0
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio_test::block_on(async {
    /// let pool = bge::BgeSessionPool::with_pool_size(
    ///     "path/to/tokenizer.json",
    ///     "path/to/model.onnx",
    ///     5
    /// ).unwrap();
    /// # });
    /// ```
    pub fn with_pool_size<P>(
        tokenizer_file_path: P,
        model_file_path: P,
        pool_size: usize,
    ) -> anyhow::Result<Self>
    where
        P: AsRef<Path>,
    {
        if pool_size == 0 {
            return Err(anyhow!("Pool size must be greater than 0"));
        }

        let (sender, receiver) = mpsc::channel(pool_size);

        // Initialize all sessions and add them to the pool
        for _ in 0..pool_size {
            let session = Bge::from_files(&tokenizer_file_path, &model_file_path)?;
            sender
                .blocking_send(session)
                .map_err(|e| anyhow!("Failed to initialize pool: {}", e))?;
        }

        Ok(Self {
            sender,
            receiver: Arc::new(tokio::sync::Mutex::new(receiver)),
            pool_size,
        })
    }

    /// Acquires a session from the pool.
    ///
    /// This method will wait asynchronously until a session becomes available.
    /// The session is automatically returned to the pool when the guard is dropped.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `BgeSessionGuard` that provides access to the session,
    /// or an error if the pool has been closed.
    async fn acquire_session(&self) -> anyhow::Result<BgeSessionGuard> {
        let mut receiver = self.receiver.lock().await;
        let session = receiver
            .recv()
            .await
            .ok_or_else(|| anyhow!("Session pool has been closed"))?;

        Ok(BgeSessionGuard {
            session: Some(session),
            sender: self.sender.clone(),
        })
    }

    /// Generates embeddings for the given input text using an available session from the pool.
    ///
    /// This method acquires a session from the pool (waiting if necessary), generates embeddings,
    /// and automatically returns the session to the pool when complete.
    ///
    /// # Arguments
    ///
    /// * `input` - The text input for which embeddings should be generated.
    ///
    /// # Returns
    ///
    /// A `Result` containing a fixed-size array of `f32` elements representing the generated
    /// embeddings, or a `BgeError` on failure.
    ///
    /// # Errors
    ///
    /// This method can return an error in several cases:
    /// - `BgeError::LargeInput` if the input text produces more tokens than the model can accept.
    /// - `BgeError::OnnxRuntimeError` for errors related to running the ONNX model.
    /// - `BgeError::Other` for all other errors, including session acquisition failures.
    ///
    /// # Examples
    ///
    /// ```
    /// # tokio_test::block_on(async {
    /// # let pool = bge::BgeSessionPool::new("path/to/tokenizer.json", "path/to/model.onnx").unwrap();
    /// let embeddings = pool.create_embeddings("This is a sample text.").await;
    /// match embeddings {
    ///     Ok(embeds) => println!("Embeddings generated successfully"),
    ///     Err(e) => eprintln!("Error: {}", e),
    /// }
    /// # });
    /// ```
    pub async fn create_embeddings(&self, input: &str) -> Result<[f32; 384], BgeError> {
        let mut guard = self
            .acquire_session()
            .await
            .map_err(|e| BgeError::Other(e))?;
        guard.session.as_mut().unwrap().create_embeddings(input)
    }

    /// Returns the configured pool size.
    ///
    /// # Returns
    ///
    /// The number of sessions maintained in this pool.
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }
}

/// RAII guard that automatically returns a BGE session to the pool when dropped.
///
/// This guard ensures that sessions are always returned to the pool, even in the
/// presence of panics or early returns.
struct BgeSessionGuard {
    session: Option<Bge>,
    sender: mpsc::Sender<Bge>,
}

impl Drop for BgeSessionGuard {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            // Use blocking_send since Drop cannot be async
            // If this fails, the session is lost but the pool continues to function
            let _ = self.sender.blocking_send(session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod test_data;

    #[test]
    fn it_works() {
        let mut bge = Bge::from_files("assets/tokenizer.json", "assets/model.onnx").unwrap();
        let res = bge.create_embeddings("Some input text to generate embeddings for.");

        assert_eq!(res.unwrap(), test_data::TEST_EMBEDDING_RESULT);
    }

    #[tokio::test]
    async fn test_session_pool() {
        let pool = BgeSessionPool::new("assets/tokenizer.json", "assets/model.onnx").unwrap();

        // Verify pool size
        assert_eq!(pool.pool_size(), DEFAULT_POOL_SIZE);

        // Test single embedding generation
        let res = pool
            .create_embeddings("Some input text to generate embeddings for.")
            .await;

        assert_eq!(res.unwrap(), test_data::TEST_EMBEDDING_RESULT);
    }

    #[tokio::test]
    async fn test_session_pool_concurrent() {
        let pool = std::sync::Arc::new(
            BgeSessionPool::with_pool_size("assets/tokenizer.json", "assets/model.onnx", 2)
                .unwrap(),
        );

        // Create multiple concurrent embedding tasks
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let pool_clone = pool.clone();
                tokio::spawn(async move {
                    pool_clone
                        .create_embeddings(&format!("Test input {}", i))
                        .await
                })
            })
            .collect();

        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_custom_pool_size() {
        let pool = BgeSessionPool::with_pool_size("assets/tokenizer.json", "assets/model.onnx", 5)
            .unwrap();

        assert_eq!(pool.pool_size(), 5);
    }

    #[test]
    fn test_zero_pool_size_fails() {
        let result = BgeSessionPool::with_pool_size("assets/tokenizer.json", "assets/model.onnx", 0);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("greater than 0"));
        }
    }
}
