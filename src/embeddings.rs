use serde::{Deserialize, Serialize};

use crate::{Usage, VoyageAi, VoyageAiError};

/// Represents the body parameters for the API request
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// A single text string, or a list of texts as a list of strings.
    /// Currently, we have two constraints on the list:
    /// - The maximum length of the list is 128.
    /// - The total number of tokens in the list is at most 320K for voyage-2,
    ///   and 120K for voyage-large-2, voyage-finance-2, voyage-multilingual-2,
    ///   voyage-law-2, and voyage-code-2.
    pub input: EmbeddingInput,

    /// Name of the model.
    pub model: EmbeddingModel,

    /// Type of the input text.
    pub input_type: Option<EmbeddingInputType>,

    /// Whether to truncate the input texts to fit within the context length.
    /// Defaults to true.
    /// If true, over-length input texts will be truncated to fit within the
    /// context length, before vectorized by the embedding model.
    /// If false, an error will be raised if any given text exceeds the context length.
    pub truncation: Option<bool>,

    /// Format in which the embeddings are encoded. We support two options:
    /// - If not specified (defaults to null): the embeddings are represented as lists of floating-point numbers;
    /// - base64: the embeddings are compressed to base64 encodings.
    pub encoding_format: Option<String>,
}

/// Represents the type of input text
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EmbeddingInputType {
    #[serde(rename = "query")]
    Query,
    #[serde(rename = "document")]
    Document,
}

/// Represents the available embedding models
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EmbeddingModel {
    #[serde(rename = "voyage-3")]
    Voyage3,
    #[serde(rename = "voyage-lite-3")]
    VoyageLite3,
    #[serde(rename = "voyage-2")]
    Voyage2,
    #[serde(rename = "voyage-large-2")]
    VoyageLarge2,
    #[serde(rename = "voyage-finance-2")]
    VoyageFinance2,
    #[serde(rename = "voyage-multilingual-2")]
    VoyageMultilingual2,
    #[serde(rename = "voyage-law-2")]
    VoyageLaw2,
    #[serde(rename = "voyage-code-2")]
    VoyageCode2,
}

/// Represents the input type, which can be either a single string or a list of strings
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Represents the response body for embeddings
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The object type, which is always "list".
    pub object: String,

    /// An array of embedding objects.
    pub data: Vec<EmbeddingObject>,

    /// Usage information for the request.
    pub usage: Usage,

    /// Name of the model.
    pub model: EmbeddingModel,
}

/// Represents a single embedding object
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingObject {
    /// The object type, which is always "embedding".
    pub object: String,

    /// The embedding vector consists of a list of floating-point numbers.
    /// The length of this vector varies depending on the specific model.
    pub embedding: Vec<f32>,

    /// An integer representing the index of the embedding within the list of embeddings.
    pub index: i32,
}

impl VoyageAi {
    pub async fn embeddings(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, VoyageAiError> {
        self.post("/v1/embeddings", request).await
    }
}
