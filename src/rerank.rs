use serde::{Deserialize, Serialize};

use crate::{Usage, VoyageAi, VoyageAiError};

/// Body parameters for the reranking request
#[derive(Debug, Serialize, Deserialize)]
pub struct RerankRequest {
    /// The query as a string. The query can contain a maximum of 1000 tokens for rerank-lite-1 and 2000 tokens for rerank-1.
    pub query: String,

    /// The documents to be reranked as a list of strings.
    ///
    /// - The number of documents cannot exceed 1000.
    /// - The sum of the number of tokens in the query and the number of tokens in any single document cannot exceed 4000 for rerank-lite-1 and 8000 for rerank-1.
    /// - The total number of tokens, defined as "the number of query tokens × the number of documents + sum of the number of tokens in all documents", cannot exceed 300K for rerank-lite-1 and 100K for rerank-1.
    pub documents: Vec<String>,

    /// Name of the model. Recommended options: rerank-lite-1, rerank-1.
    pub model: RerankModel,

    /// The number of most relevant documents to return. If not specified, the reranking results of all documents will be returned.
    pub top_k: Option<u32>,

    /// Whether to return the documents in the response. Defaults to false.
    ///
    /// - If false, the API will return a list of {"index", "relevance_score"} where "index" refers to the index of a document within the input list.
    /// - If true, the API will return a list of {"index", "document", "relevance_score"} where "document" is the corresponding document from the input list.
    pub return_documents: Option<bool>,

    /// Whether to truncate the input to satisfy the "context length limit" on the query and the documents. Defaults to true.
    ///
    /// - If true, the query and documents will be truncated to fit within the context length limit, before processed by the reranker model.
    /// - If false, an error will be raised when the query exceeds 1000 tokens for rerank-lite-1 and 2000 tokens for rerank-1, or the sum of the number of tokens in the query and the number of tokens in any single document exceeds 4000 for rerank-lite-1 and 8000 for rerank-1.
    pub truncation: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RerankModel {
    #[serde(rename = "rerank-lite-1")]
    RerankLite1,
    #[serde(rename = "rerank-1")]
    Rerank1,
}

/// Response body for the reranking request
#[derive(Debug, Serialize, Deserialize)]
pub struct RerankResponse {
    /// The object type, which is always "list".
    pub object: String,

    /// An array of the reranking results, sorted by the descending order of relevance scores.
    pub data: Vec<RerankObject>,

    /// Name of the model.
    pub model: RerankModel,

    /// Usage information for the request.
    pub usage: Usage,
}

/// Represents a single reranking result.
#[derive(Debug, Serialize, Deserialize)]
pub struct RerankObject {
    /// The index of the document in the input list.
    pub index: u32,

    /// The relevance score of the document with respect to the query.
    pub relevance_score: f64,

    /// The document string. Only returned when return_documents is set to true.
    pub document: Option<String>,
}

impl VoyageAi {
    pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, VoyageAiError> {
        self.post("/v1/rerank", request).await
    }
}
