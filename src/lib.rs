mod embeddings;
mod error;
mod rerank;

use std::sync::Arc;

use anyhow::{anyhow, Result};
use http_client::{
    http::{header::AUTHORIZATION, HeaderMap, HeaderValue, Method},
    HttpClient, Request, RequestBuilderExt, ResponseAsyncBodyExt,
};
use secrecy::{ExposeSecret, SecretString};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

pub use crate::{embeddings::*, error::*, rerank::*};

pub const BASE_URL: &str = "https://api.voyageai.com";

/// Represents usage information for the request
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    /// The total number of tokens used for computing the embeddings.
    pub total_tokens: i32,
}

pub struct VoyageAi {
    http_client: Arc<dyn HttpClient>,
    api_key: SecretString,
    base_url: String,
}

#[derive(Clone)]
pub struct VoyageAiBuilder {
    http_client: Option<Arc<dyn HttpClient>>,
    api_key: Option<SecretString>,
    base_url: Option<String>,
}

impl VoyageAi {
    pub fn builder() -> VoyageAiBuilder {
        VoyageAiBuilder {
            http_client: None,
            api_key: None,
            base_url: None,
        }
    }

    pub(crate) async fn post<P, S, D>(&self, path: P, request: S) -> Result<D, VoyageAiError>
    where
        P: Into<String>,
        S: Serialize,
        D: DeserializeOwned,
    {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key.expose_secret()))
                .expect("couldn't create header value"),
        );

        let response = self
            .http_client
            .send(
                Request::builder()
                    .uri(format!("{}{}", self.base_url, path.into()))
                    .method(Method::POST)
                    .headers(headers)
                    .json(&request)?,
            )
            .await?;

        let status = response.status();
        if !status.is_success() {
            let payload = response.json::<HttpErrorPayload>().await.ok();
            return Err(VoyageAiError::HttpError(HttpError {
                status: status.as_u16(),
                payload,
            }));
        }

        let response = response.json::<D>().await?;
        Ok(response)
    }
}

impl VoyageAiBuilder {
    pub fn with_http_client(mut self, http_client: Arc<dyn HttpClient>) -> Self {
        self.http_client = Some(http_client);
        self
    }

    pub fn with_api_key<S>(mut self, api_key: S) -> Self
    where
        S: AsRef<str>,
    {
        self.api_key = Some(api_key.as_ref().to_string().into());
        self
    }

    pub fn with_base_url<S>(mut self, base_url: S) -> Self
    where
        S: AsRef<str>,
    {
        self.base_url = Some(base_url.as_ref().into());
        self
    }

    pub fn build(self) -> Result<VoyageAi> {
        Ok(VoyageAi {
            http_client: self.http_client.ok_or_else(|| anyhow!("http_client must be specified"))?,
            api_key: self.api_key.or_else(|| std::env::var("VOYAGEAI_API_KEY").ok().map(SecretString::new))
                .ok_or_else(|| anyhow!("API key is required. Set it explicitly or use the VOYAGEAI_API_KEY environment variable"))?,
            base_url: self.base_url.unwrap_or_else(|| BASE_URL.to_string()),
        })
    }
}
