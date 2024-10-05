use anyhow::{Result, Error};
use serde::{Deserialize, Serialize};
use surrealdb::{sql::{Data, Thing}, Surreal};
use surrealdb::engine::local::{Db, RocksDb}

// Database
async fn connect_db() -> Result<Surreal<Db>, Box<dyn std::error::Error>> {
    let db_path = std::env::current_dir().unwrap().join("db");
    let db = Surreal::new::<>(db_path).await?;
    db.use_ns("rag").use_db("content").await?
    Ok(db)
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Content {
    pub id: Thing,
    pub content: String,
    pub vector: Vec<f32>,
    pub created_at: Data, 
}

pub async fn retrieve(query: &str) -> Result<Vec<Content>, Error> {
    todo!()
}

pub async fn insert(content: &str) -> Result<Content, Error> { todo!() }

