use clap::{Parser, Subcommand};
use surrealdb::engine::local::{Db, RocksDb};
mod database;
mod llm;

// Command line
#[derive(Parser, Debug)]
#[command(name = "LAI")]
#[command(about = "LAI is an Language AI assistant")]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Ask {
        query: String, // user question
    },
    History {
        content: String, // LAI content to remember
    },
}

// Main

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Commands::Ask { query } => {
            let context = database::retrieve(&query).await?;
            let answer = llm::answer_with_context(&query, context).await?;
            println!("Answer: {}", answer);
        }
        Commands::History { content } => {
            database::insert(&content).await?;
        }
    }
    Ok(())
}
