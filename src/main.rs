use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "kb", about = "Kamibiki - contextual search for git repos")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Set up Voyage AI key
    Init,
    /// Add a repository to the index
    Add {
        name: String,
        path: String,
    },
    /// Update the index
    Index {
        names: Vec<String>,
        #[arg(long)]
        compact: bool,
    },
    /// Show indexing status
    Status {
        name: Option<String>,
    },
    /// Start the server
    Start,
    /// Stop the server
    Stop,
    /// Search a repository
    Search {
        name: String,
        query: String,
    },
    /// Create a repository alias
    Alias {
        name: String,
        repos: Vec<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    // dispatch will be filled in as workstreams merge
    todo!()
}
