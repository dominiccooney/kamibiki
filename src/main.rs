use std::io::{self, Read, Write};

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
    /// Debug / diagnostic subcommands
    Debug {
        #[command(subcommand)]
        command: DebugCommands,
    },
}

#[derive(Subcommand)]
enum DebugCommands {
    /// Run the chunker on files and display the resulting chunks
    Chunk {
        /// Files to chunk. If none are given, reads from stdin (requires --stdin-filename).
        files: Vec<String>,

        /// Maximum tokens per chunk (default: 430)
        #[arg(long, default_value_t = 430)]
        max_tokens: usize,

        /// Filename to use for language detection when reading from stdin
        #[arg(long)]
        stdin_filename: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Debug { command } => match command {
            DebugCommands::Chunk {
                files,
                max_tokens,
                stdin_filename,
            } => cmd_debug_chunk(files, max_tokens, stdin_filename),
        },
        // Other commands will be filled in as workstreams merge.
        _ => todo!(),
    }
}

fn cmd_debug_chunk(
    files: Vec<String>,
    max_tokens: usize,
    stdin_filename: Option<String>,
) -> anyhow::Result<()> {
    use kb::chunk::tsv1_chunker;

    let chunker = tsv1_chunker()?;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    if files.is_empty() {
        // Read from stdin
        let filename = stdin_filename.as_deref().unwrap_or("stdin.txt");
        let mut content = Vec::new();
        io::stdin().read_to_end(&mut content)?;
        print_chunks(&mut out, filename, &content, &chunker, max_tokens)?;
    } else {
        for (i, path) in files.iter().enumerate() {
            if i > 0 {
                writeln!(out)?;
            }
            let content = std::fs::read(path)
                .map_err(|e| anyhow::anyhow!("failed to read {}: {}", path, e))?;
            print_chunks(&mut out, path, &content, &chunker, max_tokens)?;
        }
    }

    Ok(())
}

fn print_chunks(
    out: &mut impl Write,
    path: &str,
    content: &[u8],
    chunker: &impl kb::chunk::Chunker,
    max_tokens: usize,
) -> anyhow::Result<()> {
    let chunks = chunker.chunk(path, content, max_tokens)?;

    writeln!(out, "=== {} ({} chunk{}) ===", path, chunks.len(), if chunks.len() == 1 { "" } else { "s" })?;

    for (i, chunk) in chunks.iter().enumerate() {
        if i > 0 {
            writeln!(out, "---")?;
        }
        let text = String::from_utf8_lossy(&chunk.content);
        write!(out, "{}", text)?;
    }

    // Ensure there's a trailing newline after the last chunk
    if chunks.last().is_some_and(|c| !c.content.ends_with(b"\n")) {
        writeln!(out)?;
    }

    Ok(())
}
