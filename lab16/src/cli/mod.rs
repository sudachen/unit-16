mod extract;

use anyhow::Result;
use clap::{Parser, Subcommand};
use extract::ExtractArgs;

#[derive(Parser, Debug)]
#[command(name = "lab16")]
#[command(about = "Lab16 CLI tool", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Extract knowledge from a text
    Extract(ExtractArgs),
}

pub fn parse() -> Cli {
    Cli::parse()
}

pub fn route(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Extract(args) => extract::extract(args),
    }
}
