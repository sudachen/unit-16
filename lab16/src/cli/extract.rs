use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use lab16_core::extractor::{Extractor};
use std::fs::OpenOptions;
use std::io::Write;
use tracing::debug;

#[derive(Parser, Debug)]
pub struct ExtractArgs {
    /// Input file path(s) - supports multiple files
    #[arg(num_args(1..))]
    pub input: Vec<PathBuf>,
    /// Output file path
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

pub fn extract(args: ExtractArgs) -> Result<()> {
    println!("Extracting from {} file(s):", args.input.len());
    let mut outfile = OpenOptions::new()
        .create(true)
        .append(true)
        .open(args.output.unwrap_or("entities.json".into()))?;

    let mut extractor = Extractor::new();
    for (index, file) in args.input.iter().enumerate() {
        println!("  File {}: {:?}", index + 1, file);
        extractor.extract_from_file(file)?;
        /*extractor.extract(file,|e| {
            let json_line = serde_json::to_string(&e)?;
            if e.importance >= 3  && e.entity_type != EntityType::Unknown  {
                debug!("Entity verdict: {:?}", e);
                writeln!(outfile, "{}", json_line)?;
            }
            Ok(())
        })?;*/
    }
    Ok(())
}
