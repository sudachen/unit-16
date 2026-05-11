use anyhow::Result;

mod cli;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr) // Явно указываем stderr (хотя это дефолт)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    
    let cli = cli::parse();
    cli::route(cli)?;
    Ok(())
}
