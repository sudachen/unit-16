# Project Agenting Guide

## Code Style Rules

### Error Handling

- **Always use `anyhow::Result<_>` instead of `Result<_,Box<_>>`**
  - Use `anyhow::Result<T>` for functions that can fail
  - This provides better error context and is more idiomatic in Rust applications
  - Example: `fn main() -> anyhow::Result<()>` instead of `fn main() -> Result<(), Box<dyn std::error::Error>>`

