use std::path::PathBuf;
use anyhow::Result;
use super::scanner::{EntityScanner, EntityEvidence};
use serde::{Deserialize, Serialize};
use tracing::debug;
use crate::gpu_llm::Model;

const MAX_OUTPUT_TOKENS: u32 = 500;

#[derive(Debug, Deserialize, Serialize)]
pub struct EntityVerdict {
    pub name: String,
    pub aliases: Vec<String>,
    pub entity_type: EntityType,
    pub description: String,
    pub importance: u8,      // 1-10
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone, Default)]
#[serde(rename_all = "PascalCase")] // Expect from the neural network "Person", "Race", etc.
pub enum EntityType {
    Person,       // Specific character (Maxim, Guy)
    Race,         // Race or biological species (Headmen, Ludens)
    Location,     // Place of action (Saraksh, Pandora)
    Organization, // Group or structure (Combat Legion, COMCON)
    Object,       // Important object (Tank, tower, golden feather)
    Event,        // Event (mission, operation, battle)
    #[serde(other)]
    #[default]
    Unknown,      // Everything else (replacement for junk)
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AnalysisResponse {
    pub entities: Vec<EntityVerdict>,
}

pub struct Extractor {
    model: Option<Model>,
}

impl Extractor {
    pub fn new() -> Self {
        Self { model: None }
    }

    pub fn extract(&mut self, path: &PathBuf, mut write: impl FnMut(EntityVerdict) -> Result<()>) -> Result<()> {
        let mut es = EntityScanner::new();
        es.scan_fb2(path)?;
        //es.print_report(3);
        let top_candidates = es.get_top(2); // Get entities appearing at least 5 times

        if self.model.is_none() {
            debug!("Initializing entity extractor model");
            self.model = Some(Model::new(&crate::gpu_llm::get_or_download_model(
                super::ENTITY_EXTRACTOR_LLM_MODEL.0,
                super::ENTITY_EXTRACTOR_LLM_MODEL.1)?)?)
        }

        debug!("Initializing entity extractor context");
        let mut context = self.model.as_ref().unwrap().context().
            with_n_batch(std::num::NonZeroU32::new(4096).unwrap()).
            with_n_ctx(std::num::NonZeroU32::new(4096).unwrap()).
            build()?;


        for chunk in top_candidates.chunks(8) {
            debug!("Processing chunk");
            let prompt = format_chunk_prompt(chunk);
            context.clear_kv_cache();
            let response_json = context.inference(&prompt, MAX_OUTPUT_TOKENS)?;
            let json = clean_json(&response_json);
            debug!("JSON{json}");
            if let Ok(parsed) = serde_json::from_str::<AnalysisResponse>(json) {
                for entity in parsed.entities {
                    write(entity)?;
                }
            } else {
                tracing::warn!("LLM produced malformed JSON, skipping batch or trying again");
            }
        }


        Ok(())
    }
}

fn clean_json(raw: &str) -> &str {
    let start = raw.find('{').unwrap_or(0);
    let end = raw.rfind('}').map(|idx| idx + 1).unwrap_or(raw.len());
    &raw[start..end]
}

fn format_chunk_prompt(chunk: &[(&String, &EntityEvidence)]) -> String {

    let mut p = "".to_string();
    for (name, evidence) in chunk {
        p.push_str(&format!("### Entity: {}\nSnippets:\n", name));
        for s in &evidence.snippets {
            p.push_str(&format!("- \"{}\"\n", s));
        }
    }

    format!(
        r#"[INST] <<SYS>>
You are an advanced literary analysis engine. Classify entities extracted from text based on provided snippets.

EXTRACTION RULES:
1. NOMINATIVE CASE: All entity russian names MUST be converted to the Russian Nominative Case (Именительный падеж).
   Example: "Льву Абалкину" -> "Лев Абалкин", "Саракше" -> "Саракш".
2. ENTITY TYPES: Only use the following categories: [Person, Location, Organization, Object, Event, Unknown].
3. NO NOISE: Do not extract verbs, adjectives, or common nouns (e.g., "Выглядит", "Левой", "N").
4. DESCRIPTIONS: Provide a brief summary of the entity's role in the CURRENT text snippet.
5. IMPORTANCE: Rate from 1 to 10 based on the entity's relevance to the plot. Skip entities with importance < 4.
6. ZERO HALLUCINATION: Only use facts explicitly stated in the text. Do not add external information from your training data.

ENTITY TYPES:
- Person: Individual characters.
- Race: Sentient species, biological species, ethnic groups, or distinct types of beings.
- Location: Geographical places, structures, or coordinates.
- Organization: Named groups, factions, alliances, or institutional bodies.
- Object: Specific unique items, artifacts, or significant technologies.
- Event: Named incidents, codenamed operations (often in quotes), or specific historical missions.
- Unknown: Generic nouns, pronouns, or noise.

FILTERING:
- Ignore noise: single letters, verbs ("Выглядит"), directions ("Левой"), or partial names ("Вячеславович").

REQUIREMENTS:
- Output ONLY strictly valid JSON.
- The 'description' field MUST be in the same language as the snippets.
- 'importance' is an integer from 1 to 10.
- If a candidate is irrelevant, mark as 'Unknown'.

JSON SCHEMA:
{{
  "entities": [
    {{
      "name": "Full name in Nominative Case",
      "entity_type": "Person|Race|Location|Organization|Object|Event|Unknown",
      "description": "Brief context from the text",
      "importance": 1-10
    }}
  ]
}}
<</SYS>>

{p}

[/INST] JSON Output:
"#)
}