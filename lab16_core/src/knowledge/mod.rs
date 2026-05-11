use serde::{Deserialize, Serialize};

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
