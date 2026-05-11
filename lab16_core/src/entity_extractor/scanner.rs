use std::path::PathBuf;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use unicode_segmentation::UnicodeSegmentation;
use hashbrown::{HashMap, HashSet};
use anyhow::Result;

#[derive(Debug,Default)]
pub struct EntityEvidence {
    pub count: u32,
    pub snippets: Vec<String>, // Здесь будем хранить 3-5 примеров предложений
}

#[derive(Debug, Default)]
pub struct EntityScanner {
    // Map of single words found in mid-sentence with their frequencies
    pub candidates: HashMap<String, EntityEvidence>,
    // Potential full names or multi-word entities (e.g., "Maxim Kammerer")
    pub bigrams: HashMap<String, u32>,
    // Tracks words appearing at the very beginning of sentences to avoid noise
    pub sentence_starters: HashSet<String>,
}

impl EntityScanner {

    pub fn new() -> Self {
        Self::default()
    }

    pub fn scan_fb2(&mut self, path: &PathBuf) -> Result<()> {
        let mut reader = Reader::from_file(path)?;

        // In quick_xml 0.39, configuration is handled via config_mut()
        // Trim whitespace to avoid empty text events and messy strings
        reader.config_mut().trim_text(true);

        let mut buf = Vec::new();
        let mut in_p = false;

        loop {
            match reader.read_event_into(&mut buf) {
                // Check if we are inside a paragraph tag <p>
                Ok(Event::Start(e)) if e.name().as_ref() == b"p" => in_p = true,
                Ok(Event::End(e)) if e.name().as_ref() == b"p" => in_p = false,

                // Process text content only when inside <p>
                Ok(Event::Text(e)) if in_p => {
                    let text = reader.decoder().decode(&e)?;
                    self.process_text(&text);
                }

                Ok(Event::Eof) => break,
                Err(e) => return Err(e.into()),
                _ => (),
            }
            buf.clear();
        }
        Ok(())
    }

    fn process_text(&mut self, text: &str) {
        // Split text into sentences to handle sentence-start bias
        for sentence in text.split_sentence_bounds() {
            let words: Vec<&str> = sentence.unicode_words().collect();
            let word_count = words.len();

            for i in 0..word_count {
                let word = words[i];

                // Check if the current word starts with an uppercase letter
                let first_upper = word.chars().next().map_or(false, |c| c.is_uppercase());

                if first_upper {
                    let word_str = word.to_string();

                    if i == 0 {
                        // High-noise area. We track these to know if a word ONLY
                        // appears here (likely a common noun like "But", "Then").
                        self.sentence_starters.insert(word_str);
                    } else {
                        let evidence = self.candidates.entry(word_str).or_insert(EntityEvidence::default());
                        evidence.count += 1;
                        if evidence.snippets.len() < 10 {
                            let clean_sentence = sentence.replace('\n', " ").trim().to_string();
                            evidence.snippets.push(clean_sentence);
                        }
                        // If the NEXT word is also Uppercase, we capture it as a composite entity.
                        if i + 1 < word_count {
                            let next_word = words[i + 1];
                            let next_upper = next_word.chars().next().map_or(false, |c| c.is_uppercase());

                            if next_upper {
                                // Both words are capitalized. High confidence for
                                // proper names (e.g., "Maxim Kammerer") or lore titles.
                                let bigram = format!("{} {}", word, next_word);
                                *self.bigrams.entry(bigram).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn get_top(&self, q: u32) -> Vec<(&String, &EntityEvidence)> {
        let top = self.candidates.iter()
            .filter(|&(name, e)| e.count >= q && !self.sentence_starters.contains(name))
            .map(|(name, e)|(name,e))
            .collect();
        top
    }

    pub fn print_report(&self, min_freq: u32) {
        println!("--- Top Detected Entities (Excluding starters) ---");
        let mut top: Vec<_> = self.candidates.iter()
            .filter(|&(ref name, e)| e.count >= min_freq && !self.sentence_starters.contains(*name))
            .collect();
        top.sort_by(|a, b| b.1.count.cmp(&a.1.count));

        for (name, e) in top.iter().take(30) {
            println!("{}: {:?}", name, e);
        }

        println!("\n--- Top Candidate Bigrams ---");
        let mut top_bi: Vec<_> = self.bigrams.iter()
            .filter(|&(_, &count)| count >= min_freq)
            .collect();
        top_bi.sort_by(|a, b| b.1.cmp(&a.1));

        for (name, count) in top_bi.iter().take(20) {
            println!("{}: {}", name, count);
        }
    }
}