use std::path::PathBuf;
use anyhow::{Result, Context as _};
use fb2::{FictionBook, SectionPart, CiteElement, PoemStanza, TitleElement, StyleElement, StyleLinkElement};

pub struct Fb2Scan {
    sections: Vec<Section>,
}

impl Fb2Scan {
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let book: FictionBook = quick_xml::de::from_reader(reader).context("Failed to parse FB2")?;
        Ok(Self::from_book(book))
    }

    pub fn from_text(xml: &str) -> Result<Self> {
        let book: FictionBook = quick_xml::de::from_str(xml).context("Failed to parse FB2")?;
        Ok(Self::from_book(book))
    }
}

pub struct Section {
    title: Option<String>,
    text: String,
    language: Option<String>,
    segment_indices: Vec<(usize, usize)>,
}

impl Section {
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn pars(&self) -> Vec<&str> {
        self.segment_indices.iter().map(|&(start, end)| &self.text[start..end]).collect::<Vec<_>>()
    }
    
    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

fn link_elements_to_text(elements: &[StyleLinkElement]) -> String {
    let mut s = String::new();
    for elem in elements {
        match elem {
            StyleLinkElement::Strong { elements }
            | StyleLinkElement::Emphasis { elements }
            | StyleLinkElement::Style { elements }
            | StyleLinkElement::Strikethrough { elements }
            | StyleLinkElement::Subscript { elements }
            | StyleLinkElement::Superscript { elements }
            | StyleLinkElement::Code { elements } => {
                s.push_str(&link_elements_to_text(elements));
            }
            StyleLinkElement::Image(_) => {}
            StyleLinkElement::Text(t) => s.push_str(t),
        }
    }
    s
}

fn style_elements_to_md(elements: &[StyleElement]) -> String {
    let mut s = String::new();
    for elem in elements {
        match elem {
            StyleElement::Text(t) => s.push_str(t),
            StyleElement::Strong(style) => {
                let inner = style_elements_to_md(&style.elements);
                let trimmed = inner.trim();
                if !trimmed.is_empty() {
                    s.push_str("**");
                    s.push_str(trimmed);
                    s.push_str("**");
                }
            }
            StyleElement::Emphasis(style) => {
                let inner = style_elements_to_md(&style.elements);
                let trimmed = inner.trim();
                if !trimmed.is_empty() {
                    s.push('*');
                    s.push_str(trimmed);
                    s.push('*');
                }
            }
            StyleElement::Strikethrough(style) => {
                let inner = style_elements_to_md(&style.elements);
                let trimmed = inner.trim();
                if !trimmed.is_empty() {
                    s.push_str("~~");
                    s.push_str(trimmed);
                    s.push_str("~~");
                }
            }
            StyleElement::Code(style) => {
                let inner = style_elements_to_md(&style.elements);
                let trimmed = inner.trim();
                if !trimmed.is_empty() {
                    s.push('`');
                    s.push_str(trimmed);
                    s.push('`');
                }
            }
            StyleElement::Subscript(style) | StyleElement::Superscript(style) => {
                s.push_str(&style_elements_to_md(&style.elements));
            }
            StyleElement::Style(named) => {
                s.push_str(&style_elements_to_md(&named.elements));
            }
            StyleElement::Link(link) => {
                s.push_str(&link_elements_to_text(&link.elements));
            }
            StyleElement::Image(_) => {}
        }
    }
    s
}

fn para_to_md(para: &fb2::Paragraph) -> String {
    style_elements_to_md(&para.elements)
}

fn title_to_text(title: &fb2::Title) -> String {
    title
        .elements
        .iter()
        .filter_map(|e| match e {
            TitleElement::Paragraph(p) => {
                let t = para_to_md(p);
                let trimmed = t.trim().to_string();
                if trimmed.is_empty() { None } else { Some(trimmed) }
            }
            TitleElement::EmptyLine => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn append_segment(text: &mut String, indices: &mut Vec<(usize, usize)>, content: &str) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    let start = text.len();
    text.push_str(trimmed);
    text.push_str("\n\n");
    let end = text.len();
    indices.push((start, end));
}

fn process_poem(poem: &fb2::Poem, text: &mut String, indices: &mut Vec<(usize, usize)>) {
    for stanza in &poem.stanzas {
        match stanza {
            PoemStanza::Subtitle(p) => {
                let content = para_to_md(p);
                let trimmed = content.trim().to_string();
                if !trimmed.is_empty() {
                    append_segment(text, indices, &format!("*{}*", trimmed));
                }
            }
            PoemStanza::Stanza(s) => {
                let lines: Vec<String> = s
                    .lines
                    .iter()
                    .map(|l| para_to_md(l))
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| format!("*{}*", l.trim()))
                    .collect();
                if !lines.is_empty() {
                    append_segment(text, indices, &lines.join("  \n"));
                }
            }
        }
    }
}

fn process_cite(cite: &fb2::Cite, text: &mut String, indices: &mut Vec<(usize, usize)>) {
    for elem in &cite.elements {
        match elem {
            CiteElement::Paragraph(p) | CiteElement::Subtitle(p) => {
                let content = para_to_md(p);
                let trimmed = content.trim().to_string();
                if !trimmed.is_empty() {
                    append_segment(text, indices, &format!("*{}*", trimmed));
                }
            }
            CiteElement::Poem(poem) => {
                process_poem(poem, text, indices);
            }
            CiteElement::Table(_) | CiteElement::EmptyLine => {}
        }
    }
}

fn process_section_part(part: &SectionPart, text: &mut String, indices: &mut Vec<(usize, usize)>) {
    match part {
        SectionPart::Paragraph(p) => {
            let content = para_to_md(p);
            append_segment(text, indices, &content);
        }
        SectionPart::Subtitle(p) => {
            let content = para_to_md(p);
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                append_segment(text, indices, &format!("## {}", trimmed));
            }
        }
        SectionPart::Cite(cite) => {
            process_cite(cite, text, indices);
        }
        SectionPart::Poem(poem) => {
            process_poem(poem, text, indices);
        }
        SectionPart::Image(_) | SectionPart::Table(_) | SectionPart::EmptyLine => {}
    }
}

fn collect_sections(fb2_section: &fb2::Section, out: &mut Vec<Section>) {
    if let Some(content) = &fb2_section.content {
        let title = content
            .title
            .as_ref()
            .map(title_to_text)
            .filter(|s| !s.is_empty());

        let mut text = String::new();
        let mut segment_indices = Vec::new();

        for part in &content.content {
            process_section_part(part, &mut text, &mut segment_indices);
        }

        if !text.is_empty() || title.is_some() {
            let lang = whatlang::detect(&text).map(|x| x.lang().eng_name().to_string());
            out.push(Section { title, text, segment_indices, language: lang });
        }

        for subsection in &content.sections {
            collect_sections(subsection, out);
        }
    }
}

impl Fb2Scan {
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    fn from_book(book: FictionBook) -> Self {
        let mut sections = Vec::new();
        for body in &book.bodies {
            if body.name.is_some() {
                continue;
            }
            for fb2_section in &body.sections {
                collect_sections(fb2_section, &mut sections);
            }
        }
        Self { sections }
    }
}

#[cfg(test)]
mod tests;
