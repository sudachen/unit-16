use super::*;

const HEADER: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:l="http://www.w3.org/1999/xlink">
  <description>
    <title-info>
      <genre>prose</genre>
      <author><first-name>A</first-name><last-name>B</last-name></author>
      <book-title>Test</book-title>
      <lang>en</lang>
    </title-info>
  </description>"#;

fn parse_sections(body_inner: &str) -> Vec<Section> {
    let xml = format!("{}\n<body>{}</body>\n</FictionBook>", HEADER, body_inner);
    let scan = Fb2Scan::from_text(&xml).expect("parse failed");
    scan.sections
}

#[test]
fn plain_paragraph() {
    let secs = parse_sections("<section><p>Hello world</p></section>");
    assert_eq!(secs.len(), 1);
    assert_eq!(secs[0].text().trim(), "Hello world");
    assert_eq!(secs[0].pars().len(), 1);
}

#[test]
fn bold_inline() {
    let secs = parse_sections("<section><p>A <strong>bold</strong> B</p></section>");
    assert!(secs[0].text().contains("**bold**"), "got: {:?}", secs[0].text());
}

#[test]
fn emphasis_inline() {
    let secs = parse_sections("<section><p>A <emphasis>em</emphasis> B</p></section>");
    assert!(secs[0].text().contains("*em*"), "got: {:?}", secs[0].text());
}

#[test]
fn strikethrough_inline() {
    let secs = parse_sections("<section><p>A <strikethrough>del</strikethrough> B</p></section>");
    assert!(secs[0].text().contains("~~del~~"), "got: {:?}", secs[0].text());
}

#[test]
fn code_inline() {
    let secs = parse_sections("<section><p>A <code>fn x()</code> B</p></section>");
    assert!(secs[0].text().contains("`fn x()`"), "got: {:?}", secs[0].text());
}

#[test]
fn subtitle_becomes_heading() {
    let secs = parse_sections("<section><subtitle>Chapter One</subtitle></section>");
    assert!(secs[0].text().contains("## Chapter One"), "got: {:?}", secs[0].text());
}

#[test]
fn cite_paragraph_italic() {
    let secs = parse_sections("<section><cite><p>Quoted text</p></cite></section>");
    assert!(secs[0].text().contains("*Quoted text*"), "got: {:?}", secs[0].text());
}

#[test]
fn image_skipped() {
    let body = r##"<section><image l:href="#img.jpg"/><p>After image</p></section>"##;
    let secs = parse_sections(body);
    assert_eq!(secs.len(), 1);
    assert!(!secs[0].text().contains("img"), "image leaked: {:?}", secs[0].text());
    assert!(secs[0].text().contains("After image"));
}

#[test]
fn section_title_extracted() {
    let secs = parse_sections(
        "<section><title><p>My Chapter</p></title><p>Content</p></section>",
    );
    assert_eq!(secs[0].title(), Some("My Chapter"));
    assert!(secs[0].text().contains("Content"));
}

#[test]
fn multiple_top_sections() {
    let secs = parse_sections(
        "<section><title><p>Ch 1</p></title><p>First</p></section>\
         <section><title><p>Ch 2</p></title><p>Second</p></section>",
    );
    assert_eq!(secs.len(), 2);
    assert_eq!(secs[0].title(), Some("Ch 1"));
    assert_eq!(secs[1].title(), Some("Ch 2"));
}

#[test]
fn named_body_skipped() {
    let xml = format!(
        "{}\n<body><section><p>Main</p></section></body>\
         <body name=\"notes\"><section><p>Notes</p></section></body>\n</FictionBook>",
        HEADER
    );
    let scan = Fb2Scan::from_text(&xml).expect("parse failed");
    assert_eq!(scan.sections().len(), 1);
    assert!(scan.sections()[0].text().contains("Main"));
}

#[test]
fn segment_count_matches_paragraphs() {
    let secs = parse_sections("<section><p>One</p><p>Two</p><p>Three</p></section>");
    assert_eq!(secs[0].pars().len(), 3);
}

#[test]
fn segment_content_by_index() {
    let secs = parse_sections("<section><p>Alpha</p><p>Beta</p></section>");
    let pars = secs[0].pars();
    assert_eq!(pars.len(), 2);
    assert_eq!(pars[0].trim(), "Alpha");
    assert_eq!(pars[1].trim(), "Beta");
}

#[test]
fn empty_lines_not_segments() {
    let secs = parse_sections("<section><p>A</p><empty-line/><p>B</p></section>");
    assert_eq!(secs[0].pars().len(), 2);
}

#[test]
fn poem_stanza_lines_italic() {
    let secs = parse_sections(
        "<section><poem><stanza><v>Line one</v><v>Line two</v></stanza></poem></section>",
    );
    let text = secs[0].text();
    assert!(text.contains("*Line one*"), "got: {:?}", text);
    assert!(text.contains("*Line two*"), "got: {:?}", text);
}

#[test]
fn recursive_subsections_flattened() {
    let secs = parse_sections(
        "<section>\
           <title><p>Parent</p></title>\
           <section><title><p>Child</p></title><p>Child content</p></section>\
         </section>",
    );
    assert_eq!(secs.len(), 2);
    assert_eq!(secs[0].title(), Some("Parent"));
    assert_eq!(secs[1].title(), Some("Child"));
    assert!(secs[1].text().contains("Child content"));
}
