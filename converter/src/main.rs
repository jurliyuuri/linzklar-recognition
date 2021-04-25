#![warn(clippy::pedantic)]
use std::fs::File;
use std::io::prelude::*;

use serde_derive::{Deserialize as De, Serialize as Ser};

#[derive(Ser, De)]
struct CharData {
    character: String,
    data: Vec<Vec<Coord>>,
    #[serde(rename = "initialDotCaptured")] // to comply with Rust coding standards
    initial_dot_captured: Option<bool>,
}

#[derive(Ser, De, Copy, Clone)]
struct Coord {
    x: f64,
    y: f64,
}

use std::ops::Sub;

impl Sub for Coord {
    type Output = Coord;
    fn sub(self, other: Coord) -> Coord {
        Coord {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

fn gen_svg(strokes: &[Vec<Coord>]) -> String {
    let mut ans = r#"<?xml version="1.0" standalone="no"?>
    <svg width="256" height="256" xmlns="http://www.w3.org/2000/svg" version="1.1">
    "#
    .to_string();
    for stroke in strokes {
        ans.push_str(
            r#"<polyline stroke="black" stroke-width="3" stroke-linecap="round" fill="transparent" stroke-linejoin="round" points=""#,
        );
        for coord in stroke {
            ans.push_str(&format!("{} {} ", coord.x, coord.y));
        }
        ans.push_str(
            r#"" />
        "#,
        );
    }
    ans.push_str("</svg>");
    ans
}

fn main() -> std::io::Result<()> {
    use std::collections::HashMap;
    use std::env;

    let args: Vec<String> = env::args().collect();
    let dry_run: bool = args.contains(&String::from("--dry-run"));

    if dry_run {
        println!("dry run: no svgs will be generated.");
    }

    let write_svg = if dry_run { nothing } else { write_svg_ };

    let mut file = File::open("../datalist.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let filenames: Vec<String> = serde_json::from_str(&contents).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Failed to interpret ../datalist.json",
        )
    })?;

    let mut total = 0;
    let mut char_count = HashMap::new();
    let len = filenames.len();

    for (i, src) in filenames.iter().enumerate() {
        let mut datasetfile = File::open(format!("../data/{}", src))?;
        let mut datasetcontents = String::new();
        datasetfile.read_to_string(&mut datasetcontents)?;
        let characters: Vec<CharData> = serde_json::from_str(&datasetcontents).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to interpret ../data/{}", src),
            )
        })?;
        println!(
            "({:>3}/{}) {} {:>5} characters in {}.",
            i,
            len,
            if dry_run { "Found" } else { "Converting" },
            characters.len(),
            src
        );
        total += characters.len();
        for (i, c) in characters.iter().enumerate() {
            *char_count.entry(c.character.clone()).or_insert(0) += 1;

            if c.initial_dot_captured == Some(true) {
                write_svg(&c.character, "initial_dot_captured", &src, i, &c.data)?;

                write_svg(
                    &c.character,
                    "initial_dot_captured_or_augmented",
                    &src,
                    i,
                    &c.data,
                )?;

                let mut initial_dot_omitted = Vec::new();
                for stroke in &c.data {
                    let mut k = stroke.clone();
                    k.remove(0);
                    initial_dot_omitted.push(k);
                }

                write_svg(
                    &c.character,
                    "initial_dot_omitted",
                    &src,
                    i,
                    &initial_dot_omitted,
                )?;
            } else {
                write_svg(&c.character, "initial_dot_omitted", &src, i, &c.data)?;

                let mut initial_dot_augmented = Vec::new();
                for stroke in &c.data {
                    let mut k = stroke.clone();
                    if k.len() < 2 {
                        // cannot augment
                        initial_dot_augmented.push(k);
                    } else {
                        let stroke_neg1 = stroke[0] - (stroke[1] - stroke[0]);
                        k.insert(0, stroke_neg1);
                        initial_dot_augmented.push(k);
                    }
                }

                write_svg(
                    &c.character,
                    "initial_dot_captured_or_augmented",
                    &src,
                    i,
                    &initial_dot_augmented,
                )?;
            }
        }
    }

    let mut count_vec: Vec<_> = char_count.iter().collect();
    count_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (c, count) in count_vec {
        println!("{}, {}", c, count);
    }
    
    if !dry_run {
        println!("Converted {} characters into svg.", total);
    }

    Ok(())
}

fn nothing(
    _transcription: &str,
    _folder_name: &str,
    _src: &str,
    _i: usize,
    _strokes: &[Vec<Coord>],
) -> std::io::Result<()> {
    Ok(())
}

fn write_svg_(
    transcription: &str,
    folder_name: &str,
    src: &str,
    i: usize,
    strokes: &[Vec<Coord>],
) -> std::io::Result<()> {
    use std::fs;
    fs::create_dir_all(format!(
        "../data_images/svg/{}/{}",
        folder_name, transcription
    ))?;

    fs::write(
        format!(
            "../data_images/svg/{}/{}/{}__{}.svg",
            folder_name, transcription, src, i
        ),
        gen_svg(strokes),
    )?;

    Ok(())
}
