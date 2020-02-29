use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::RandomState;
use indexmap::IndexSet;
use regex::Regex;
use std::io;
use std::borrow;
use bitvec::prelude::{bitbox, Lsb0, BitBox};

use fnv::{FnvHasher, FnvHashSet};
use std::{collections::HashSet, hash::BuildHasherDefault};
type FnvBuilder = BuildHasherDefault<FnvHasher>;
type FnvIndexSet<K> = IndexSet<K, FnvBuilder>;

// Seed for the random number generator, for consistent output.
// Comment from original code follows:
// from OEIS A001519, the title should be "Itera Aeno", md5sum = 4dcf116dc35156ec939f8cafd61bdf18
const RANDOM_SEED: u32 = 63245986;

const BOOK_SIZE: usize = 1<<30;               // 1 GB
const DISTINCT_WORDS: usize = 5000000;        // bigger number will allow longer longest word
const MEAN: usize = 15000;                    // bigger number will increase average word length
const LAMBDA: f64 = 1.0 / (MEAN as f64);

const VOWELS: &str = "aeiouy";
const FORBIDDEN_REGEX: &str = "satan|lenin|stalin|hitl|naz|rus|putin";

const TAB: &str = "   ";
const LINE_WIDTH: usize = 76;

#[derive(Default)]
struct MarkovNode {
    total: u64,
    letters: [Option<Box<MarkovNode>>; 26],
}

// Helpers for indexing into MarkovNode.letters
fn index_to_char(i: usize) -> char { (b'a' + i as u8) as char }
fn byte_to_index(b: u8) -> usize { b as usize - b'a' as usize }
fn char_to_index(c: char) -> usize { c as usize - b'a' as usize }

struct Markov {
    root: MarkovNode,
}

impl Markov {

    fn new() -> Markov {
        Markov {
            root: Default::default(),
        }
    }

    fn train(&mut self, word: &str) {
        let mut m = &mut self.root;
        m.total += 1;
        for &b in word.as_bytes() {
            m = m.letters[byte_to_index(b)].get_or_insert_with(Default::default);
            m.total += 1;
        }
    }

    fn next_letter(&self, word: &str, random: &mut RandomState) -> char {
        let mut m = &self.root;

        for &b in word.as_bytes().get(word.len().saturating_sub(2)..).unwrap() {
            // This assumes the letter must be present in the root node.
            // This is not guaranteed to be the case, but with the chosen random
            // seed the unwrap() happens to never fail.
            m = m.letters[byte_to_index(b)].as_ref().unwrap_or_else(|| self.root.letters[byte_to_index(b)].as_ref().unwrap());
        }

        let mut num = random.randint(0, m.total - 1);

        for index in 0..m.letters.len() {
            if let Some(ref x) = &m.letters[index] {
                if x.total > num {
                    return index_to_char(index);
                }
                num -= x.total;
            }
        }

        // Totals of children do not sum to the total of the parent, but the
        // code above assumes it does. This is definitely a bug, but it's just
        // never hit with the hardcoded random number generator seed...
        panic!("inconsistent tree");
    }
}


struct Book {
    title: String,
    author: String,
    year: String,
    verlag: String,
    line: String,
    front: bool,
    capitalize: bool,
    counter: Vec<usize>,
    words: Vec<String>,
    length: usize,
}

impl Book {
    fn new(words: Vec<String>) -> Book {
        Book {
            title: Default::default(),
            author: Default::default(),
            year: "2019".to_owned(),
            verlag: Default::default(),
            line: Default::default(),
            front: false,
            capitalize: true,
            counter: vec![0; words.len()],
            words,
            length: 0,
        }
    }

    fn len(&self) -> usize {
        self.length
    }

    fn print_front(&mut self, write: &mut dyn Write) -> io::Result<()> {
        writeln!(write, "{}\n", self.title.to_uppercase().trim_end())?;

        writeln!(write, "{}\n", self.author.trim_end())?;

        writeln!(write, "(c) {}, {}, Public domain\n", self.year, self.verlag)?;

        self.line.clear();
        self.line += TAB;
        self.front = true;

        Ok(())
    }

    fn next_word<W: Write>(&mut self, random: &mut RandomState, write: &mut W) -> io::Result<()> {
        let mut i = random.expovariate(LAMBDA) as usize;
        while i >= self.words.len() {
            i = random.expovariate(LAMBDA) as usize;
        }
        let word = &self.words[i];

        self.counter[i] += 1;

        // Put word in a Cow (copy-on-write) so we can avoid copying it if we
        // don't end up making any changes to it such as capitalization or
        // adding punctuation.
        let mut word = borrow::Cow::from(word);
        if self.capitalize {
            word.to_mut()[0..1].make_ascii_uppercase();

            // Only capitalize one word at a time, except if we're still in
            // the front matter block, in that case following words should
            // still be capitalized as well.
            // In the original code capitalization is handled separately for
            // front matter and for normal text.
            if self.front {
                self.capitalize = false;
            }
        }

        if !self.front {
            if self.title.split(' ').count() < 3 {
                self.title += &word;
                self.title.push(' ');
            }
            else if self.author.split(' ').count() < 3 {
                self.author += &word;
                self.author.push(' ');
            }
            else {
                self.verlag += &word;
                self.print_front(write)?;
            }
            return Ok(());
        }

        let mut paragraph = false;
        if random.randint(0, 9) == 0 {
            word.to_mut().push(',');
        }
        else if random.randint(0, 9) == 0 {
            word.to_mut().push('.');
            self.capitalize = true;
            if random.randint(0, 9) == 0 {
                paragraph = true;
            }
        }

        if self.line.len() + 1 + word.len() > LINE_WIDTH {
            self.length += self.line.len() + 1;
            writeln!(write, "{}", &self.line)?;
            self.line.clear();
            self.line += &word;
        }
        else if paragraph {
            self.line.push(' ');
            self.line += &word;
            self.line.push('\n');
            self.length += self.line.len() + 1;
            writeln!(write, "{}", &self.line)?;
            self.line.clear();
            self.line += TAB;
        }
        else {
            self.line.push(' ');
            self.line += &word;
        }

        Ok(())
    }

    fn end(&mut self, write: &mut dyn Write) -> io::Result<()> {
        writeln!(write, "{}.", &self.line.trim_end())?;

        let mut tmp = self.counter.iter().zip(self.words.iter()).collect::<Vec<_>>();
        tmp.as_mut_slice().sort_by_key(|&(c, _)| c);
        writeln!(write, "\n--\n\n\n\n\n\n\nMost common words:")?;
        for (_, w) in tmp.iter().rev().take(10) {
            writeln!(write, "- {}", &w)?;
        }

        Ok(())
    }

}

fn main() -> Result<(), Box<dyn std::error::Error>>  {


    println!("Getting the reference text");

    let data = fs::read("11940-8.txt")?;

    let text = decode_latin1(&data).to_lowercase();
    let text = text.replace("ä", "a").replace("å", "a").replace("ö", "o");

    let mut slice = &text[..];
    slice = &slice[slice.find("start of th").unwrap()..];
    slice = &slice[slice.find("\n").unwrap()+1..];
    slice = &slice[..slice.find("end of th").unwrap()];
    slice = &slice[..slice.rfind("\n").unwrap()];
    slice = slice.trim();
    slice = &slice[slice.find("\n").unwrap()+1..];
    slice = &slice[slice.find("\n").unwrap()+1..];

    println!("Getting reference words");

    let all_words = slice.split(|c: char| !c.is_ascii_lowercase()).collect::<FnvHashSet<_>>();

    println!("{} reference words", all_words.len());

    println!("Training Markov chain");

    let mut markov = Markov::new();

    for word in &all_words {
        let mut word = &word[..];
        while word.len() >= 3 {
            markov.train(word);
            word = &word[1..];
        }
    }

    // let mut f = fs::File::create("/tmp/markov_rust.txt").unwrap();
    // fn dump_markov(f: &mut fs::File, m: &MarkovNode, indent: &str) {
    //     write!(f, "{}total: {} / sum: {}\n", indent, m.total, &m.letters.iter().map(|x| x.as_ref().map_or(0, |x| x.total)).sum::<u64>());
    //     for c in b'a'..=b'z' {
    //         let index = (c - b'a') as usize;
    //         if let Some(x) = &m.letters[index] {
    //             let new_indent = format!("{}{}:  ", indent, c as char);
    //             dump_markov(f, &*x, &new_indent);
    //         }
    //     }
    // }
    // dump_markov(&mut f, &markov.root, "");


    println!("Generating artificial words");

    // Initialize python-compatible RNG
    let mut random = RandomState::new();
    random.seed_u32(RANDOM_SEED);

    let mut word_set = FnvHashSet::<String>::with_capacity_and_hasher(DISTINCT_WORDS, Default::default());

    // With 6 entries this uses about 40mb of ram, with 7 it's about a gigabyte.
    let mut word_set_short = [
        bitbox![Lsb0, u64; 0; 26],
        bitbox![Lsb0, u64; 0; 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26 * 26 * 26 * 26],
    ];

    let mut word_list = Vec::<String>::with_capacity(DISTINCT_WORDS);

    let forbidden_regex = Regex::new(FORBIDDEN_REGEX).unwrap();

    const SHORT_CUTOFF: usize = 7;
    assert!(SHORT_CUTOFF == word_set_short.len());

    let mut w = String::new();
    let mut h: usize;

    let mut short_reads: usize = 0;
    let mut short_writes: usize = 0;
    let mut long_reads: usize = 0;
    let mut long_writes: usize = 0;

    while word_list.len() < DISTINCT_WORDS {

        let mut letter;

        w.clear();

        // Bug in original code: as soon as it finds a word that isn't in
        // the hashset yet it will exit this loop. But if the word contains
        // no vowels, it will not be inserted, so next time the loop will
        // break at that point again, instead of iterating further and
        // maybe getting a vowel. End result: all words start with a vowel.
        // We can exploit knowledge of this bug to skip some work.
        loop {
            letter = markov.next_letter(&w, &mut random);
            if VOWELS.contains(letter) {
                break;
            }
        }

        w.push(letter);
        h = char_to_index(letter);

        while if w.len() <= SHORT_CUTOFF { short_reads += 1; word_set_short[w.len()-1][h] } else { long_reads += 1; word_set.contains(&w) } {
            letter = markov.next_letter(&w, &mut random);
            w.push(letter);
            if w.len() <= SHORT_CUTOFF {
                h = h * 26 + char_to_index(letter);
            }
        }

        if !forbidden_regex.is_match(&w) {
            if w.len() <= SHORT_CUTOFF {
                word_set_short[w.len()-1].set(h, true);
                short_writes += 1;
            } else {
                word_set.insert(w.clone());
                long_writes += 1;
            }
            word_list.push(w.clone());
            if word_list.len() % 100000 == 0 {
                println!("{} words generated", word_list.len());
            }
        }
    }

    println!("writes: long: {:-10}, short: {:-10}", long_writes, short_writes);
    println!(" reads: long: {:-10}, short: {:-10}", long_reads, short_reads);

    println!("Capitalizing some words");

    for i in 0..word_list.len() {
        if random.randint(0, 100) == 0 {
            word_list[i][0..1].make_ascii_uppercase();
        }
    }

    println!("Generating text");

    let mut f = io::BufWriter::new(fs::File::create("/tmp/giganovel.txt").unwrap());

    let mut book = Book::new(word_list);

    while book.len() < BOOK_SIZE {
        book.next_word(&mut random, &mut f)?;
    }
    book.end(&mut f)?;

    println!("rand iterations: {}", random.counter());

    Ok(())
}
