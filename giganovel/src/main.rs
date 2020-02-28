use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::RandomState;
use indexmap::IndexSet;
use regex::Regex;
use std::io;

use fnv::{FnvHasher, FnvHashSet};
use std::hash::BuildHasherDefault;
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
        for letter in word.chars() {
            let index = letter as usize - 'a' as usize;
            m = m.letters[index].get_or_insert_with(Default::default);
            m.total += 1;
        }
    }

    fn next_letter(&self, word: &str, random: &mut RandomState) -> char {
        let mut m = &self.root;

        for letter in word.get(word.len().saturating_sub(2)..).unwrap().chars() {
            let index = letter as usize - 'a' as usize;

            // This assumes the letter must be present in the root node.
            // This is not guaranteed to be the case, but with the chosen random
            // seed the unwrap() happens to never fail.
            m = m.letters[index].as_ref().unwrap_or_else(|| self.root.letters[index].as_ref().unwrap());
        }

        let mut i = random.randint(0, m.total - 1);
        for index in 0..m.letters.len() {
            if let Some(ref x) = &m.letters[index] {
                if x.total > i {
                    return (b'a' + index as u8) as char;
                }
                i -= x.total;
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

        self.line = TAB.to_owned();
        self.front = true;

        Ok(())
    }

    fn next_word(&mut self, random: &mut RandomState, write: &mut dyn Write) -> io::Result<()> {
        let mut i = random.expovariate(LAMBDA) as usize;
        while i >= self.words.len() {
            i = random.expovariate(LAMBDA) as usize;
        }
        let word = &self.words[i];

        self.counter[i] += 1;

        let mut capitalized_word = word.clone();
        capitalized_word[0..1].make_ascii_uppercase();
        if !self.front {
            if self.title.split(' ').count() < 3 {
                self.title += &capitalized_word;
                self.title.push(' ');
            }
            else if self.author.split(' ').count() < 3 {
                self.author += &capitalized_word;
                self.author.push(' ');
            }
            else {
                self.verlag += &capitalized_word;
                self.print_front(write)?;
            }
            return Ok(());
        }
        let mut word = word.to_owned();
        if self.capitalize {
            word = capitalized_word.clone();
            self.capitalize = false;
        }
        let mut paragraph = false;
        if random.randint(0, 9) == 0 {
            word.push(',');
        }
        else if random.randint(0, 9) == 0 {
            word.push('.');
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

    // IndexSet preserves insertion order as long as you don't delete anything,
    // so it can replace the separate list and set here.
    // TODO Probably did something wrong with the type, I don't understand why
    //      it doesn't work with just `new` instead of `with_capacity_and_hasher`.
    let mut words = FnvIndexSet::<String>::with_capacity_and_hasher(DISTINCT_WORDS, Default::default());

    let forbidden_regex = Regex::new(FORBIDDEN_REGEX).unwrap();

    while words.len() < DISTINCT_WORDS {
        let mut w = String::new();
        let mut has_vowel = false;
        while w.len() == 0 || words.contains(&w) {
            let letter = markov.next_letter(&w, &mut random);
            w.push(letter);
            has_vowel = has_vowel || VOWELS.contains(letter);
        }
        if has_vowel && !forbidden_regex.is_match(&w) {
            words.insert(w);
            if words.len() % 100000 == 0 {
                println!("{} words generated", words.len());
            }
        }
    }

    println!("Capitalizing some words");

    // We want to change some keys, which sets don't allow.
    // We don't need the set functionality anymore so make it a Vec.
    let mut word_list = words.into_iter().collect::<Vec<_>>();

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

    Ok(())
}
