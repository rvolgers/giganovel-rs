use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::{PythonRandom, MersenneTwister};
use regex::Regex;
use std::io;
use std::borrow;
use bitvec::prelude::{bitbox, Lsb0};
use fnv::FnvHashSet;

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
        for b in word.bytes() {
            m = m.letters[byte_to_index(b)].get_or_insert_with(Default::default);
            m.total += 1;
        }
    }

    fn next_letter(&self, word: &str, random: &mut PythonRandom) -> char {
        let mut m = &self.root;

        for b in word.bytes().rev().take(2).rev() {
            m = m.letters[byte_to_index(b)].as_ref()
                    .or_else(|| self.root.letters[byte_to_index(b)].as_ref())
                    // Bug in original implementation:
                    // This assumes the letter must be present in the root node.
                    // This is not guaranteed to be the case, but with the chosen
                    // random seed the unwrap() happens to never fail.
                    .unwrap();
        }

        assert!(m.total > 0);
        let mut num = random.randint(0, m.total - 1);

        for index in 0..m.letters.len() {
            if let Some(ref x) = &m.letters[index] {
                if x.total > num {
                    return index_to_char(index);
                }
                num -= x.total;
            }
        }

        // Bug in original implementation:
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

    fn next_word<W: Write>(&mut self, random: &mut PythonRandom, write: &mut W) -> io::Result<()> {
        // Pick a random word. This was moved to next_word so we can use the
        // generated index to count how often each word is used without needing
        // a separate hash lookup to get the count for a word.
        let word;
        loop {
            let i = random.expovariate(LAMBDA) as usize;
            if i < self.words.len() {
                word = &self.words[i];
                self.counter[i] += 1;
                break;
            }
        }

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
            if self.title.matches(' ').count() < 2 {
                self.title += &word;
                self.title.push(' ');
            }
            else if self.author.matches(' ').count() < 2 {
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
            self.line.push('\n');
            self.length += self.line.len();
            write.write_all(self.line.as_bytes())?;
            self.line.clear();
            self.line += &word;
        }
        else if paragraph {
            self.line.push(' ');
            self.line += &word;
            self.line.push('\n');
            self.line.push('\n');
            self.length += self.line.len();
            write.write_all(self.line.as_bytes())?;
            self.line.clear();
            self.line += TAB;
        }
        else {
            self.line.push(' ');
            self.line += &word;
        }

        Ok(())
    }

    fn end<W: Write>(&mut self, write: &mut W) -> io::Result<()> {
        writeln!(write, "{}.", &self.line.trim_end())?;

        let mut tmp = self.counter.iter()
            .zip(self.words.iter())
            .collect::<Vec<_>>();

        tmp.sort_by_key(|&(c, _)| c);

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

    println!("{}", &slice[..slice.char_indices().nth(101).unwrap().0]);
    println!("...");
    println!("{}", &slice[slice.char_indices().rev().nth(99).unwrap().0..]);

    println!("Getting reference words");

    // Order is not important here, so ensure uniqueness by collecting into a set.
    let all_words = slice.trim()
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|s| s.len() > 0)
        .collect::<FnvHashSet<&str>>();

    println!("{} reference words", all_words.len());

    println!("Training Markov chain");

    let mut markov = Markov::new();

    for mut word in all_words {
        while word.len() >= 3 {
            markov.train(word);
            word = &word[1..];
        }
    }

    println!("Generating artificial words");

    // Initialize python-compatible RNG
    let mt = MersenneTwister::new();
    let mut random = PythonRandom::new(mt);
    random.seed_u32(RANDOM_SEED);

    // For words with length <= 7 we encode the word as an index in a bitset directly to check
    // if we've seen it before. For larger words we use a pretty normal HashSet (but with a
    // hasher that is not built for security and is favorable for short strings.)
    // With for length <= 6 the bitsets use about 40mb of ram total, with <= 7 it's about a gigabyte.
    // You can just add or remove entries from the bottom to tweak this tradeoff.
    let mut word_set = FnvHashSet::<String>::with_capacity_and_hasher(DISTINCT_WORDS, Default::default());
    let mut word_set_short = [
        bitbox![Lsb0, u64; 0; 0], 
        bitbox![Lsb0, u64; 0; 26],
        bitbox![Lsb0, u64; 0; 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26 * 26 * 26],
        bitbox![Lsb0, u64; 0; 26 * 26 * 26 * 26 * 26 * 26 * 26],
    ];

    // The actual ordered list of words we will use later to pick words based on random numbers.
    // We could use a set that preserves insertion order, but the hash lookups we save by using
    // the word_set_short optimization outweighs the benefit of not duplicating storage.
    let mut word_list = Vec::<String>::with_capacity(DISTINCT_WORDS);

    // Regex to match forbidden substrings
    let forbidden_regex = Regex::new(FORBIDDEN_REGEX).unwrap();

    // String that holds the current word.
    let mut w = String::new();

    // The word encoded as a numeric id for indexing into word_set_short.
    let mut h: usize;

    while word_list.len() < DISTINCT_WORDS {

        w.clear();
        h = 0;

        // Add more letters until we find a word that hasn't been accepted yet.
        // (Could still be a word that will never be accepted due to later checks.)
        loop {
            // Add a new letter to the word and update the word_set_short index.
            let letter = markov.next_letter(&w, &mut random);
            w.push(letter);

            // Have we seen this word before? If so, break.
            if w.len() < word_set_short.len() {
                h = h * 26 + char_to_index(letter);
                if !word_set_short[w.len()][h] { break; }
            } else {
                if !word_set.contains(&w) { break; }
            }
        }

        // Check for vowels and forbidden words.
        // Bug in original implementation:
        // The word generation loop ends as soon as it produces a word that has
        // not previously been accepted. But then it will only accept a word if
        // it contains a vowel. Unintended consequence: every word must start
        // with a vowel. So we can just perform this check on the first letter.
        let firstchar = w.chars().next().unwrap();
        if VOWELS.contains(firstchar) && !forbidden_regex.is_match(&w) {

            // Accepted, so make a note that we've seen this word now.
            if w.len() < word_set_short.len() {
                word_set_short[w.len()].set(h, true);
            } else {
                word_set.insert(w.clone());
            }

            // Put the word in the ordered list of words as well.
            word_list.push(w.clone());

            // Progress report.
            if word_list.len() % 100000 == 0 {
                println!("{} words generated", word_list.len());
            }
        }
    }

    println!("Capitalizing some words");

    for i in 0..word_list.len() {
        if random.randint(0, 100) == 0 {
            word_list[i][0..1].make_ascii_uppercase();
        }
    }

    println!("Generating text");

    let mut f = io::BufWriter::new(fs::File::create("giganovel.txt").unwrap());

    let mut book = Book::new(word_list);

    // Random number generation has been moved to next_word() so it can directly use the generated id
    // to maintain a count of how often each word was produced instead of needing a separate hashmap.
    while book.len() < BOOK_SIZE {
        book.next_word(&mut random, &mut f)?;
    }
    book.end(&mut f)?;

    Ok(())
}
