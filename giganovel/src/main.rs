


use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::{PythonRandom, MersenneTwister};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use std::{fmt::Debug, io, collections::HashSet, ops::{DerefMut, Deref}};
use bstr::{BString, BStr, ByteSlice};
use arrayvec::ArrayVec;

use lazy_static::lazy_static;

// Seed for the random number generator, for consistent output.
// Comment from original code follows:
// from OEIS A001519, the title should be "Itera Aeno", md5sum = 4dcf116dc35156ec939f8cafd61bdf18
const RANDOM_SEED: u32 = 63245986;

const BOOK_SIZE: usize = 1<<30;               // 1 GB
const DISTINCT_WORDS: usize = 5000000;        // bigger number will allow longer longest word
const MEAN: usize = 15000;                    // bigger number will increase average word length
const LAMBDA: f64 = 1.0 / (MEAN as f64);

const VOWELS: &[u8; 6] = b"aeiouy";
const FORBIDDEN: &[&str] = &["satan", "lenin", "stalin", "hitl", "naz", "rus", "putin"];

const TAB: &[u8] = b"   ";
const LINE_WIDTH: usize = 76;

lazy_static! {
    static ref FORBIDDEN_MATCHER: AhoCorasick = AhoCorasickBuilder::new()
        .auto_configure(FORBIDDEN)
        .build(FORBIDDEN);
}

// Used to store the generated words.
// Storing them inline instead of as boxed strings saves a lot of memory traffic.
// Since the longest word generated is 16 bytes, we need to get a little creative
// to store it efficiently while maintaining both good alignment and a fixed size.
// Basically we store them padded with '\0' bytes, which cannot occur in generated
// words, and determine the length based on that when we need it.
#[derive(Default, Copy, Clone, Debug)]
struct ShortWord {
    data: [u8; 16],
}

impl ShortWord {
    fn len(&self) -> usize {
        // wacky micro optimization. this does the same as the following line.
        // self.data.iter().position(|&b| b == 0).unwrap_or(16)
        16 - (u128::from_le_bytes(self.data).leading_zeros() / 8) as usize
    }
}

impl From<&[u8]> for ShortWord {
    fn from(src: &[u8]) -> Self {
        let mut data = [0u8; 16];
        data[..src.len()].copy_from_slice(src);
        Self { data }
    }
}

impl Deref for ShortWord {
    type Target=[u8];
    fn deref(&self) -> &Self::Target {
        &self.data[..ShortWord::len(self)]
    }
}

impl DerefMut for ShortWord {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = ShortWord::len(self);
        &mut self.data[..len]
    }
}

// Used to store the set of words we have seen.
// accepted and rejected are bitmasks.
// if a word has not been accepted or rejected, it is tested against the rules for words and
// the bitmasks are updated accordingly.
// once a word is rejected, it is is rejected forever.
// once a word is accepted, the next time Exists will be returned for that word and the next level in
// the tree for that letter will be initialized and returned, since we will keep building our word
// and need somewhere to put the next letter.

#[derive(Default)]
struct WordTreeNode {
    accepted: u32,
    rejected: u32,
    next: [Option<Box<Self>>; 26],
}

enum LookupResult<'a> {
    Accepted,
    Rejected,
    Exists(&'a mut WordTreeNode),
}

impl WordTreeNode {
    fn lookup(&mut self, word: &[u8]) -> LookupResult {
        let index = byte_to_index(word[word.len() - 1]);
        let bit = 1 << index;
        if self.accepted & bit != 0 {
            LookupResult::Exists(self.next[index as usize].get_or_insert_with(Default::default).as_mut())
        } else if self.rejected & bit != 0 {
            LookupResult::Rejected
        } else if VOWELS.contains(&word[0]) && !FORBIDDEN_MATCHER.is_match(&word) {
            self.accepted |= bit;
            LookupResult::Accepted
        } else {
            self.rejected |= bit;
            LookupResult::Rejected
        }
    }
}

#[derive(Default, Debug)]
struct TrainMarkovNode {
    total: u64,
    letter: u8,
    letters: [Option<Box<Self>>; 26],
}

impl TrainMarkovNode {

    fn train(&mut self, word: &[u8]) {
        let mut m = self;
        m.total += 1;
        for b in word.bytes() {
            m = m.letters[byte_to_index(b)].get_or_insert_with(Default::default);
            m.letter = b;
            m.total += 1;
        }
    }

}

#[derive(Default, Debug)]
struct LookupMarkovNode {
    total: u64,
    letter: u8,
    present: u32,  // bitmask specifying which letters are present
    letters: ArrayVec::<[Box<Self>; 26]>,  // packed array containing only present letters
}

impl LookupMarkovNode {
    fn bit(letter: u8) -> u32 {
        1u32 << byte_to_index(letter)
    }

    fn rank(&self, letter: u8) -> usize {
        let mask = Self::bit(letter) - 1;
        (self.present & mask).count_ones() as usize
    }

    fn from_training(training: &TrainMarkovNode) -> Self {
        Self {
            total: training.total,
            letter: training.letter,
            present: training.letters.iter()
                .map(|n| if let Some(n) = n { Self::bit(n.letter) } else { 0 })
                .sum(),
            letters: training.letters.iter()
                .flatten()
                .map(|n| Box::new(Self::from_training(n)))
                .collect(),
        }
    }

    fn iter_present(&self) -> impl Iterator<Item=&Self> {
        assert!(self.letters.len() == self.present.count_ones() as usize);
        self.letters.iter().map(|x| &**x)
    }

    fn lookup(&self, letter: u8) -> Option<&Self> {
        if self.present & Self::bit(letter) != 0 {
            Some(&*self.letters[self.rank(letter)])
        } else {
            None
        }
    }

}

// Helpers for indexing into MarkovNode.letters
fn index_to_byte(i: usize) -> u8 { b'a' + i as u8 }
fn byte_to_index(b: u8) -> usize { b as usize - b'a' as usize }

#[derive(Default)]
struct MarkovLookup {
    root: LookupMarkovNode,
}

impl MarkovLookup {

    fn new(training_root: TrainMarkovNode) -> Self {
        MarkovLookup {
            root: LookupMarkovNode::from_training(&training_root),
        }
    }

    fn next_letter(&self, word_tail: &[u8], random: &mut PythonRandom) -> u8 {

        // Bug in original implementation:
        // This assumes the letter must be present in the root node.
        // This is not guaranteed to be the case, but with the chosen
        // random seed the unwrap() happens to never fail.
        let mut m = &self.root;
        for &b in word_tail {
            m = m.lookup(b).or_else(|| self.root.lookup(b)).unwrap();
        }

        assert!(m.total > 0);
        let mut num = random.randint(0, m.total - 1);

        for x in m.iter_present() {
            if x.total > num {
                return x.letter
            }
            num -= x.total;
        }

        // Bug in original implementation:
        // Totals of children do not sum to the total of the parent, but the
        // code above assumes it does. This is definitely a bug, but it's just
        // never hit with the hardcoded random number generator seed...
        panic!("inconsistent tree");
    }
}


struct Book {
    title: BString,
    author: BString,
    year: BString,
    verlag: BString,
    line: BString,
    front: bool,
    capitalize: bool,
    counter: Vec<usize>,
    words: Vec<ShortWord>,
    length: usize,
}

impl Book {
    fn new<'a>(words: Vec<ShortWord>) -> Book {
        Book {
            title: Default::default(),
            author: Default::default(),
            year: "2019".into(),
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
        writeln!(write, "{}\n", self.title.to_uppercase().trim_end().as_bstr())?;

        writeln!(write, "{}\n", self.author.trim_end().as_bstr())?;

        writeln!(write, "(c) {}, {}, Public domain\n", self.year, self.verlag)?;

        self.line.clear();
        self.line.extend(TAB.bytes());
        self.front = true;

        Ok(())
    }

    fn next_word_index(&self, random: &mut PythonRandom) -> usize {
        loop {
            let i = random.expovariate(LAMBDA) as usize;
            if i < self.words.len() { return i }
        }
    }

    fn next_word<W: Write>(&mut self, random: &mut PythonRandom, write: &mut W) -> io::Result<()> {
        // Pick a random word. This was moved to this type so we can use the
        // generated index to count how often each word is used without needing
        // a separate hash lookup to get the count for a word.
        let i = self.next_word_index(random);
        self.counter[i] += 1;

        let word = self.words[i];

        let mut punctuation: Option<u8> = None;

        // Only capitalize one word at a time, except if we're still in
        // the front matter block, in that case following words should
        // still be capitalized as well.
        // In the original code capitalization is handled separately for
        // front matter and for normal text.
        let capitalize = self.capitalize;
        if self.front {
            self.capitalize = false;
        }

        let push_word = |line: &mut BString| {
            if capitalize {
                line.push(word[0].to_ascii_uppercase());
                line.extend(&word[1..]);
            }
            else {
                line.extend(word.bytes());
            }
        };

        if !self.front {
            if self.title.fields().count() < 2 {
                push_word(&mut self.title);
                self.title.push(b' ');
            }
            else if self.author.fields().count() < 2 {
                push_word(&mut self.author);
                self.author.push(b' ');
            }
            else {
                push_word(&mut self.verlag);
                self.print_front(write)?;
            }
            return Ok(());
        }

        let mut paragraph = false;

        if random.randint(0, 9) == 0 {
            punctuation = Some(b',');
        }
        else if random.randint(0, 9) == 0 {
            punctuation = Some(b'.');
            self.capitalize = true;
            if random.randint(0, 9) == 0 {
                paragraph = true;
            }
        }

        if self.line.len() + 1 + word.len() + punctuation.iter().count() > LINE_WIDTH {
            self.line.push(b'\n');
            self.length += self.line.len();
            write.write_all(self.line.as_bytes())?;
            self.line.clear();
            push_word(&mut self.line);
            if let Some(b) = punctuation {
                self.line.push(b);
            }
        }
        else if paragraph {
            self.line.push(b' ');
            push_word(&mut self.line);
            if let Some(b) = punctuation {
                self.line.push(b);
            }
            self.line.push(b'\n');
            self.line.push(b'\n');
            self.length += self.line.len();
            write.write_all(self.line.as_bytes())?;
            self.line.clear();
            self.line.extend(TAB.bytes());
        }
        else {
            self.line.push(b' ');
            push_word(&mut self.line);
            if let Some(b) = punctuation {
                self.line.push(b);
            }
        }

        Ok(())
    }

    fn end<W: Write>(&mut self, write: &mut W) -> io::Result<()> {
        writeln!(write, "{}.", &self.line.trim_end().as_bstr())?;

        let mut tmp = self.counter.iter()
            .zip(&self.words)
            .collect::<Vec<_>>();

        tmp.sort_by_key(|&(c, _)| c);

        writeln!(write, "\n--\n\n\n\n\n\n\nMost common words:")?;

        for (_, word) in tmp.iter().rev().take(10) {
            write.write_all(b"- ")?;
            write.write_all(word)?;
            write.write_all(b"\n")?;
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
    let all_words: HashSet<&[u8]> = slice.as_bytes()
        .fields_with(|b| !b.is_ascii_lowercase())
        .collect();

    println!("{} reference words", all_words.len());

    println!("Training Markov chain");

    let mut markov = TrainMarkovNode::default();

    for word in all_words {
        let mut word = &word[..];
        while word.len() >= 3 {
            markov.train(&word[..3]);
            word = &word[1..];
        }
    }

    let markov = MarkovLookup::new(markov);

    println!("Generating artificial words");

    // Initialize python-compatible RNG
    let mt = MersenneTwister::new();
    let mut random = PythonRandom::new(mt);
    random.seed_u32(RANDOM_SEED);

    let mut word_tree_root = WordTreeNode::default();

    // String that holds the current word.
    let mut word = BString::from("");

    let mut word_list: Vec<ShortWord> = Vec::with_capacity(DISTINCT_WORDS);

    while word_list.len() != DISTINCT_WORDS {

        word.clear();
        let mut word_tree_node = &mut word_tree_root;

        loop {
            // Add a new letter to the word and update the word_set_short index.

            let word_tail = &word[word.len().saturating_sub(2)..];

            let letter = markov.next_letter(&word_tail, &mut random);

            word.push(letter);

            word_tree_node = match word_tree_node.lookup(&word) {
                LookupResult::Exists(x) => x,
                LookupResult::Rejected => break,
                LookupResult::Accepted => {
                    // let (head, tail) = wordbuf_write.split_at_mut(WORD_SIZE);
                    // wordbuf_write = tail;
                    // head[..w.len()].copy_from_slice(&w);

                    word_list.push(ShortWord::from(&word[..]));

                    // Progress report.
                    //if (DISTINCT_WORDS - (wordbuf_write.len() / WORD_SIZE)) % 100000 == 0 {
                    if word_list.len() % 100000 == 0 {
                        //println!("{} words generated", DISTINCT_WORDS - (wordbuf_write.len() / WORD_SIZE));
                        println!("{} words generated", word_list.len());
                    }

                    break
                }
            };

        }

    }

    // Leak memory because it's faster O:)
    std::mem::forget(word_tree_root);

    println!("Capitalizing some words");

    for i in 0..DISTINCT_WORDS {
        if random.randint(0, 100) == 0 {
            word_list[i][0].make_ascii_uppercase();
        }
    }

    println!("Generating text");

    let writer = fs::File::create("giganovel.txt").unwrap();
    let mut writer = io::BufWriter::with_capacity(1024 * 1024 * 4, writer);

    let mut book = Book::new(word_list);

    // Random number generation has been moved to next_word() so it can directly use the generated id
    // to maintain a count of how often each word was produced instead of needing a separate hashmap.
    while book.len() < BOOK_SIZE {
        book.next_word(&mut random, &mut writer)?;
    }
    book.end(&mut writer)?;

    Ok(())
}
