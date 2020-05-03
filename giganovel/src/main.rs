// This is a rust port of "giganovel.py" (original description is included below.)
// I optimized the data structures a bit but the main logic is just about identical.
// The original takes about half and hour vs a little over 20 seconds for this version.
// (The generated file is of course byte-identical, that's the point of the program.)
//
// Original comments from giganovel.py:
// ---
// A script to generate text files that look like a novel in TXT form.
// Words are completely made up, but vaguely resemble the Finnish language.
// The resulting text uses ASCII encoding with only printable characters.
// Distribution of words follows Zipf's law.
//
// Standard parameters generate 1 GB text with 148391 distinct words.
//
// Used to benchmark solutions of the Bentley's k most frequent words problem:
//     https://codegolf.stackexchange.com/q/188133/
// ---

use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::{PythonRandom, MersenneTwister};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use std::{fmt::Debug, io, collections::HashSet, ops::{DerefMut, Deref}};
use bstr::{BString, ByteSlice, BStr};
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


// Helper for converting b'a'..=b'z' => 0..=25
fn byte_to_index(b: u8) -> usize { b as usize - b'a' as usize }

// Helper for converting b'a'..=b'z' => 2**0..=2**25
fn byte_to_bit(b: u8) -> u32 { 1 << byte_to_index(b) }


// Used to store the generated words.
// Storing them inline instead of as boxed strings saves a lot of memory traffic.
// Since the longest word generated is 16 bytes, we need to get a little creative
// to store it efficiently while maintaining both good alignment and a fixed size.
// Basically we store them padded with '\0' bytes, which cannot occur in generated
// words, and determine the length based on that when we need it. Reminds me of
// strncpy in C :)
#[derive(Default, Copy, Clone)]
#[repr(align(16))]
struct ShortWord {
    data: [u8; 16],
}

impl ShortWord {
    fn new(src: &[u8]) -> Self {
        let mut data = [0u8; 16];
        data[..src.len()].copy_from_slice(src);
        Self { data }
    }

    fn len(&self) -> usize {
        // wacky micro optimization. this does the same as the following line.
        // self.data.iter().position(|&b| b == 0).unwrap_or(16)
        16 - (u128::from_le_bytes(self.data).leading_zeros() / 8) as usize
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

// Packed map from a key b'a'..b'z' to a generic value.
// The main benefit of this over [Option<V>; 26] is that it puts more relevant
// data (i.e. not None values) in the first cache line of a structure.
// Since many nodes are quite sparse this means we spend less time waiting for
// memory for no reason.
#[derive(Default, Debug, Clone)]
struct LetterMap<V: Default> {
    bits: u32,
    values: ArrayVec::<[V; 26]>,
}

impl<V: Default> LetterMap<V> {
    fn rank(&self, letter: u8) -> usize {
        let mask = byte_to_bit(letter) - 1;
        (self.bits & mask).count_ones() as usize
    }

    fn insert(&mut self, letter: u8, value: V) -> &mut V {
        let idx = self.rank(letter);
        debug_assert!(self.bits & byte_to_bit(letter) == 0);
        self.bits |= byte_to_bit(letter);
        self.values.insert(idx, value);
        &mut self.values[idx]
    }

    fn get_or_insert_with<F>(&mut self, letter: u8, f: F) -> &mut V
        where F: FnOnce() -> V
    {
        if self.bits & byte_to_bit(letter) != 0 {
            let idx = self.rank(letter);
            &mut self.values[idx]
        } else {
            self.insert(letter, f())
        }
    }

    fn iter_values(&self) -> impl Iterator<Item=&V> {
        self.values.iter()
    }

    fn get(&self, letter: u8) -> Option<&V> {
        if self.bits & byte_to_bit(letter) != 0 {
            Some(&self.values[self.rank(letter)])
        } else {
            None
        }
    }
}

// Used to keep track of the set of words we have seen so far.
// Words are built letter by letter, and after each letter it is checked if we have
// found a new word yet or if this one has been seen before. It should be clear that
// it is very inefficient to do a hash lookup for every added letter. By keeping the
// set of words in a tree we can walk down this tree as we add letters.
// We keep a separate "accepted" bitfield instead of just using presence in `next`,
// so we can create the next tree level lazily. After all, just because we created
// this word doesn't mean we will create a word that has this word as a prefix.
// Keeping a `rejected` bitmask as well is probably kind of pointless as the
// rejection check does not show up on benchmarks, but we might as well cache it.
#[derive(Default)]
struct WordTreeNode {
    accepted: u32,
    rejected: u32,
    next: LetterMap<Box<Self>>,
}

impl WordTreeNode {
    fn get_mut(&mut self, letter: u8) -> &mut Self {
        self.next.get_or_insert_with(letter, Default::default).as_mut()
    }

    fn is_rejected(&self, letter: u8) -> bool {
        self.rejected & byte_to_bit(letter) != 0
    }

    fn is_accepted(&self, letter: u8) -> bool {
        self.accepted & byte_to_bit(letter) != 0
    }

    fn set_rejected(&mut self, letter: u8) {
        self.rejected |= byte_to_bit(letter);
    }

    fn set_accepted(&mut self, letter: u8) {
        self.accepted |= byte_to_bit(letter);
    }
}

// One bad thing about this is that the most expensive processing we do on this
// structure (the loop in next_letter) has to chase a pointer every time just to
// read the `total` field. But it's not really possible to optimize further without
// changing the logic beyond recognition.
#[derive(Default, Debug)]
struct MarkovNode {
    total: u32,
    letter: u8,
    letters: LetterMap<Box<Self>>,
}

impl MarkovNode {
    fn get(&self, letter: u8) -> Option<&Self> {
        self.letters.get(letter).map(|v| &**v)
    }

    fn get_mut(&mut self, letter: u8) -> &mut Self {
        self.letters.get_or_insert_with(letter, 
            || Box::new(Self { letter, ..Default::default() }))
    }

    fn iter_present(&self) -> impl Iterator<Item=&Self> {
        self.letters.iter_values().map(|v| & **v)
    }
}

fn train_markov(root: &mut MarkovNode, ngram: &[u8]) {
    let mut m = root;
    m.total += 1;
    for &b in ngram {
        m = m.get_mut(b);
        m.total += 1;
    }
}

fn next_letter(root: &MarkovNode, ngram: &[u8], random: &mut PythonRandom) -> u8 {

    // Bug in original implementation:
    // This assumes the letter must be present in the root node.
    // This is not guaranteed to be the case, but with the chosen
    // random seed the unwrap() happens to never fail.
    let mut m = root;
    for &b in ngram {
        m = m.get(b).or_else(|| root.get(b)).unwrap();
    }

    debug_assert!(m.total > 0);
    let mut num = random.randint(0, m.total as u64 - 1) as u32;

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

#[derive(Eq, PartialEq, Debug)]
enum WordFlags {
    Normal,
    Comma,
    Period,
    Paragraph,
}

#[derive(Debug)]
enum Pregenerated {
    WordIndex(usize),
    WordFlags(WordFlags),
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
    pregenerated: Vec<Pregenerated>,
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
            pregenerated: Vec::new(),
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

    fn generate_next_word_index(&mut self, random: &mut PythonRandom) -> usize {
        if let Some(pregenerated) = self.pregenerated.pop() {
            if let Pregenerated::WordIndex(i) = pregenerated {
                return i;
            } else {
                panic!();
            }
        }
        loop {
            let i = random.expovariate(LAMBDA) as usize;
            if i < self.words.len() {
                return i;
            }
        }
    }

    fn generate_next_flags(&mut self, random: &mut PythonRandom) -> WordFlags {
        if let Some(pregenerated) = self.pregenerated.pop() {
            if let Pregenerated::WordFlags(w) = pregenerated {
                return w;
            } else {
                panic!();
            }
        }
        if random.randint(0, 9) == 0 {
            WordFlags::Comma
        } else if random.randint(0, 9) == 0 {
            if random.randint(0, 9) == 0 {
                WordFlags::Paragraph
            } else {
                WordFlags::Period
            }
        } else {
            WordFlags::Normal
        }
    }

    fn write_line<W: Write>(&mut self, write: &mut W) -> io::Result<()> {
        self.line.push(b'\n');
        self.length += self.line.len();
        write.write_all(&self.line)?;
        self.line.clear();
        Ok(())
    }

    fn next_word<W: Write>(&mut self, random: &mut PythonRandom, write: &mut W) -> io::Result<()> {
        // Pick a random word. This was moved to this type so we can use the
        // generated index to count how often each word is used without needing
        // a separate hash lookup to get the count for a word.
        let word_index = self.generate_next_word_index(random);

        self.counter[word_index] += 1;

        let mut word = ArrayVec::<[u8; 17]>::new();
        word.try_extend_from_slice(&self.words[word_index]).unwrap();

        if self.capitalize {
            word[0].make_ascii_uppercase();
        }

        // Only capitalize one word at a time, except if we're still in
        // the front matter block, in that case following words should
        // still be capitalized as well.
        // In the original code capitalization is handled separately for
        // front matter and for normal text.
        if self.front {
            self.capitalize = false;
        }

        if !self.front {
            if self.title.fields().count() < 2 {
                self.title.extend_from_slice(&word);
                self.title.push(b' ');
            }
            else if self.author.fields().count() < 2 {
                self.author.extend_from_slice(&word);
                self.author.push(b' ');
            }
            else {
                self.verlag.extend_from_slice(&word);
                self.print_front(write)?;
            }
            return Ok(());
        }

        let wordflags = self.generate_next_flags(random);

        let tmp = self.generate_next_word_index(random);
        let tmp1 = Pregenerated::WordIndex(tmp);
        let tmp2 = Pregenerated::WordFlags(self.generate_next_flags(random));
        self.pregenerated.push(tmp2);
        self.pregenerated.push(tmp1);
        unsafe {
            core::arch::x86_64::_mm_prefetch(&self.words[tmp] as *const _ as *const i8, core::arch::x86_64::_MM_HINT_T2);
        }

        match wordflags {
            WordFlags::Normal => {}
            WordFlags::Comma => {
                word.push(b',');
            }
            WordFlags::Period => {
                word.push(b'.');
            }
            WordFlags::Paragraph => {
                word.push(b'.');
                self.capitalize = true;
            }
        }

        let paragraph = wordflags == WordFlags::Paragraph;

        if self.line.len() + 1 + word.len() > LINE_WIDTH {
            self.write_line(write)?;
            self.line.extend_from_slice(&word);
        }
        else if paragraph {
            self.line.push(b' ');
            self.line.extend_from_slice(&word);
            self.line.push(b'\n');
            self.write_line(write)?;
            self.line.extend_from_slice(TAB);
        }
        else {
            self.line.push(b' ');
            self.line.extend_from_slice(&word);
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

        write.flush()?;

        Ok(())
    }

}

fn main() -> Result<(), Box<dyn std::error::Error>>  {

    let data = fs::read("11940-8.txt")
        .expect("input not found, please wget http://www.gutenberg.org/files/11940/11940-8.txt");

    let text = decode_latin1(&data)
        .to_lowercase()
        .replace("ä", "a")
        .replace("å", "a")
        .replace("ö", "o");

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
    let all_words: HashSet<&[u8]> = slice
        .as_bytes()
        .fields_with(|b| !b.is_ascii_lowercase())
        .collect();

    println!("{} reference words", all_words.len());

    println!("Training Markov chain");

    let mut markov = MarkovNode::default();

    for word in all_words {
        for trigram in word.windows(3) {
            train_markov(&mut markov, trigram);
        }
    }

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
        let mut letter;

        loop {
            let ngram = &word[word.len().saturating_sub(2)..];

            letter = next_letter(&markov, &ngram, &mut random);

            word.push(letter);

            if !word_tree_node.is_accepted(letter) {
                break;
            }

            word_tree_node = word_tree_node.get_mut(letter);
        }

        if word_tree_node.is_rejected(letter) {
            continue;
        }

        // Bug in original implementation:
        // A word must contain a vowel. Okay. But the way words are constructed, every prefix
        // of a word needs to be a valid word itself. End result: only words that start with
        // a vowel can be generated by this program. So we only check the first letter.
        // Note that we still have to *generate* the whole word, just to keep the state of the
        // random number generator in sync with the original implementation.
        if !VOWELS.contains(&word[0]) || FORBIDDEN_MATCHER.is_match(&word) {
            word_tree_node.set_rejected(letter);
            continue;
        }

        word_tree_node.set_accepted(letter);
        word_list.push(ShortWord::new(&word));

        // Progress report.
        if word_list.len() % 100000 == 0 {
            println!("{} words generated", word_list.len());
        }

    }

    // Leak memory because it's faster, we will end the entire process soon enough.
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
    // Progress reporting was added to this phase; the original script does not have it.
    let mut last_mb = 0usize;
    while book.len() < BOOK_SIZE {
        book.next_word(&mut random, &mut writer)?;
        if book.len() > (last_mb + 20) * 1024 * 1024 {
            last_mb += 20;
            println!("{}MiB of text generated", last_mb);
        }
    }
    book.end(&mut writer)?;

    Ok(())
}
