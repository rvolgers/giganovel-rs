use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::{PythonRandom, MersenneTwister};
use aho_corasick::AhoCorasickBuilder;
use std::{fmt::Debug, io};
use bitvec::prelude::{bitbox, Lsb0};
use fnv::FnvHashSet;
use bstr::{BString, ByteSlice};

// Seed for the random number generator, for consistent output.
// Comment from original code follows:
// from OEIS A001519, the title should be "Itera Aeno", md5sum = 4dcf116dc35156ec939f8cafd61bdf18
const RANDOM_SEED: u32 = 63245986;

const BOOK_SIZE: usize = 1<<30;               // 1 GB
const DISTINCT_WORDS: usize = 5000000;        // bigger number will allow longer longest word
const MEAN: usize = 15000;                    // bigger number will increase average word length
const LAMBDA: f64 = 1.0 / (MEAN as f64);

const VOWELS: &[u8] = b"aeiouy";
const FORBIDDEN: &[&str] = &["satan", "lenin", "stalin", "hitl", "naz", "rus", "putin"];

const TAB: &[u8] = b"   ";
const LINE_WIDTH: usize = 76;


// TODO use a ready made huffman crate?
// TODO lots of allocation happening here, not tried to optimize yet
fn huffman<T: Debug>(weights_and_symbols: impl Iterator<Item=(usize, T)>) -> Vec::<(usize, u8, T)> {

    enum HuffmanNodeContents<T> {
        Node([Box<HuffmanNode<T>>; 2]),
        Leaf(T),
    }

    struct HuffmanNode<T> {
        weight: usize,
        contents: HuffmanNodeContents<T>,
    }

    use HuffmanNodeContents::*;

    let mut tmp = weights_and_symbols
        .map(|(weight, s)| Box::new(HuffmanNode { weight, contents: Leaf(s) }))
        .collect::<Vec<_>>();

    let key_func = |n: &Box<HuffmanNode<T>>| std::usize::MAX - n.weight;

    tmp.sort_by_key(key_func);

    while tmp.len() > 1 {
        let a = tmp.pop().unwrap();
        let b = tmp.pop().unwrap();
        let new_node = Box::new(HuffmanNode {
            weight: a.weight + b.weight,
            contents: Node([a, b]),
        });
        tmp.insert(tmp.binary_search_by_key(&key_func(&new_node), key_func).unwrap_or_else(|x| x), new_node);
    }

    let mut codes = Vec::<(usize, u8, T)>::new();
    let mut stack = vec![(0usize, 0u8, tmp.pop().into_iter().collect::<Vec<_>>())];

    // TODO just use recursion for this?
    while let Some((mut code, mut bits, mut nodes)) = stack.pop() {
        while let Some(node) = nodes.pop() {
            match node.contents {
                Leaf(t) => {
                    codes.push((code, bits, t));
                    code |= 1;
                },
                Node([a, b]) => {  // Uuuugh https://github.com/rust-lang/rust/issues/25725
                    // subtle: after this scope is done, resume with code |= 1
                    stack.push((code | 1, bits, nodes));
                    code <<= 1;
                    bits += 1;
                    nodes = vec![b, a];
                },
            }
        }
    }

    codes
}

#[derive(Default, Debug)]
struct MarkovNode {
    total: u64,
    huffman: (usize, u8),
    letters: [Option<Box<MarkovNode>>; 26],
    //eof_huffman: (usize, u8),
}

impl MarkovNode {
    pub fn huffman_letters(&mut self) {
        // let mut eof = [Box::new(MarkovNode::default())];
        // eof[0].total = self.letters.iter().flatten().map(|n| n.total).max().unwrap_or(1);
        // eof[0].eof_huffman = (std::usize::MAX, 0);  // Marker
        let node_iter = self.letters.iter_mut()
            .flatten()
            //.chain(eof.iter_mut())
            .map(|n| (n.total as usize, n));
        for (code, bits, n) in huffman(node_iter) {
            n.huffman = (code, bits);

            // Prevent infinite recursion
            //if n.eof_huffman != (std::usize::MAX, 0) {
                n.huffman_letters();
            //}
        }
        //self.eof_huffman = eof[0].huffman;
    }
}

// Helpers for indexing into MarkovNode.letters
fn index_to_byte(i: usize) -> u8 { b'a' + i as u8 }
fn byte_to_index(b: u8) -> usize { b as usize - b'a' as usize }

struct Markov {
    root: MarkovNode,
}

impl Markov {

    fn new() -> Markov {
        Markov {
            root: Default::default(),
        }
    }

    fn huffman_letters(&mut self) {
        self.root.huffman_letters();
    }

    fn train(&mut self, word: &[u8]) {
        let mut m = &mut self.root;
        m.total += 1;
        for b in word.bytes() {
            m = m.letters[byte_to_index(b)].get_or_insert_with(Default::default);
            m.total += 1;
        }
    }

    fn lookup(&self, word_tail: impl Iterator<Item=u8>) -> Option<&MarkovNode> {
        let mut m = &self.root;

        for b in word_tail {
            m = m.letters[byte_to_index(b)].as_ref()
                .or_else(|| self.root.letters[byte_to_index(b)].as_ref())?;
        }

        Some(m)
    }

    fn next_letter(&self, word_tail: &[u8], random: &mut PythonRandom, huffman: &mut (usize, u32)) -> u8 {
        // Bug in original implementation:
        // This assumes the letter must be present in the root node.
        // This is not guaranteed to be the case, but with the chosen
        // random seed the unwrap() happens to never fail.
        let m = self.lookup(word_tail.bytes()).unwrap();

        assert!(m.total > 0);
        let mut num = random.randint(0, m.total - 1);

        for index in 0..m.letters.len() {
            if let Some(ref x) = &m.letters[index] {
                if x.total > num {

                    huffman.0 = (huffman.0.wrapping_shl(x.huffman.1 as u32)) | x.huffman.0;
                    huffman.1 = huffman.1.checked_add(x.huffman.1 as u32).unwrap();

                    assert!(huffman.1 >= 63 || huffman.0 < (1usize << huffman.1), "{:?} {:?}", huffman, x.huffman);

                    return index_to_byte(index);
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

    // fn next_eof(&self, word_tail: &[u8], huffman: &(usize, u32), eof_huffman: &mut (usize, u32)) {

    //     let x = self.lookup(word_tail.bytes()).unwrap();

    //     eof_huffman.0 = (huffman.0.wrapping_shl(x.eof_huffman.1 as u32)) | x.eof_huffman.0;
    //     eof_huffman.1 = huffman.1.checked_add(x.eof_huffman.1 as u32).unwrap();

    //     assert!(huffman.1 >= 63 || huffman.0 < (1usize << huffman.1), "{:?} {:?}", huffman, x.eof_huffman);
    // }
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
    words: Vec<BString>,
    length: usize,
}

impl Book {
    fn new(words: Vec<BString>) -> Book {
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

    fn next_word<W: Write>(&mut self, random: &mut PythonRandom, write: &mut W) -> io::Result<()> {
        // Pick a random word. This was moved to next_word so we can use the
        // generated index to count how often each word is used without needing
        // a separate hash lookup to get the count for a word.
        let word;
        loop {
            let i = random.expovariate(LAMBDA) as usize;
            if i < self.words.len() {
                word = self.words[i].as_bytes();
                self.counter[i] += 1;
                break;
            }
        }

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
    let all_words: FnvHashSet<&[u8]> = slice.as_bytes()
        .fields_with(|b| !b.is_ascii_lowercase())
        .collect();

    println!("{} reference words", all_words.len());

    println!("Training Markov chain");

    let mut markov = Markov::new();

    for word in all_words {
        let mut word = &word[..];
        while word.len() >= 3 {
            markov.train(&word[..3]);
            word = &word[1..];
        }
    }

    markov.huffman_letters();

    println!("Generating artificial words");

    // Initialize python-compatible RNG
    let mt = MersenneTwister::new();
    let mut random = PythonRandom::new(mt);
    random.seed_u32(RANDOM_SEED);

    let mut word_set = FnvHashSet::<BString>::with_capacity_and_hasher(0, Default::default());

    const SET_BITS: usize = 33;

    let mut word_bitvec = bitbox![Lsb0, u64; 0; 1usize << SET_BITS];

    println!("Using {} megabytes of RAM for word_bitvec", word_bitvec.len() / 8 / 1024 / 1024);

    // The actual ordered list of words we will use later to pick words based on random numbers.
    // We could use a set that preserves insertion order, but the hash lookups we save by using
    // the word_set_short optimization outweighs the benefit of not duplicating storage.
    let mut word_list = Vec::<BString>::with_capacity(DISTINCT_WORDS);

    // Could also use `regex`, but for this simple case we can use aho-corasick directly
    // and save a dependency (regex depends on this library).
    let forbidden = AhoCorasickBuilder::new()
        .auto_configure(FORBIDDEN)
        .build(FORBIDDEN);

    // String that holds the current word.
    let mut w = BString::from("");

    // TODO instead of this, it might be better to put an EOF token in each huffman tree,
    //      and accumulate the bits in the other direction. Should improve locality.
    // Cheating: hardcoded counts for each word length
    let lengths: [usize; 20] = [
        0,
        6,
        113,
        1080,
        11228,
        99400,
        496678,
        1211867,
        1486115,
        1034258,
        464479,
        149554,
        36546,
        7273,
        1200,
        183,
        20,
        0,
        0,
        0,
    ];

    let length_huffman = {
        let mut tmp: [Option<(usize, u8)>; 20] = Default::default();

        for (code, bits, length) in huffman(lengths.iter().enumerate().filter(|(_, &c)| c > 0).map(|(i, c)| (*c, i))) {
            tmp[length] = Some((code, bits));
        }

        tmp
    };

    let mut huffgood = 0usize;
    let mut huffbad = 0usize;
    let mut huffmaxbits = 0u32;

    while word_list.len() < DISTINCT_WORDS {

        w.clear();

        let mut huffman = (0usize, 0u32);
        //let mut eof_huffman = (0usize, 0u32);
        let mut huffindex: Option<usize>;

        // Add more letters until we find a word that hasn't been accepted yet.
        // (Could still be a word that will never be accepted due to later checks.)
        loop {
            // Add a new letter to the word and update the word_set_short index.
            let word_tail = &w[w.len().saturating_sub(2)..];

            let letter = markov.next_letter(word_tail, &mut random, &mut huffman);
            w.push(letter);

            huffmaxbits = huffman.1.max(huffmaxbits);

            //markov.next_eof(&w[w.len().saturating_sub(2)..], &huffman, &mut eof_huffman);

            huffindex = None;
            // TODO debug the eof-in-huffman code.
            // The reason for wanting a dedicated eof symbol at the end is that it might improve
            // locality near the end of the hashing maybe?
            // Having eof as a huffman symbol seems to make sense, but the problem is its
            // probability distribution is sort of indepenent of the markov tables. Or at
            // least not as well correlated, and it's hard to get a good estimate.
            // NOTE You need to make sure the start of your bitstream is always aligned the same
            //      (i.e. either hi-align it or flip the bit order).
            if w.len() < length_huffman.len() {
                if let Some(len_huff) = length_huffman[w.len()] {
                    if len_huff.1 as u32 + huffman.1 < 64 {
                        let total_bits = len_huff.1 as u32 + huffman.1;
                        if 1usize << total_bits <= word_bitvec.len() {
                            let mut tmp = huffman.0;
                            tmp = tmp | (len_huff.0).checked_shl(huffman.1).unwrap();
                            tmp = tmp.reverse_bits() >> (64 - total_bits);
                            //tmp = tmp << (SET_BITS as u32 - total_bits);
                            huffindex = Some(tmp);
                        }
                    }
                }
            }
            // if eof_huffman.1 < 63 && (1 << eof_huffman.1) < word_bitvec.len() {
            //     huffindex = Some(eof_huffman.0);
            // }

            if huffindex.is_some() {
                huffgood += 1;
            } else {
                huffbad += 1;
            }

            if let Some(huffindex) = huffindex {
                if !word_bitvec[huffindex] {
                    //assert!(!word_set.contains(&w), "oops {} {:?} {:?} {}", &w, &huffman, length_huffman[w.len()], huffindex);
                    break;
                }
                //assert!(word_set.contains(&w), "oops {} {:?} {:?} {}", &w, &huffman, length_huffman[w.len()], huffindex);
            }
            else {
                if !word_set.contains(&w) { break; }
            }
        }

        // Check for vowels and forbidden words.
        // Bug in original implementation:
        // The word generation loop ends as soon as it produces a word that has
        // not previously been accepted. But then it will only accept a word if
        // it contains a vowel. Unintended consequence: every word must start
        // with a vowel. So we can just perform this check on the first letter.
        if VOWELS.contains(&w.as_bytes()[0]) && !forbidden.is_match(&w) {
            // Accepted, so make a note that we've seen this word now.
            if let Some(huffindex) = huffindex {
                word_bitvec.set(huffindex, true);
                //word_set.insert(w.clone());
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

    println!("{} out of {} lookups used the word_bitvec, {} didn't, longest huffman encoded word (sans length) was {} bits", huffgood, huffgood + huffbad, huffbad, huffmaxbits);

    // let mut lengths = [0usize; 20];
    // for w in &word_list {
    //     lengths[w.len()] += 1;
    // }
    // dbg!(lengths);

    println!("Capitalizing some words");

    for i in 0..word_list.len() {
        if random.randint(0, 100) == 0 {
            word_list[i][0..1].make_ascii_uppercase();
        }
    }

    println!("Generating text");

    let mut f = io::BufWriter::with_capacity(1024 * 1024 * 4, fs::File::create("giganovel.txt").unwrap());

    let mut book = Book::new(word_list);

    // Random number generation has been moved to next_word() so it can directly use the generated id
    // to maintain a count of how often each word was produced instead of needing a separate hashmap.
    while book.len() < BOOK_SIZE {
        book.next_word(&mut random, &mut f)?;
    }
    book.end(&mut f)?;

    Ok(())
}
