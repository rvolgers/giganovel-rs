use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::io::Write;
use std::fs;
use encoding_rs::mem::decode_latin1;
use rand_python::{PythonRandom, MersenneTwister};
use aho_corasick::AhoCorasickBuilder;
use std::{fmt::Debug, io, cmp::PartialOrd, convert::TryInto};
use bitvec::prelude::{bitbox, Lsb0};
use fnv::FnvHashSet;
use bstr::{BString, BStr, ByteSlice};

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



const WORD_SIZE: usize = 16;

/// Value type that behaves like a vector of bits with max length 64.
/// Used for representing (sequences of) huffman codes.
#[derive(Clone, Copy, Default, Debug)]
struct SmolBitvec {
    used: u8,
    bits: u64,
}

impl SmolBitvec {
    fn new() -> Self { Default::default() }

    fn len(&self) -> u8 {
        self.used
    }

    fn checked_extend(&self, other: &SmolBitvec) -> Option<Self> {
        if self.used + other.used > 64  {
            return None;
        }
        Some(Self {
            used: self.used + other.used,
            bits: (self.bits << other.used) | other.bits,
        })
    }

    fn checked_push(&self, bit: bool) -> Option<Self> {
        if self.used + 1 > 64 {
            return None;
        }
        Some(Self {
            used: self.used + 1,
            bits: (self.bits << 1) | (bit as u64),
        })
    }

    fn reverse(&self) -> Self {
        Self {
            used: self.used,
            bits: self.bits.reverse_bits() >> (64 - self.used),
        }
    }

    fn to_u64(&self) -> u64 {
        self.bits
    }
}

/// Generates huffman codes for a given set of symbols and their corresponding weights.
///
/// This does lots of allocations, but it's not called that often so probably not worth optimizing.
fn huffman<T: Debug>(weights_and_symbols: impl Iterator<Item=(usize, T)>) -> Vec::<(SmolBitvec, T)> {
    enum HuffmanNodeContents<T> {
        Node(Box<HuffmanNode<T>>, Box<HuffmanNode<T>>),
        Leaf(T),
    }

    struct HuffmanNode<T> {
        weight: usize,
        contents: HuffmanNodeContents<T>,
    }

    // Um, std::collections, all I wanted was a minheap by HuffmanNode.weight, ok?
    // This is a crazy amount of impls you're asking for.

    impl<T> PartialOrd for HuffmanNode<T> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl<T> Ord for HuffmanNode<T> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.weight.cmp(&other.weight).reverse()
        }
    }

    impl<T> PartialEq for HuffmanNode<T> {
        fn eq(&self, other: &Self) -> bool {
            self.weight == other.weight
        }
    }

    impl<T> Eq for HuffmanNode<T> {}

    // Initialize heap with every node a leaf node
    let mut heap = weights_and_symbols
        .map(|(weight, s)| Box::new(HuffmanNode { weight, contents: HuffmanNodeContents::Leaf(s) }))
        .collect::<BinaryHeap<_>>();

    // Put the two nodes with lowest weight under a new common parent until a single node remains.
    while heap.len() > 1 {
        let a = heap.pop().unwrap();
        let b = heap.pop().unwrap();
        heap.push( Box::new(HuffmanNode {
            weight: a.weight + b.weight,
            contents: HuffmanNodeContents::Node(a, b),
        }));
    }

    let mut codes = Vec::<(SmolBitvec, T)>::new();

    // Walk the tree, appending the symbol and the huffman code for each leaf node to `codes`.
    // Use recursion for simplicity here since we will never have many nodes.
    // (Note that heap.pop().is_none() can happen here, but only if the input was zero length.)
    if let Some(root) = heap.pop() {
        fn collect<T>(node: Box<HuffmanNode<T>>, code: SmolBitvec, dest: &mut Vec<(SmolBitvec, T)>) {
            match node.contents {
                HuffmanNodeContents::Node(a, b) => {
                    collect(a, code.checked_push(false).unwrap(), dest);
                    collect(b, code.checked_push(true).unwrap(), dest);
                }
                HuffmanNodeContents::Leaf(t) => {
                    dest.push((code, t));
                }
            }
        };

        collect(root, SmolBitvec::new(), &mut codes);
    }

    codes
}

#[derive(Default, Debug)]
struct MarkovNode {
    total: u64,
    huffman: SmolBitvec,
    letter: u8,
    present: u32,
    letters: [Option<Box<MarkovNode>>; 26],
    //eof_huffman: (usize, u8),
}

impl MarkovNode {
    pub fn huffman_letters(&mut self) {
        assert!(self.present == 0);

        let node_iter = self.letters.iter_mut()
            .flatten()
            .map(|n| (n.total as usize, n));
        for (code, n) in huffman(node_iter) {
            n.huffman = code;
            n.huffman_letters();
        }
    }

    pub fn pack(&mut self) {
        assert!(self.present == 0);

        let mut dest_idx = 0;

        for i in 0..26 {
            if dest_idx < i {
                if let Some(mut n) = self.letters[i].take() {
                    n.pack();
                    self.letters[dest_idx] = Some(n);
                    dest_idx += 1;
                    self.present |= 1 << i;
                }
            }
            else if let Some(n) = &mut self.letters[i] {
                dest_idx += 1;
                self.present |= 1 << i;
                n.pack();
            }
        }

        self.present |= 1<<31;
    }

    fn iter_present(&self) -> impl Iterator<Item=&MarkovNode> {
        let max = (self.present & ((1 << 26) - 1)).count_ones();
        self.letters[..max as usize].iter().flatten().map(|c| c.as_ref())
    }

    fn index_packed(&self, letter: u8) -> Option<&MarkovNode> {
        assert!(self.present != 0);

        let abc_index = byte_to_index(letter);
        let bit = 1 << abc_index;
        if self.present & bit == 0 { return None; }
        let offset = ((bit - 1) & self.present).count_ones();
        let m: Option<&MarkovNode> = self.letters[offset as usize].as_ref().map(|c| c.as_ref());
        m
    }

}

// Helpers for indexing into MarkovNode.letters
fn index_to_byte(i: usize) -> u8 { b'a' + i as u8 }
fn byte_to_index(b: u8) -> usize { b as usize - b'a' as usize }

struct Markov {
    root: MarkovNode,
    huffman_vowels: [SmolBitvec; 26],
}

impl Markov {

    fn new() -> Markov {
        Markov {
            root: Default::default(),
            huffman_vowels: Default::default(),
        }
    }

    fn huffman_letters(&mut self) {
        self.root.huffman_letters();

        // Special huffman table for first character of word, which has only vowels
        let node_iter = self.root.letters.iter_mut()
            .flatten()
            .filter(|m| VOWELS.contains(&m.letter))
            .map(|n| (n.total as usize, n));

        for (code, n) in huffman(node_iter) {
            self.huffman_vowels[byte_to_index(n.letter)] = code;
        }
    }

    fn pack(&mut self) {
        self.root.pack();
    }


    fn lookup(&self, word_tail: impl Iterator<Item=u8>) -> Option<&MarkovNode> {
        let mut m = &self.root;

        for b in word_tail {
            m = m.index_packed(b)
                .or_else(|| self.root.index_packed(b))?;
        }

        Some(m)
    }

    fn train(&mut self, word: &[u8]) {
        let mut m = &mut self.root;
        m.total += 1;
        for b in word.bytes() {
            m = m.letters[byte_to_index(b)].get_or_insert_with(Default::default);
            m.letter = b;
            m.total += 1;
        }
    }


    fn next_letter(&self, word_tail: &[u8], random: &mut PythonRandom, huffman: &mut Option<SmolBitvec>) -> u8 {
        // Bug in original implementation:
        // This assumes the letter must be present in the root node.
        // This is not guaranteed to be the case, but with the chosen
        // random seed the unwrap() happens to never fail.
        let m = self.lookup(word_tail.bytes()).unwrap();

        assert!(m.total > 0);
        let mut num = random.randint(0, m.total - 1);

        for x in m.iter_present() {
            if x.total > num {

                let next_huffman = if word_tail.len() == 0 {
                    &self.huffman_vowels[byte_to_index(x.letter)]
                } else {
                    &x.huffman
                };

                *huffman = huffman.and_then(|v| v.checked_extend(next_huffman));

                return x.letter;
            }
            num -= x.total;
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


struct Book<'a> {
    title: BString,
    author: BString,
    year: BString,
    verlag: BString,
    line: BString,
    front: bool,
    capitalize: bool,
    counter: Vec<usize>,
    words: &'a [u8],
    length: usize,
}

impl Book<'_> {
    fn new<'a>(words: &'a [u8]) -> Book<'a> {
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
        let mut word;
        loop {
            let i = random.expovariate(LAMBDA) as usize;
            if i < self.words.len() / WORD_SIZE {
                word = &self.words[i * WORD_SIZE .. (i+1) * WORD_SIZE];
                while word.len() > 0 && word[word.len()-1] == b'\0' {
                    word = &word[..word.len() - 1];
                }
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
            .zip(self.words.chunks(WORD_SIZE))
            .collect::<Vec<_>>();

        tmp.sort_by_key(|&(c, _)| c);

        writeln!(write, "\n--\n\n\n\n\n\n\nMost common words:")?;

        for (_, word) in tmp.iter().rev().take(10) {
            let mut word: &[u8] = &word[..];
            while word.len() > 0 && word[word.len()-1] == b'\0' {
                word = &word[..word.len() - 1];
            }
            write.write_all(b"- ")?;
            write.write_all(word)?;
            write.write_all(b"\n")?;
            //writeln!(write, "- {}", word)?;
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

    markov.pack();

    println!("Generating artificial words");

    // Initialize python-compatible RNG
    let mt = MersenneTwister::new();
    let mut random = PythonRandom::new(mt);
    random.seed_u32(RANDOM_SEED);

    let mut word_set = FnvHashSet::<BString>::with_capacity_and_hasher(0, Default::default());

    const SET_BITS: usize = 32;

    let mut word_bitvec = bitbox![Lsb0, u64; 0; 1usize << SET_BITS];

    println!("Using {} megabytes of RAM for word_bitvec", word_bitvec.len() / 8 / 1024 / 1024);

    // The actual ordered list of words we will use later to pick words based on random numbers.
    // We could use a set that preserves insertion order, but the hash lookups we save by using
    // the word_set_short optimization outweighs the benefit of not duplicating storage.
    //let mut word_list = Vec::<BString>::with_capacity(DISTINCT_WORDS);

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
        let mut tmp: [Option<SmolBitvec>; 20] = Default::default();

        let weights_and_symbols_iter = lengths.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(length, &count)| (count, length));

        for (code, length) in huffman(weights_and_symbols_iter) {
            tmp[length] = Some(code);
        }

        tmp
    };

    let get_length_huffman = |wordlen: usize| {
        length_huffman.get(wordlen).and_then(|&x| x)
    };

    let mut huffgood = 0usize;
    let mut huffbad = 0usize;
    let mut huffmaxbits = 0u8;

    let mut wordbuf = vec![0u8; WORD_SIZE * DISTINCT_WORDS];
    let mut wordbuf_write = &mut wordbuf[..];

    let mut seen_above_8 = 0usize;

    while wordbuf_write.len() > 0 {

        w.clear();

        let mut huffman = Some(SmolBitvec::new());
        let mut huffindex: Option<usize>;

        // Add more letters until we find a word that hasn't been accepted yet.
        // (Could still be a word that will never be accepted due to later checks.)
        loop {
            // Add a new letter to the word and update the word_set_short index.
            let word_tail = &w[w.len().saturating_sub(2)..];

            let letter = markov.next_letter(word_tail, &mut random, &mut huffman);

            w.push(letter);

            huffindex = None;
            if let Some(huffman) = huffman {
                huffmaxbits = huffmaxbits.max(huffman.len());

                if let Some(huff_tmp) = get_length_huffman(w.len()).and_then(|l| l.checked_extend(&huffman)) {
                    if 1usize << huff_tmp.len() <= word_bitvec.len() {
                        huffindex = Some(huff_tmp.reverse().to_u64() as usize);
                    }
                }
            }

            if w.len() == 1 && !VOWELS.contains(&w.as_bytes()[0]) {
                break;
            }

            // TODO debug the eof-in-huffman code.
            // The reason for wanting a dedicated eof symbol at the end is that it might improve
            // locality near the end of the hashing maybe?
            // Having eof as a huffman symbol seems to make sense, but the problem is its
            // probability distribution is sort of indepenent of the markov tables. Or at
            // least not as well correlated, and it's hard to get a good estimate.
            // NOTE You need to make sure the start of your bitstream is always aligned the same
            //      (i.e. either hi-align it or flip the bit order).

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

            if w.len() > 8 {
                seen_above_8 += 1;
            }

            let (head, tail) = wordbuf_write.split_at_mut(WORD_SIZE);
            wordbuf_write = tail;
            head[..w.len()].copy_from_slice(&w);

            // Progress report.
            if (DISTINCT_WORDS - (wordbuf_write.len() / WORD_SIZE)) % 100000 == 0 {
                println!("{} words generated", DISTINCT_WORDS - (wordbuf_write.len() / WORD_SIZE));
                //println!("Length above 8: {}", seen_above_8);
            }
        }
    }

    drop(wordbuf_write);

    println!("{} out of {} lookups used the word_bitvec, {} didn't, longest huffman encoded word (sans length) was {} bits", huffgood, huffgood + huffbad, huffbad, huffmaxbits);

    // let mut lengths = [0usize; 20];
    // for w in &word_list {
    //     lengths[w.len()] += 1;
    // }
    // dbg!(lengths);

    drop(word_bitvec);
    //drop(word_set);

    println!("Capitalizing some words");

    for i in 0..DISTINCT_WORDS {
        if random.randint(0, 100) == 0 {
            wordbuf[i * WORD_SIZE..i * WORD_SIZE + 1].make_ascii_uppercase();
        }
    }

    println!("Generating text");

    let mut writer = io::BufWriter::with_capacity(1024 * 1024 * 4, fs::File::create("giganovel.txt").unwrap());

    //let mut writer = io::Cursor::new(vec![0u8; 1073742004]);

    let mut book = Book::new(&wordbuf);

    // Random number generation has been moved to next_word() so it can directly use the generated id
    // to maintain a count of how often each word was produced instead of needing a separate hashmap.
    while book.len() < BOOK_SIZE {
        book.next_word(&mut random, &mut writer)?;
    }
    book.end(&mut writer)?;

    //fs::File::create("giganovel.txt").unwrap().write_all(writer.into_inner().as_mut())?;

    Ok(())
}
