use std::cmp::max;
use std::fmt;
use std::slice;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

pub struct RandomState {
    index: usize,
    state: [u32; N],
}

impl fmt::Debug for RandomState {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("RandomState")
            .field("index", &self.index)
            .field("state", &&self.state[..])
            .finish()
    }
}

type BigInt = u64; // could use `num` module to make this generic so you can use bigints etc

impl RandomState {
    pub fn new() -> RandomState {
        RandomState {
            index: 0,
            state: [0; N],
        }
    }

    fn genrand_int32(&mut self) -> u32 {
        let mut y: u32;
        let mt = &mut self.state;

        if self.index >= N {
            for kk in 0..(N - M) {
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            }

            for kk in (N - M)..(N - 1) {
                y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk - (N - M)] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);
            }

            y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1) * MATRIX_A);

            self.index = 0;
        }

        y = mt[self.index];
        self.index += 1;

        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    fn init_genrand(&mut self, s: u32) {
        let mt = &mut self.state;

        mt[0] = s;
        for mti in 1..N {
            mt[mti] = 1812433253u32
                .wrapping_mul(mt[mti - 1] ^ (mt[mti - 1] >> 30))
                .wrapping_add(mti as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {

        self.init_genrand(19650218);

        let mt = &mut self.state;

        let mut i: usize = 1;
        let mut j: usize = 0;

        for _ in 0..max(N, init_key.len()) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32);

            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= init_key.len() {
                j = 0;
            }
        }

        for _ in 0..(N - 1) {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }

        mt[0] = 0x80000000;
    }

    fn getrandbits(&mut self, k: u32) -> BigInt {
        assert!(0 < k && k <= BigInt::from(0u8).leading_zeros());

        if k <= 32 {
            return BigInt::from(self.genrand_int32() >> (32 - k));
        }

        let mut tmp = BigInt::from(0u8);
        let mut k = k;
        let mut shift = 0;
        while k > 0 {
            tmp |= BigInt::from(self.genrand_int32() >> 32u32.saturating_sub(k)) << shift;
            k = k.saturating_sub(32);
            shift += 32;
        }

        tmp
    }

    fn randbelow(&mut self, n: BigInt) -> BigInt {
        let n_bits = BigInt::from(0u8).leading_zeros() - n.leading_zeros();
        let mut r = self.getrandbits(n_bits);
        while r >= n {
            r = self.getrandbits(n_bits);
        }
        r
    }

    pub fn seed_u32(&mut self, s: u32) {
        self.init_by_array(slice::from_ref(&s));
    }

    pub fn random(&mut self) -> f64 {
        let a = self.genrand_int32() >> 5;
        let b = self.genrand_int32() >> 6;
        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }

    pub fn expovariate(&mut self, lambda: f64) -> f64 {
        -(1.0 - self.random()).ln() / lambda
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..(x.len())).rev() {
            let j = self.randbelow(BigInt::from(i as u64) + 1) as usize;
            x.swap(i, j);
        }
    }

    pub fn randint(&mut self, start: BigInt, stop: BigInt) -> BigInt {
        self.randrange(start, stop + 1)
    }

    pub fn randrange(&mut self, start: BigInt, stop: BigInt) -> BigInt {
        start + self.randbelow(stop - start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity_checks() {
        // Known-good values generated using the following python program:
        //
        // import random
        // random.seed(63245986)
        // print(random.getstate())
        // print(random.random())
        // print(random.randrange(0, 100000))
        // print(random.expovariate(1.0 / 15000.0))
        //
        // tmp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        // random.shuffle(tmp)
        //
        // print(tmp)

        let mut rand = RandomState::new();

        rand.seed_u32(63245986);

        // println!("{:?}", &rand);

        assert_eq!(rand.random(), 0.5213761361171212);

        assert_eq!(rand.randbelow(100000u64), 58671);

        assert_eq!(rand.expovariate(1.0 / 15000.0), 13775.46713470634);

        let mut list: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        rand.shuffle(&mut list[..]);

        assert_eq!(&list, &[10, 3, 6, 1, 8, 5, 7, 4, 2, 9]);
    }
}
