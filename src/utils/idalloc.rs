use bitvec::vec::BitVec;

#[derive(Default)]
pub struct IdAlloc {
    bits: BitVec,
}

impl IdAlloc {
    pub fn new() -> Self {
        Self {
            bits: BitVec::new(),
        }
    }
    pub fn alloc_one(&mut self) -> u32 {
        if let Some(indice) = self.bits.first_zero() {
            return indice as u32;
        }
        let indice = self.bits.len();
        self.bits.push(true);
        indice as u32
    }
    pub fn alloc(&mut self, n: u32) -> u32 {
        let mut range_start: u32 = 0;
        let mut len: u32 = 0;
        for i in self.bits.iter_zeros() {
            let i = i as u32;
            if len == 0 {
                // Initial case
                range_start = i;
                len += 1;
                if n == 1 {
                    self.bits.set(i as usize, true);
                    return i;
                }
            } else if i == range_start + len {
                // Consecutive number
                len += 1;
                if len == n {
                    // matched all numbers.
                    let slice = &mut self.bits[range_start as usize..(range_start + len) as usize];
                    slice.fill(true);
                    return range_start;
                }
            } else {
                // Sequence broke. Return to initial state.
                len = 1;
                range_start = i;
            }
        }

        let i = self.bits.len();
        // Bitfield not long enough. Extend bitfield.
        self.bits.extend(std::iter::repeat_n(true, n as usize));
        i as u32
    }
    pub fn free(&mut self, id: u32, n: u32) {
        let slice = &mut self.bits[id as usize..(id + n) as usize];
        slice.fill(false);
    }
}
