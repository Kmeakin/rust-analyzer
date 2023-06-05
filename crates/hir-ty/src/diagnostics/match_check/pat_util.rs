//! Pattern utilities.
//!
//! Originates from `rustc_hir::pat_util`

use std::iter::{Enumerate, ExactSizeIterator};

pub(crate) struct EnumerateAndAdjust<I> {
    enumerate: Enumerate<I>,
    gap_pos: u32,
    gap_len: u32,
}

impl<I> Iterator for EnumerateAndAdjust<I>
where
    I: Iterator,
{
    type Item = (u32, <I as Iterator>::Item);

    fn next(&mut self) -> Option<(u32, <I as Iterator>::Item)> {
        self.enumerate
            .next()
            .map(|(i, elem)| (i as u32, elem))
            .map(|(i, elem)| (if i < self.gap_pos { i } else { i + self.gap_len }, elem))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.enumerate.size_hint()
    }
}

pub(crate) trait EnumerateAndAdjustIterator {
    fn enumerate_and_adjust(
        self,
        expected_len: u32,
        gap_pos: Option<u32>,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized;
}

impl<T: ExactSizeIterator> EnumerateAndAdjustIterator for T {
    fn enumerate_and_adjust(
        self,
        expected_len: u32,
        gap_pos: Option<u32>,
    ) -> EnumerateAndAdjust<Self>
    where
        Self: Sized,
    {
        let actual_len = self.len() as u32;
        EnumerateAndAdjust {
            enumerate: self.enumerate(),
            gap_pos: gap_pos.unwrap_or(expected_len),
            gap_len: expected_len - actual_len,
        }
    }
}
