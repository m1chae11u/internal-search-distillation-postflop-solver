use std::cell::UnsafeCell;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "bincode")]
use bincode::{
    Decode, Encode, BorrowDecode,
    de::{Decoder, BorrowDecoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
};

/// Mutex-like wrapper, but it actually does not perform any locking.
///
/// Use this wrapper when:
///   1. [`Send`], [`Sync`] and the interior mutability is needed,
///   2. it is (manually) guaranteed that data races will not occur, and
///   3. the performance is critical.
///
/// **Note**: This wrapper completely bypasses the "shared XOR mutable" rule of Rust.
/// Therefore, using this wrapper is **extremely unsafe** and should be avoided whenever possible.
#[derive(Debug)]
#[repr(transparent)]
pub struct MutexLike<T: ?Sized> {
    data: UnsafeCell<T>,
}

/// Smart pointer like wrapper that is returned when [`MutexLike`] is "locked".
#[derive(Debug)]
pub struct MutexGuardLike<'a, T: ?Sized + 'a> {
    mutex: &'a MutexLike<T>,
}

unsafe impl<T: ?Sized + Send> Send for MutexLike<T> {}
unsafe impl<T: ?Sized + Send> Sync for MutexLike<T> {}
unsafe impl<'a, T: ?Sized + Sync + 'a> Sync for MutexGuardLike<'a, T> {}

impl<T> MutexLike<T> {
    /// Creates a new [`MutexLike`] with the given value.
    ///
    /// # Examples
    /// ```
    /// use postflop_solver::MutexLike;
    ///
    /// let mutex_like = MutexLike::new(0);
    /// ```
    #[inline]
    pub fn new(val: T) -> Self {
        Self {
            data: UnsafeCell::new(val),
        }
    }
}

impl<T: ?Sized> MutexLike<T> {
    /// Acquires a mutex-like object **without** performing any locking.
    ///
    /// # Examples
    /// ```
    /// use postflop_solver::MutexLike;
    ///
    /// let mutex_like = MutexLike::new(0);
    /// *mutex_like.lock() = 10;
    /// assert_eq!(*mutex_like.lock(), 10);
    /// ```
    #[inline]
    pub fn lock(&self) -> MutexGuardLike<T> {
        MutexGuardLike { mutex: self }
    }
}

impl<T: ?Sized + Default> Default for MutexLike<T> {
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<'a, T: ?Sized + 'a> Deref for MutexGuardLike<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T: ?Sized + 'a> DerefMut for MutexGuardLike<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

#[cfg(feature = "bincode")]
impl<T: Encode> Encode for MutexLike<T> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        self.lock().encode(encoder)
    }
}

#[cfg(feature = "bincode")]
impl<C, T: Decode<C>> Decode<C> for MutexLike<T> {
    fn decode<D: Decoder<Context = C>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let inner = T::decode(decoder)?;
        Ok(MutexLike::new(inner))
    }
}

#[cfg(feature = "bincode")]
impl<'de, C, T: BorrowDecode<'de, C>> BorrowDecode<'de, C> for MutexLike<T> {
    fn borrow_decode<D: BorrowDecoder<'de, Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, DecodeError> {
        let inner = T::borrow_decode(decoder)?;
        Ok(MutexLike::new(inner))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let mutex_like = MutexLike::new(42);
        let mut encoder = bincode::enc::Encoder::new();
        mutex_like.encode(&mut encoder).unwrap();
        let decoded_mutex_like = Decode::decode(&mut bincode::de::Decoder::new(encoder.into_inner())).unwrap();
        assert_eq!(*decoded_mutex_like.lock(), 42);
    }

    #[test]
    fn test_borrow_decode() {
        let mutex_like = MutexLike::new("Hello, world!");
        let mut encoder = bincode::enc::Encoder::new();
        mutex_like.encode(&mut encoder).unwrap();
        let decoded_mutex_like = BorrowDecode::borrow_decode(&mut bincode::de::BorrowDecoder::new(encoder.into_inner())).unwrap();
        assert_eq!(*decoded_mutex_like.lock(), "Hello, world!");
    }
}
