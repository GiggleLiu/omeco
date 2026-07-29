//! Label trait for tensor index labels.
//!
//! Labels can be characters (e.g., 'i', 'j', 'k') or integers,
//! used to identify tensor dimensions in einsum expressions.

use std::fmt::Debug;
use std::hash::Hash;

/// Trait for tensor index labels.
///
/// Labels are used to identify tensor dimensions in einsum notation.
/// Common choices are `char` (e.g., `'i'`, `'j'`) or `usize` for programmatic use.
///
/// Requires `Ord` for deterministic preprocessing operations that rely on
/// consistent ordering of labels.
///
/// # Example
/// ```
/// use omeco::Label;
///
/// fn process_labels<L: Label>(labels: &[L]) {
///     // Works with any label type
/// }
/// ```
pub trait Label: Clone + Eq + Ord + Hash + Debug + Send + Sync + 'static {}

// Implement Label for common types
impl Label for char {}
impl Label for u8 {}
impl Label for u16 {}
impl Label for u32 {}
impl Label for u64 {}
impl Label for usize {}
impl Label for i8 {}
impl Label for i16 {}
impl Label for i32 {}
impl Label for i64 {}
impl Label for isize {}
impl Label for String {}

#[cfg(test)]
mod tests {
    use super::*;

    fn requires_label<L: Label>(_: L) {}

    #[test]
    fn test_char_is_label() {
        requires_label('i');
    }

    #[test]
    fn test_usize_is_label() {
        requires_label(42usize);
    }

    #[test]
    fn test_string_is_label() {
        requires_label(String::from("index"));
    }
}
