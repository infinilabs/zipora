//! Numeric string comparison utilities
//!
//! Provides string comparison functions that properly handle numeric values,
//! including signed decimals and real numbers with decimal points.
//!
//! Ported from topling-zip's fstring decimal_strcmp and realnum_strcmp.

use std::cmp::Ordering;

/// Compare two numeric strings as decimal integers
///
/// Handles:
/// - Optional leading sign (+/-)
/// - Leading zeros are significant for ordering
/// - Returns `None` for invalid numeric strings
///
/// # Examples
///
/// ```rust
/// use zipora::string::decimal_strcmp;
///
/// assert_eq!(decimal_strcmp("123", "456"), Some(Ordering::Less));
/// assert_eq!(decimal_strcmp("-10", "5"), Some(Ordering::Less));
/// assert_eq!(decimal_strcmp("100", "99"), Some(Ordering::Greater));
/// assert_eq!(decimal_strcmp("42", "42"), Some(Ordering::Equal));
/// assert_eq!(decimal_strcmp("abc", "123"), None); // Invalid
/// ```
pub fn decimal_strcmp(a: &str, b: &str) -> Option<Ordering> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let (a_str, a_neg) = parse_sign(a)?;
    let (b_str, b_neg) = parse_sign(b)?;

    // Validate digits only
    if !a_str.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if !b_str.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }

    Some(decimal_strcmp_with_sign(a_str, a_neg, b_str, b_neg))
}

/// Compare two numeric strings with pre-parsed signs
///
/// This is the lower-level function when you've already parsed the sign.
///
/// # Arguments
///
/// * `a` - First number string (digits only, no sign)
/// * `a_neg` - Whether first number is negative
/// * `b` - Second number string (digits only, no sign)
/// * `b_neg` - Whether second number is negative
pub fn decimal_strcmp_with_sign(a: &str, a_neg: bool, b: &str, b_neg: bool) -> Ordering {
    // Different signs: negative < positive
    match (a_neg, b_neg) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }

    // Same sign: compare magnitudes
    let cmp = compare_decimal_magnitude(a, b);

    // For negative numbers, reverse the comparison
    if a_neg {
        cmp.reverse()
    } else {
        cmp
    }
}

/// Compare two real number strings (with optional decimal point)
///
/// Handles:
/// - Optional leading sign (+/-)
/// - At most one decimal point
/// - No scientific notation (use a different function for that)
///
/// # Examples
///
/// ```rust
/// use zipora::string::realnum_strcmp;
///
/// assert_eq!(realnum_strcmp("3.14", "2.71"), Some(std::cmp::Ordering::Greater));
/// assert_eq!(realnum_strcmp("-1.5", "1.5"), Some(std::cmp::Ordering::Less));
/// assert_eq!(realnum_strcmp("10", "9.99"), Some(std::cmp::Ordering::Greater));
/// assert_eq!(realnum_strcmp("1.0", "1.0"), Some(std::cmp::Ordering::Equal));
/// ```
pub fn realnum_strcmp(a: &str, b: &str) -> Option<Ordering> {
    if a.is_empty() || b.is_empty() {
        return None;
    }

    let (a_str, a_neg) = parse_sign(a)?;
    let (b_str, b_neg) = parse_sign(b)?;

    // Validate: digits and at most one decimal point
    if !validate_realnum(a_str) {
        return None;
    }
    if !validate_realnum(b_str) {
        return None;
    }

    Some(realnum_strcmp_with_sign(a_str, a_neg, b_str, b_neg))
}

/// Compare two real number strings with pre-parsed signs
pub fn realnum_strcmp_with_sign(a: &str, a_neg: bool, b: &str, b_neg: bool) -> Ordering {
    // Different signs: negative < positive
    match (a_neg, b_neg) {
        (true, false) => return Ordering::Less,
        (false, true) => return Ordering::Greater,
        _ => {}
    }

    // Find decimal point positions
    let a_dot = a.find('.').unwrap_or(a.len());
    let b_dot = b.find('.').unwrap_or(b.len());

    let cmp = if a_dot == b_dot {
        // Same integer part length - lexicographic comparison works
        a.cmp(b)
    } else {
        // Different integer part lengths - longer integer part is larger
        a_dot.cmp(&b_dot)
    };

    // For negative numbers, reverse the comparison
    if a_neg {
        cmp.reverse()
    } else {
        cmp
    }
}

// Helper: parse optional sign, return (remaining_str, is_negative)
fn parse_sign(s: &str) -> Option<(&str, bool)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    match bytes[0] {
        b'+' => {
            let rest = &s[1..];
            if rest.is_empty() {
                None
            } else {
                Some((rest, false))
            }
        }
        b'-' => {
            let rest = &s[1..];
            if rest.is_empty() {
                None
            } else {
                Some((rest, true))
            }
        }
        _ => Some((s, false)),
    }
}

// Helper: validate real number string (digits and at most one dot)
fn validate_realnum(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut dot_count = 0;
    for c in s.bytes() {
        if c == b'.' {
            dot_count += 1;
            if dot_count > 1 {
                return false;
            }
        } else if !c.is_ascii_digit() {
            return false;
        }
    }
    true
}

// Helper: compare decimal magnitude (same-sign comparison)
fn compare_decimal_magnitude(a: &str, b: &str) -> Ordering {
    // Strip leading zeros for comparison
    let a_stripped = a.trim_start_matches('0');
    let b_stripped = b.trim_start_matches('0');

    // Handle empty after stripping (means value is 0)
    let a_stripped = if a_stripped.is_empty() { "0" } else { a_stripped };
    let b_stripped = if b_stripped.is_empty() { "0" } else { b_stripped };

    // Compare by length first (longer = larger for positive integers)
    match a_stripped.len().cmp(&b_stripped.len()) {
        Ordering::Equal => a_stripped.cmp(b_stripped),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_strcmp_basic() {
        assert_eq!(decimal_strcmp("123", "456"), Some(Ordering::Less));
        assert_eq!(decimal_strcmp("456", "123"), Some(Ordering::Greater));
        assert_eq!(decimal_strcmp("42", "42"), Some(Ordering::Equal));
    }

    #[test]
    fn test_decimal_strcmp_signed() {
        assert_eq!(decimal_strcmp("-10", "5"), Some(Ordering::Less));
        assert_eq!(decimal_strcmp("5", "-10"), Some(Ordering::Greater));
        assert_eq!(decimal_strcmp("-5", "-10"), Some(Ordering::Greater)); // -5 > -10
        assert_eq!(decimal_strcmp("-10", "-5"), Some(Ordering::Less));
        assert_eq!(decimal_strcmp("+5", "5"), Some(Ordering::Equal));
    }

    #[test]
    fn test_decimal_strcmp_leading_zeros() {
        assert_eq!(decimal_strcmp("007", "7"), Some(Ordering::Equal));
        assert_eq!(decimal_strcmp("00", "0"), Some(Ordering::Equal));
    }

    #[test]
    fn test_decimal_strcmp_invalid() {
        assert_eq!(decimal_strcmp("", "123"), None);
        assert_eq!(decimal_strcmp("123", ""), None);
        assert_eq!(decimal_strcmp("abc", "123"), None);
        assert_eq!(decimal_strcmp("12.3", "123"), None); // decimal not allowed
        assert_eq!(decimal_strcmp("+", "123"), None);
    }

    #[test]
    fn test_realnum_strcmp_basic() {
        assert_eq!(realnum_strcmp("3.14", "2.71"), Some(Ordering::Greater));
        assert_eq!(realnum_strcmp("2.71", "3.14"), Some(Ordering::Less));
        assert_eq!(realnum_strcmp("1.0", "1.0"), Some(Ordering::Equal));
    }

    #[test]
    fn test_realnum_strcmp_integer_vs_decimal() {
        assert_eq!(realnum_strcmp("10", "9.99"), Some(Ordering::Greater));
        assert_eq!(realnum_strcmp("9", "9.99"), Some(Ordering::Less));
    }

    #[test]
    fn test_realnum_strcmp_signed() {
        assert_eq!(realnum_strcmp("-1.5", "1.5"), Some(Ordering::Less));
        assert_eq!(realnum_strcmp("-1.5", "-2.5"), Some(Ordering::Greater));
    }

    #[test]
    fn test_realnum_strcmp_invalid() {
        assert_eq!(realnum_strcmp("", "1.0"), None);
        assert_eq!(realnum_strcmp("1.2.3", "1.0"), None); // multiple dots
        assert_eq!(realnum_strcmp("abc", "1.0"), None);
    }
}
