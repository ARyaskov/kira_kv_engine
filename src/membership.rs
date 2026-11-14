/// Membership information available to an index's ordinary lookup methods.
/// Inspect this before using a candidate position as proof that a key exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MembershipMode {
    /// The caller must restrict queries to the original build set.
    ClosedWorld,
    /// Negative answers are definitive, but positive answers may be false.
    Probabilistic,
    /// Positive answers compare the original key rather than a fingerprint.
    Exact,
}
