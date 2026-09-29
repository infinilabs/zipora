//! Strategy Traits for Hash Map Implementations
//!
//! This module defines strategy traits for building a hash table over a
//! caller-owned array of [`HashBucket`]s, plus the concrete strategies the
//! crate provides.
//!
//! # Strategy Architecture
//!
//! The strategy pattern allows different algorithms to be combined:
//! - **CollisionResolutionStrategy**: How to handle hash collisions
//! - **StorageLayoutStrategy**: How to organize data in memory
//! - **HashOptimizationStrategy**: Performance optimizations (SIMD, cache, etc.)
//!
//! The crate implements [`RobinHoodStrategy`] (collision resolution) and
//! [`StandardStorageStrategy`] (storage layout). It ships no
//! `HashOptimizationStrategy` implementation.
//!
//! [`ZiporaHashMap`](crate::hash_map::ZiporaHashMap) does not use these
//! traits; it is configured through the enums in
//! [`ZiporaHashMapConfig`](crate::hash_map::ZiporaHashMapConfig).

use crate::containers::FastVec;
use crate::error::{Result, ZiporaError};
use std::borrow::Borrow;
use std::collections::VecDeque;
use std::hash::Hash;

/// Collision resolution strategy for hash maps
pub trait CollisionResolutionStrategy<K, V> {
    /// Configuration for this strategy
    type Config: Clone;

    /// Context/state maintained by this strategy
    type Context: Default;

    /// Insert a key-value pair using this collision resolution strategy
    fn insert(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: K,
        value: V,
        hash: u64,
        config: &Self::Config,
    ) -> Result<Option<V>>;

    /// Lookup a key using this collision resolution strategy
    fn get<'a, Q>(
        &self,
        context: &Self::Context,
        buckets: &'a [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Remove a key using this collision resolution strategy
    fn remove<Q>(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Check if resize is needed
    fn needs_resize(&self, context: &Self::Context, capacity: usize, len: usize) -> bool;

    /// Get probe statistics for this strategy
    fn probe_stats(&self, context: &Self::Context) -> ProbeStats;
}

/// Storage layout strategy for memory organization
pub trait StorageLayoutStrategy<K, V> {
    /// Configuration for this strategy
    type Config: Clone;

    /// Storage context/state
    type Storage;

    /// Initialize storage with given capacity
    fn create_storage(capacity: usize, config: &Self::Config) -> Self::Storage;

    /// Resize storage to new capacity
    fn resize_storage(
        storage: &mut Self::Storage,
        new_capacity: usize,
        config: &Self::Config,
    ) -> Result<()>;

    /// Get bucket array from storage
    fn get_buckets_mut(storage: &mut Self::Storage) -> &mut [HashBucket<K, V>];

    /// Get bucket array from storage (immutable)
    fn get_buckets(storage: &Self::Storage) -> &[HashBucket<K, V>];

    /// Get capacity of storage
    fn capacity(storage: &Self::Storage) -> usize;

    /// Estimate memory usage in bytes
    fn memory_usage(storage: &Self::Storage) -> usize;

    /// Perform layout-specific optimizations
    fn optimize_layout(storage: &mut Self::Storage, config: &Self::Config) -> Result<()>;
}

/// Hash optimization strategy for performance features
pub trait HashOptimizationStrategy<K, V> {
    /// Configuration for optimizations
    type Config: Clone;

    /// Optimization context/state
    type Context: Default;

    /// Pre-insert optimization (e.g., prefetching)
    fn pre_insert(
        &self,
        context: &mut Self::Context,
        key: &K,
        hash: u64,
        buckets: &[HashBucket<K, V>],
        config: &Self::Config,
    );

    /// Post-insert optimization (e.g., cache management)
    fn post_insert(
        &self,
        context: &mut Self::Context,
        key: &K,
        inserted: bool,
        config: &Self::Config,
    );

    /// Pre-lookup optimization
    fn pre_lookup<Q>(
        &self,
        context: &mut Self::Context,
        key: &Q,
        hash: u64,
        buckets: &[HashBucket<K, V>],
        config: &Self::Config,
    ) where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Bulk optimization for multiple operations
    fn bulk_optimize(
        &self,
        context: &mut Self::Context,
        operations: &[OptimizationHint],
        config: &Self::Config,
    );

    /// Get optimization metrics
    fn metrics(&self, context: &Self::Context) -> OptimizationMetrics;
}

/// Generic hash bucket for different strategies
#[repr(align(64))] // Cache line alignment
#[derive(Clone)]
pub struct HashBucket<K, V> {
    /// Hash value (cached for performance)
    pub hash: u64,
    /// Key
    pub key: Option<K>,
    /// Value
    pub value: Option<V>,
    /// Probe distance for Robin Hood hashing
    pub probe_distance: u16,
    /// Strategy-specific flags
    pub flags: u16,
    /// Next bucket index for chaining
    pub next: Option<u32>,
    /// Strategy-specific data
    pub strategy_data: u64,
}

impl<K, V> Default for HashBucket<K, V> {
    fn default() -> Self {
        Self {
            hash: 0,
            key: None,
            value: None,
            probe_distance: 0,
            flags: 0,
            next: None,
            strategy_data: 0,
        }
    }
}

impl<K, V> HashBucket<K, V> {
    /// Check if bucket is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.key.is_none()
    }

    /// Check if bucket is deleted (tombstone)
    pub fn is_deleted(&self) -> bool {
        self.flags & 0x8000 != 0
    }

    /// Mark bucket as deleted
    pub fn mark_deleted(&mut self) {
        self.flags |= 0x8000;
        self.key = None;
        self.value = None;
    }

    /// Clear bucket
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Probe statistics for collision resolution
#[derive(Debug, Default, Clone)]
pub struct ProbeStats {
    pub average_probe_distance: f64,
    pub max_probe_distance: u16,
    pub total_probes: u64,
    pub collision_count: u64,
    pub variance: f64,
}

/// Optimization metrics
#[derive(Debug, Default, Clone)]
pub struct OptimizationMetrics {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub prefetch_hits: u64,
    pub simd_operations: u64,
    pub bulk_operations: u64,
}

/// Optimization hints for bulk operations
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Sequential access pattern
    Sequential { start_hash: u64, count: usize },
    /// Random access pattern
    Random { hashes: Vec<u64> },
    /// Bulk insertion
    BulkInsert { count: usize },
    /// Bulk lookup
    BulkLookup { count: usize },
    /// Cache warming
    CacheWarm {
        bucket_range: std::ops::Range<usize>,
    },
}

// Concrete strategy implementations

/// `HashBucket::flags` bit set by [`HashBucket::mark_deleted`].
const DELETED_FLAG: u16 = 0x8000;

/// A slot that holds neither a live entry nor a tombstone.
#[inline]
fn is_vacant<K, V>(bucket: &HashBucket<K, V>) -> bool {
    bucket.key.is_none() && !bucket.is_deleted()
}

/// A slot that holds a key/value pair.
#[inline]
fn is_live<K, V>(bucket: &HashBucket<K, V>) -> bool {
    bucket.key.is_some() && !bucket.is_deleted()
}

/// Robin Hood collision resolution strategy
///
/// Works on a caller-owned bucket slice whose length is a non-zero power of
/// two. For any other length, `insert` returns `Err` and `get`/`remove` return
/// `None`.
///
/// # Removal modes
///
/// [`RobinHoodConfig::backward_shift`] selects the mode for each `remove`
/// call. The two modes may be mixed on one slice.
/// - `true`: backward-shift deletion. Each following slot (entry or
///   tombstone) moves back one place, until a vacant slot or an entry that is
///   at its home slot.
/// - `false`: the slot becomes a tombstone that keeps the removed entry's
///   `probe_distance` (its "phantom distance"). Lookups treat a tombstone as
///   an occupied slot with that distance, so the Robin Hood early exit (stop
///   once the probe has travelled further than the slot's occupant) stays
///   valid. `insert` reuses a tombstone only when the entry it is placing has
///   travelled at least that far. Reusing it sooner would cut off the entries
///   behind it. Only a rehash drops tombstones: `resize_storage` or
///   `optimize_layout` on [`StandardStorageStrategy`].
///
/// # Configuration
///
/// `RobinHoodConfig::max_probe_distance` bounds the distance at which `insert`
/// may place any entry, whether the new one or one it displaces. An insert
/// that would exceed it returns `Err` and leaves the slice unchanged. Lookups
/// are bounded only by the slice length and the early exit, so entries that a
/// rehash placed further out stay reachable.
/// `RobinHoodConfig::variance_reduction` has no effect. Of the arguments to
/// [`RobinHoodStrategy::new`], only `max_probe_distance` is read, and only by
/// `needs_resize`.
pub struct RobinHoodStrategy {
    max_probe_distance: u16,
    _variance_reduction: bool,
    _backward_shift: bool,
}

impl RobinHoodStrategy {
    /// `variance_reduction` and `backward_shift` are stored but never read;
    /// the per-call [`RobinHoodConfig`] controls behaviour.
    /// `max_probe_distance` is the threshold that `needs_resize` compares the
    /// context's high-water mark against.
    pub fn new(max_probe_distance: u16, variance_reduction: bool, backward_shift: bool) -> Self {
        Self {
            max_probe_distance,
            _variance_reduction: variance_reduction,
            _backward_shift: backward_shift,
        }
    }
}

/// Per-call configuration for [`RobinHoodStrategy`].
#[derive(Debug, Clone)]
pub struct RobinHoodConfig {
    /// Largest probe distance at which `insert` may place an entry.
    pub max_probe_distance: u16,
    /// Not read by [`RobinHoodStrategy`].
    pub variance_reduction: bool,
    /// `true`: backward-shift deletion. `false`: tombstones.
    pub backward_shift: bool,
}

/// Running statistics for a [`RobinHoodStrategy`].
///
/// `total_probe_distance`, `sum_squared_probe_distance` and `live_entries`
/// cover the entries placed by `insert` and not yet removed. They are updated
/// incrementally and are not recomputed when a [`StorageLayoutStrategy`]
/// rehashes the slice. `max_probe_distance` is a high-water mark.
/// `collision_count` counts the slots stepped past by each `insert` that
/// adds an entry, and `eviction_count` counts Robin Hood displacements. An
/// update of an existing key and a rejected insert change neither.
#[derive(Debug, Default)]
pub struct RobinHoodContext {
    pub total_probe_distance: u64,
    pub max_probe_distance: u16,
    pub collision_count: u64,
    pub eviction_count: u64,
    /// Number of live entries counted in the distance sums.
    pub live_entries: u64,
    /// Sum of the squared probe distances of live entries.
    pub sum_squared_probe_distance: u64,
}

// The three helpers are non-generic. Without `#[inline]`, the generic
// `insert`/`remove`, monomorphized in the caller's crate, call them
// out of line.
impl RobinHoodContext {
    #[inline]
    fn record_placed(&mut self, distance: u16) {
        let d = u64::from(distance);
        self.total_probe_distance = self.total_probe_distance.saturating_add(d);
        self.sum_squared_probe_distance = self.sum_squared_probe_distance.saturating_add(d * d);
        self.live_entries = self.live_entries.saturating_add(1);
        self.max_probe_distance = self.max_probe_distance.max(distance);
    }

    #[inline]
    fn record_moved(&mut self, from: u16, to: u16) {
        let (old_d, new_d) = (u64::from(from), u64::from(to));
        self.total_probe_distance = self
            .total_probe_distance
            .saturating_sub(old_d)
            .saturating_add(new_d);
        self.sum_squared_probe_distance = self
            .sum_squared_probe_distance
            .saturating_sub(old_d * old_d)
            .saturating_add(new_d * new_d);
        self.max_probe_distance = self.max_probe_distance.max(to);
    }

    #[inline]
    fn record_removed(&mut self, distance: u16) {
        let d = u64::from(distance);
        self.total_probe_distance = self.total_probe_distance.saturating_sub(d);
        self.sum_squared_probe_distance = self.sum_squared_probe_distance.saturating_sub(d * d);
        self.live_entries = self.live_entries.saturating_sub(1);
    }
}

/// Robin Hood lookup shared by `get` and `remove`. `insert` applies the same
/// rule inline and resumes it with [`rh_find_from`]. Returns the slot index of
/// the live entry for `key`, if any.
fn rh_find<K, V, Q>(buckets: &[HashBucket<K, V>], key: &Q, hash: u64) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    if buckets.is_empty() || !buckets.len().is_power_of_two() {
        return None;
    }
    let mask = buckets.len() - 1;
    rh_find_from(buckets, key, hash, mask, (hash as usize) & mask, 0)
}

/// The lookup of [`rh_find`], resumed at slot `pos` after stepping past
/// `start_distance` slots from the home slot. `mask` is `buckets.len() - 1`
/// for a non-empty power-of-two slice.
fn rh_find_from<K, V, Q>(
    buckets: &[HashBucket<K, V>],
    key: &Q,
    hash: u64,
    mask: usize,
    mut pos: usize,
    start_distance: usize,
) -> Option<usize>
where
    K: Borrow<Q>,
    Q: Eq + ?Sized,
{
    for probe_distance in start_distance..buckets.len() {
        let bucket = &buckets[pos];

        if is_vacant(bucket) {
            return None;
        }

        if is_live(bucket)
            && bucket.hash == hash
            && bucket.key.as_ref().map(|k| k.borrow()) == Some(key)
        {
            return Some(pos);
        }

        // Robin Hood: if we've probed further than this bucket's distance,
        // the key doesn't exist. A tombstone keeps the distance of the entry
        // it replaced, so the rule also holds across tombstones.
        if probe_distance > usize::from(bucket.probe_distance) {
            return None;
        }

        pos = (pos + 1) & mask;
    }

    None
}

/// Writes an entry at `distance` into a vacant slot or a tombstone.
fn rh_place<K, V>(
    context: &mut RobinHoodContext,
    bucket: &mut HashBucket<K, V>,
    hash: u64,
    key: Option<K>,
    value: Option<V>,
    distance: u16,
) {
    bucket.hash = hash;
    bucket.key = key;
    bucket.value = value;
    bucket.probe_distance = distance;
    bucket.flags &= !DELETED_FLAG;
    context.record_placed(distance);
}

impl<K, V> CollisionResolutionStrategy<K, V> for RobinHoodStrategy
where
    K: Hash + Eq,
{
    type Config = RobinHoodConfig;
    type Context = RobinHoodContext;

    fn insert(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: K,
        value: V,
        hash: u64,
        config: &Self::Config,
    ) -> Result<Option<V>> {
        if buckets.is_empty() || !buckets.len().is_power_of_two() {
            return Err(ZiporaError::invalid_state(
                "RobinHoodStrategy requires a non-empty power-of-two bucket slice",
            ));
        }

        let cap = buckets.len();
        let mask = cap - 1;
        let max_distance = usize::from(config.max_probe_distance);

        // Until the placement walk first acts, the new entry's distance is
        // the number of slots stepped past (`step`), and the lookup
        // (`rh_find`'s rule) ends exactly where the walk acts:
        // - a vacant slot: the lookup misses; the walk places the new entry;
        // - a live entry at a shorter distance: the lookup misses; the walk
        //   displaces that entry;
        // - a tombstone with a shorter phantom distance: the lookup misses;
        //   the walk takes the slot over.
        // Both step past every other slot, except a tombstone whose phantom
        // distance equals `step`: the walk may take it over, but the lookup
        // goes on past it. So one probe does both, and nothing changes
        // before an existing entry for `key` would have been found.
        let mut pos = (hash as usize) & mask;
        let (start, start_distance) = 'probe: {
            for step in 0..cap {
                if step > max_distance {
                    // No slot from here on is close enough for the new entry,
                    // but an existing entry for `key` is still updated.
                    return match rh_find_from(buckets, &key, hash, mask, pos, step) {
                        Some(found) => Ok(buckets[found].value.replace(value)),
                        None => Err(ZiporaError::invalid_state("Exceeded maximum probe distance")),
                    };
                }
                let bucket = &buckets[pos];
                let slot_distance = usize::from(bucket.probe_distance);
                let take_slot = if is_vacant(bucket) {
                    true
                } else if is_live(bucket) {
                    if bucket.hash == hash && bucket.key.as_ref() == Some(&key) {
                        // Key already exists - update value
                        return Ok(buckets[pos].value.replace(value));
                    }
                    if slot_distance < step {
                        break 'probe (pos, step);
                    }
                    false
                } else if slot_distance == step {
                    let next = (pos + 1) & mask;
                    if let Some(found) = rh_find_from(buckets, &key, hash, mask, next, step + 1) {
                        return Ok(buckets[found].value.replace(value));
                    }
                    true
                } else {
                    slot_distance < step
                };
                if take_slot {
                    context.collision_count += step as u64;
                    // `step <= max_distance <= u16::MAX`.
                    rh_place(context, &mut buckets[pos], hash, Some(key), Some(value), step as u16);
                    return Ok(None);
                }
                pos = (pos + 1) & mask;
            }
            return Err(ZiporaError::invalid_state(
                "RobinHoodStrategy: no vacant or reusable slot in the bucket slice",
            ));
        };

        // The probe reached the first entry to displace, at `start`, so the
        // lookup has missed. A dry run replays the rest of the walk below
        // without mutating anything. Every entry this insert would place (the
        // new one and each one it displaces) must fit within
        // `max_probe_distance`, and a vacant or reusable slot must come within
        // the `cap - start_distance` slots left of one cycle from home. So a
        // failed insert leaves the slice untouched.
        let steps_left = cap - start_distance;
        let mut probe = start;
        let mut sim_distance = start_distance;
        let mut slot_found = false;
        for _ in 0..steps_left {
            if sim_distance > max_distance {
                return Err(ZiporaError::invalid_state("Exceeded maximum probe distance"));
            }
            let bucket = &buckets[probe];
            // In a valid layout (each slot's distance is at most one more
            // than the previous slot's), no slot this walk reaches has a
            // larger distance than the entry in hand, so any tombstone reached
            // here may be taken. The distance test only matters on a slice
            // whose `HashBucket` fields were set by hand; it is kept identical
            // in the walk so that the two stay in step on such a slice.
            if is_vacant(bucket)
                || (bucket.is_deleted() && usize::from(bucket.probe_distance) <= sim_distance)
            {
                slot_found = true;
                break;
            }
            if is_live(bucket) && sim_distance > usize::from(bucket.probe_distance) {
                sim_distance = usize::from(bucket.probe_distance);
            }
            sim_distance += 1;
            probe = (probe + 1) & mask;
        }
        if !slot_found {
            return Err(ZiporaError::invalid_state(
                "RobinHoodStrategy: no vacant or reusable slot in the bucket slice",
            ));
        }

        // The new entry reached `start` at distance `start_distance`, having
        // stepped past that many slots. The dry run bounded every distance on
        // the walk by `max_probe_distance`, so each fits in `u16`.
        context.collision_count += start_distance as u64;
        let mut pos = start;
        let mut inserting_key = Some(key);
        let mut inserting_value = Some(value);
        let mut inserting_hash = hash;
        let mut inserting_distance = start_distance as u16;

        for _ in 0..steps_left {
            let bucket = &mut buckets[pos];

            // Empty bucket, or a tombstone this entry may take over (the dry
            // run's test) - insert here
            if is_vacant(bucket)
                || (bucket.is_deleted() && bucket.probe_distance <= inserting_distance)
            {
                rh_place(
                    context,
                    bucket,
                    inserting_hash,
                    inserting_key,
                    inserting_value,
                    inserting_distance,
                );
                return Ok(None);
            }

            // Robin Hood: displace if we've traveled further
            if is_live(bucket) && inserting_distance > bucket.probe_distance {
                context.record_moved(bucket.probe_distance, inserting_distance);
                // Swap the inserting entry with the bucket entry
                std::mem::swap(&mut bucket.hash, &mut inserting_hash);
                std::mem::swap(&mut bucket.key, &mut inserting_key);
                std::mem::swap(&mut bucket.value, &mut inserting_value);
                std::mem::swap(&mut bucket.probe_distance, &mut inserting_distance);
                context.eviction_count += 1;
            }

            pos = (pos + 1) & mask;
            // The dry run bounded every distance on this walk by
            // `max_probe_distance`, so this never saturates.
            inserting_distance = inserting_distance.saturating_add(1);
            context.collision_count += 1;
        }

        // Unreachable: the dry run replayed this exact walk and found a slot.
        Err(ZiporaError::invalid_state(
            "RobinHoodStrategy: insert walk diverged from its dry run",
        ))
    }

    fn get<'a, Q>(
        &self,
        _context: &Self::Context,
        buckets: &'a [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        _config: &Self::Config,
    ) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let pos = rh_find(buckets, key, hash)?;
        buckets[pos].value.as_ref()
    }

    fn remove<Q>(
        &self,
        context: &mut Self::Context,
        buckets: &mut [HashBucket<K, V>],
        key: &Q,
        hash: u64,
        config: &Self::Config,
    ) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Find the key
        let pos = rh_find(buckets, key, hash)?;
        let bucket = &mut buckets[pos];
        context.record_removed(bucket.probe_distance);
        let value = bucket.value.take();

        if config.backward_shift {
            // Backward shift deletion
            self.backward_shift_delete(context, buckets, pos);
        } else {
            // Mark as deleted (tombstone). `mark_deleted` keeps
            // `probe_distance`, which lookups and inserts read as the
            // tombstone's phantom distance.
            bucket.mark_deleted();
        }

        value
    }

    fn needs_resize(&self, context: &Self::Context, capacity: usize, len: usize) -> bool {
        if capacity == 0 {
            return true;
        }
        let load_factor = len as f64 / capacity as f64;
        load_factor > 0.7 || context.max_probe_distance > self.max_probe_distance
    }

    fn probe_stats(&self, context: &Self::Context) -> ProbeStats {
        let (avg_probe, variance) = if context.live_entries > 0 {
            let n = context.live_entries as f64;
            let mean = context.total_probe_distance as f64 / n;
            let mean_sq = context.sum_squared_probe_distance as f64 / n;
            (mean, (mean_sq - mean * mean).max(0.0))
        } else {
            (0.0, 0.0)
        };

        ProbeStats {
            average_probe_distance: avg_probe,
            max_probe_distance: context.max_probe_distance,
            total_probes: context.total_probe_distance,
            collision_count: context.collision_count,
            variance,
        }
    }
}

impl RobinHoodStrategy {
    fn backward_shift_delete<K, V>(
        &self,
        context: &mut RobinHoodContext,
        buckets: &mut [HashBucket<K, V>],
        mut pos: usize,
    ) {
        if buckets.is_empty() || !buckets.len().is_power_of_two() {
            return;
        }
        let mask = buckets.len() - 1;
        buckets[pos].clear();

        // At most `len - 1` other slots can move back.
        for _ in 1..buckets.len() {
            let next_pos = (pos + 1) & mask;
            let next_bucket = &buckets[next_pos];

            // Stop if next bucket is empty or at its ideal position
            if is_vacant(next_bucket) || next_bucket.probe_distance == 0 {
                break;
            }

            // Move the next bucket backward. `buckets[pos]` is the cleared
            // slot, so the swap leaves `next_pos` cleared for the next step.
            // A tombstone moves too, so its phantom distance stays accurate.
            if is_live(next_bucket) {
                let d = next_bucket.probe_distance;
                context.record_moved(d, d - 1);
            }
            buckets.swap(pos, next_pos);
            buckets[pos].probe_distance -= 1;

            pos = next_pos;
        }
    }
}

/// Largest power of two `<= n` (0 for `n == 0`).
#[inline]
fn prev_power_of_two(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Largest probe distance in the linear-probing layout of a table where
/// `home_counts[h]` entries have home slot `h`. The Robin Hood layout orders
/// each cluster by home slot, and all such layouts of the same entries have
/// the same distances, so this is also the largest Robin Hood distance.
/// Requires `home_counts.iter().sum::<usize>() <= home_counts.len()`.
fn max_linear_probe_distance(home_counts: &[usize]) -> usize {
    let cap = home_counts.len();
    if cap == 0 {
        return 0;
    }
    // Start the sweep just after the lowest point of the running sum of
    // (count - 1). No probe sequence crosses into that slot from before it, so
    // the sweep never has to account for entries that wrapped around.
    let mut running: i128 = 0;
    let mut lowest: i128 = 0;
    let mut start = 0usize;
    for (slot, &count) in home_counts.iter().enumerate() {
        running += count as i128 - 1;
        if running < lowest {
            lowest = running;
            start = slot + 1;
        }
    }
    let start = start % cap;

    // FIFO of (sweep step of the home slot, entries still unplaced).
    let mut waiting: VecDeque<(usize, usize)> = VecDeque::new();
    let mut max_distance = 0usize;
    for step in 0..cap {
        let count = home_counts[(start + step) % cap];
        if count > 0 {
            waiting.push_back((step, count));
        }
        if let Some(front) = waiting.front_mut() {
            max_distance = max_distance.max(step - front.0);
            front.1 -= 1;
            if front.1 == 0 {
                waiting.pop_front();
            }
        }
    }
    max_distance
}

/// Rehashes the live entries of `storage` into a new power-of-two table of at
/// least `new_capacity` slots, dropping tombstones and recomputing Robin Hood
/// probe distances. On error, `storage` is left unchanged.
fn rehash_bucket_storage<K, V>(
    storage: &mut FastVec<HashBucket<K, V>>,
    new_capacity: usize,
) -> Result<()> {
    if new_capacity == 0 {
        return Err(ZiporaError::invalid_data(
            "Hash table storage capacity must be greater than 0",
        ));
    }
    let target_capacity = new_capacity
        .checked_next_power_of_two()
        .ok_or_else(|| ZiporaError::out_of_memory(new_capacity))?;
    let mask = target_capacity - 1;

    let live_count = storage.as_slice().iter().filter(|b| is_live(b)).count();
    if target_capacity < live_count {
        return Err(ZiporaError::invalid_data(
            "Cannot resize hash storage below active entry count",
        ));
    }

    // Every slot between an entry's home and its position holds another entry
    // of the same cluster, so a distance never exceeds `live_count - 1`. The
    // u16 `probe_distance` can therefore only overflow with more than 65,536
    // live entries, and only then is the layout checked up front.
    if live_count > usize::from(u16::MAX) + 1 {
        let mut home_counts: Vec<usize> = Vec::new();
        home_counts.try_reserve_exact(target_capacity).map_err(|_| {
            ZiporaError::out_of_memory(
                target_capacity.saturating_mul(std::mem::size_of::<usize>()),
            )
        })?;
        home_counts.resize(target_capacity, 0);
        for bucket in storage.as_slice().iter().filter(|b| is_live(b)) {
            home_counts[(bucket.hash as usize) & mask] += 1;
        }
        if max_linear_probe_distance(&home_counts) > usize::from(u16::MAX) {
            return Err(ZiporaError::invalid_data(
                "Rehash would place an entry beyond the u16 probe distance range",
            ));
        }
    }

    let mut rehashed = FastVec::with_capacity(target_capacity)?;
    rehashed.resize_with(target_capacity, HashBucket::default)?;
    let buckets = rehashed.as_mut_slice();
    for old in storage.as_mut_slice().iter_mut() {
        if !is_live(old) {
            continue;
        }
        let mut entry = std::mem::take(old);
        entry.probe_distance = 0;
        let mut pos = (entry.hash as usize) & mask;
        // Terminates: at most `target_capacity` entries are placed, so a
        // vacant slot lies ahead of every walk.
        loop {
            if is_vacant(&buckets[pos]) {
                buckets[pos] = entry;
                break;
            }
            if entry.probe_distance > buckets[pos].probe_distance {
                std::mem::swap(&mut buckets[pos], &mut entry);
            }
            pos = (pos + 1) & mask;
            // Cannot overflow: bounded by the distance check above.
            entry.probe_distance += 1;
        }
    }
    *storage = rehashed;
    Ok(())
}

/// Standard storage layout strategy
///
/// Keeps the buckets in a [`FastVec`] whose length is zero or a power of two,
/// as [`RobinHoodStrategy`] requires. `resize_storage` rehashes the live
/// entries into the new table and drops tombstones. `optimize_layout` does the
/// same at the current size when the table holds tombstones.
pub struct StandardStorageStrategy;

/// Configuration for [`StandardStorageStrategy`]. The strategy reads neither
/// field; `create_storage` and `resize_storage` take explicit capacities.
#[derive(Debug, Clone)]
pub struct StandardStorageConfig {
    pub initial_capacity: usize,
    pub growth_factor: f64,
}

impl<K, V> StorageLayoutStrategy<K, V> for StandardStorageStrategy {
    type Config = StandardStorageConfig;
    type Storage = FastVec<HashBucket<K, V>>;

    fn create_storage(capacity: usize, _config: &Self::Config) -> Self::Storage {
        // Mask-based probing requires a power-of-two length. A capacity too
        // large to round up gets the same empty result as a failed allocation.
        let target_capacity = if capacity == 0 {
            0
        } else {
            capacity.checked_next_power_of_two().unwrap_or(0)
        };
        // Graceful fallback: try full capacity, then half, then empty vec
        let mut storage = FastVec::with_capacity(target_capacity)
            .or_else(|_| FastVec::with_capacity(target_capacity / 2))
            .unwrap_or_else(|_| FastVec::new());

        // Only resize if we got some capacity
        if storage.capacity() > 0 {
            let len = prev_power_of_two(storage.capacity());
            if storage.resize_with(len, HashBucket::default).is_err() {
                storage.clear();
            }
        }
        storage
    }

    fn resize_storage(
        storage: &mut Self::Storage,
        new_capacity: usize,
        _config: &Self::Config,
    ) -> Result<()> {
        rehash_bucket_storage(storage, new_capacity)
    }

    fn get_buckets_mut(storage: &mut Self::Storage) -> &mut [HashBucket<K, V>] {
        storage.as_mut_slice()
    }

    fn get_buckets(storage: &Self::Storage) -> &[HashBucket<K, V>] {
        storage.as_slice()
    }

    fn capacity(storage: &Self::Storage) -> usize {
        storage.len()
    }

    fn memory_usage(storage: &Self::Storage) -> usize {
        storage.capacity() * std::mem::size_of::<HashBucket<K, V>>()
    }

    fn optimize_layout(storage: &mut Self::Storage, _config: &Self::Config) -> Result<()> {
        // Drop tombstones by rehashing at the current size.
        if storage.as_slice().iter().any(|b| b.is_deleted()) {
            let cap = storage.len();
            rehash_bucket_storage(storage, cap)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_robin_hood_strategy() {
        let strategy = RobinHoodStrategy::new(64, true, true);
        let config = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: true,
        };
        let mut context = RobinHoodContext::default();
        let mut buckets = vec![HashBucket::default(); 16];

        // Test insertion
        let result = strategy.insert(
            &mut context,
            &mut buckets,
            "key1".to_string(),
            "value1".to_string(),
            123456,
            &config,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());

        // Test lookup
        let found = strategy.get(&context, &buckets, "key1", 123456, &config);
        assert!(found.is_some());
        assert_eq!(found.unwrap(), "value1");
    }

    #[test]
    fn test_standard_storage_strategy() {
        let _strategy = StandardStorageStrategy;
        let config = StandardStorageConfig {
            initial_capacity: 16,
            growth_factor: 2.0,
        };

        let mut storage: <StandardStorageStrategy as StorageLayoutStrategy<String, i32>>::Storage =
            StandardStorageStrategy::create_storage(16, &config);
        assert_eq!(StandardStorageStrategy::capacity(&storage), 16);

        let buckets: &mut [HashBucket<String, i32>] =
            StandardStorageStrategy::get_buckets_mut(&mut storage);
        assert_eq!(buckets.len(), 16);
    }

    #[test]
    fn test_c7_4_robin_hood_tombstones_resize_rehash_and_variance() {
        let strategy = RobinHoodStrategy::new(64, true, false);
        let tombstone_config = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: true,
            backward_shift: false, // tombstone mode
        };
        let mut context = RobinHoodContext::default();

        // 1. Empty bucket slice must not panic with underflow
        let mut empty_buckets: Vec<HashBucket<u32, u32>> = Vec::new();
        assert!(strategy
            .insert(&mut context, &mut empty_buckets, 1, 10, 1, &tombstone_config)
            .is_err());
        assert_eq!(
            strategy.get(&context, &empty_buckets, &1, 1, &tombstone_config),
            None
        );
        assert_eq!(
            strategy.remove(&mut context, &mut empty_buckets, &1, 1, &tombstone_config),
            None
        );
        assert!(
            <RobinHoodStrategy as CollisionResolutionStrategy<u32, u32>>::needs_resize(
                &strategy, &context, 0, 0
            )
        );

        // 2. Tombstone mode: removing middle key in collision chain must not orphan later keys
        // or create duplicate keys on re-insertion
        let storage_cfg = StandardStorageConfig {
            initial_capacity: 8,
            growth_factor: 2.0,
        };
        let mut storage = StandardStorageStrategy::create_storage(8, &storage_cfg);
        {
            let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
            // Colliding keys at hash = 3 (slot 3, 4, 5)
            assert_eq!(
                strategy
                    .insert(&mut context, buckets, 10u32, 100u32, 3, &tombstone_config)
                    .unwrap(),
                None
            );
            assert_eq!(
                strategy
                    .insert(&mut context, buckets, 20u32, 200u32, 3, &tombstone_config)
                    .unwrap(),
                None
            );
            assert_eq!(
                strategy
                    .insert(&mut context, buckets, 30u32, 300u32, 3, &tombstone_config)
                    .unwrap(),
                None
            );

            // Probe distances are 0, 1, 2 -> non-zero variance
            let stats = <RobinHoodStrategy as CollisionResolutionStrategy<u32, u32>>::probe_stats(
                &strategy, &context,
            );
            assert!(
                stats.variance > 0.0,
                "expected positive probe distance variance, got {}",
                stats.variance
            );

            // Remove key 20 (leaves tombstone at slot 4)
            assert_eq!(
                strategy.remove(&mut context, buckets, &20u32, 3, &tombstone_config),
                Some(200)
            );
            // Key 30 at slot 5 must still be reachable past the tombstone!
            assert_eq!(
                strategy.get(&context, buckets, &30u32, 3, &tombstone_config),
                Some(&300)
            );
            // Updating key 30 must update slot 5 in-place and return Some(300), not duplicate at slot 4
            assert_eq!(
                strategy
                    .insert(&mut context, buckets, 30u32, 333u32, 3, &tombstone_config)
                    .unwrap(),
                Some(300)
            );
        }

        // 3. resize_storage must rehash existing entries so keys whose (hash & new_mask) changes remain findable
        {
            let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
            // hash = 9 lands in slot 1 (9 & 7 == 1) when capacity == 8,
            // but belongs in slot 9 (9 & 15 == 9) when capacity == 16!
            assert_eq!(
                strategy
                    .insert(&mut context, buckets, 99u32, 990u32, 9, &tombstone_config)
                    .unwrap(),
                None
            );
        }
        StandardStorageStrategy::resize_storage(&mut storage, 16, &storage_cfg).unwrap();
        {
            let buckets = StandardStorageStrategy::get_buckets(&storage);
            assert_eq!(
                strategy.get(&context, buckets, &99u32, 9, &tombstone_config),
                Some(&990),
                "entry with hash 9 lost after resize_storage(8 -> 16)"
            );
            assert_eq!(
                strategy.get(&context, buckets, &30u32, 3, &tombstone_config),
                Some(&333)
            );
        }

        // 4. Failed insert due to max_probe_distance must not lose displaced existing entries
        let tight_config = RobinHoodConfig {
            max_probe_distance: 1,
            variance_reduction: true,
            backward_shift: true,
        };
        let mut tight_ctx = RobinHoodContext::default();
        let mut tight_buckets = vec![HashBucket::default(); 8];
        // Slot 0: key 1 (hash 0, dist 0)
        // Slot 1: key 2 (hash 1, dist 0)
        // Slot 2: key 3 (hash 1, dist 1)
        strategy
            .insert(&mut tight_ctx, &mut tight_buckets, 1u32, 10u32, 0, &tight_config)
            .unwrap();
        strategy
            .insert(&mut tight_ctx, &mut tight_buckets, 2u32, 20u32, 1, &tight_config)
            .unwrap();
        strategy
            .insert(&mut tight_ctx, &mut tight_buckets, 3u32, 30u32, 1, &tight_config)
            .unwrap();
        // Inserting key 4 at hash 0 reaches slot 1 with dist 1 > slot 1's dist 0 (displacing key 2),
        // then key 2 reaches slot 2 with dist 1 == slot 2's dist 1, and reaches slot 3 with dist 2 > max_probe_distance (1).
        assert!(strategy
            .insert(&mut tight_ctx, &mut tight_buckets, 4u32, 40u32, 0, &tight_config)
            .is_err());
        // Existing keys 1, 2, 3 must all still be intact (previously key 2 was overwritten and lost)!
        assert_eq!(
            strategy.get(&tight_ctx, &tight_buckets, &1u32, 0, &tight_config),
            Some(&10)
        );
        assert_eq!(
            strategy.get(&tight_ctx, &tight_buckets, &2u32, 1, &tight_config),
            Some(&20)
        );
        assert_eq!(
            strategy.get(&tight_ctx, &tight_buckets, &3u32, 1, &tight_config),
            Some(&30)
        );
    }

    #[test]
    fn test_c7_4_update_beyond_max_probe_distance_and_full_table() {
        // Capacity 8. Keys 0..4 share home slot 0, so key k sits at slot k,
        // distance k.
        let strategy = RobinHoodStrategy::new(64, false, true);
        let wide = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: true,
        };
        let tight = RobinHoodConfig {
            max_probe_distance: 1,
            ..wide.clone()
        };
        let mut ctx = RobinHoodContext::default();
        let mut buckets: Vec<HashBucket<u32, u32>> = vec![HashBucket::default(); 8];
        for k in 0..4u32 {
            assert_eq!(
                strategy
                    .insert(&mut ctx, &mut buckets, k, k * 10, 0, &wide)
                    .unwrap(),
                None
            );
        }
        // Keys 1, 2 and 3 stepped past 1, 2 and 3 slots.
        assert_eq!((ctx.collision_count, ctx.eviction_count), (6, 0));
        let snapshot = |buckets: &[HashBucket<u32, u32>], ctx: &RobinHoodContext| {
            let slots: Vec<_> = buckets
                .iter()
                .map(|b| (b.hash, b.key, b.value, b.probe_distance, b.flags))
                .collect();
            (slots, format!("{ctx:?}"))
        };

        // Key 3 sits at distance 3, beyond the per-call bound of 1. An update
        // places nothing, so the bound does not apply to it.
        assert_eq!(
            strategy
                .insert(&mut ctx, &mut buckets, 3, 33, 0, &tight)
                .unwrap(),
            Some(30)
        );
        assert_eq!(strategy.get(&ctx, &buckets, &3, 0, &tight), Some(&33));

        // A new key with home 0 would sit at distance 4: rejected, and
        // neither the slice nor the statistics change.
        let before = snapshot(&buckets, &ctx);
        assert!(strategy
            .insert(&mut ctx, &mut buckets, 9, 90, 0, &tight)
            .is_err());
        assert_eq!(snapshot(&buckets, &ctx), before);

        // Fill slots 4..8 with keys whose home is their own slot.
        for k in 4..8u32 {
            assert_eq!(
                strategy
                    .insert(&mut ctx, &mut buckets, k, k * 10, u64::from(k), &wide)
                    .unwrap(),
                None
            );
        }
        // Updates still work in the full table...
        assert_eq!(
            strategy
                .insert(&mut ctx, &mut buckets, 7, 77, 7, &wide)
                .unwrap(),
            Some(70)
        );
        assert_eq!(
            strategy
                .insert(&mut ctx, &mut buckets, 2, 22, 0, &wide)
                .unwrap(),
            Some(20)
        );
        // Placements at home add no collisions, and updates are not counted.
        assert_eq!((ctx.collision_count, ctx.eviction_count), (6, 0));
        // ...but a new key has no slot: rejected, and nothing changes.
        let before = snapshot(&buckets, &ctx);
        assert!(strategy
            .insert(&mut ctx, &mut buckets, 8, 80, 0, &wide)
            .is_err());
        assert_eq!(snapshot(&buckets, &ctx), before);
        for (k, h, v) in [
            (0u32, 0u64, 0u32),
            (1, 0, 10),
            (2, 0, 22),
            (3, 0, 33),
            (4, 4, 40),
            (5, 5, 50),
            (6, 6, 60),
            (7, 7, 77),
        ] {
            assert_eq!(strategy.get(&ctx, &buckets, &k, h, &wide), Some(&v), "key {k}");
        }
        assert_eq!(strategy.get(&ctx, &buckets, &8, 0, &wide), None);
        check_robin_hood_layout(&buckets, 8);

        // Free slot 5. A new key with home 0 then steps past slots 0..=3,
        // displaces key 4 at slot 4, and carries it one slot on to slot 5:
        // 4 + 1 collisions and one displacement.
        assert_eq!(strategy.remove(&mut ctx, &mut buckets, &5, 5, &wide), Some(50));
        assert_eq!(
            strategy
                .insert(&mut ctx, &mut buckets, 9, 90, 0, &wide)
                .unwrap(),
            None
        );
        assert_eq!((ctx.collision_count, ctx.eviction_count), (11, 1));
        assert_eq!((buckets[4].key, buckets[4].probe_distance), (Some(9), 4));
        assert_eq!((buckets[5].key, buckets[5].probe_distance), (Some(4), 1));
        assert_eq!(strategy.get(&ctx, &buckets, &4, 4, &wide), Some(&40));
        assert_eq!(strategy.get(&ctx, &buckets, &9, 0, &wide), Some(&90));
        check_robin_hood_layout(&buckets, 8);
    }

    /// splitmix64: a small deterministic generator for the randomized tests.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Checks the Robin Hood layout invariants over live entries and
    /// tombstones: every live entry's stored distance is its real offset from
    /// its home slot, and along the table a slot's distance exceeds the
    /// previous slot's by at most one (and is 0 right after a vacant slot).
    fn check_robin_hood_layout<K, V>(buckets: &[HashBucket<K, V>], expected_live: usize) {
        let cap = buckets.len();
        assert!(cap.is_power_of_two(), "table length {cap} is not a power of two");
        let mask = cap - 1;
        let occupied = |b: &HashBucket<K, V>| b.key.is_some() || b.is_deleted();
        let mut live = 0;
        for (pos, bucket) in buckets.iter().enumerate() {
            if bucket.key.is_some() && !bucket.is_deleted() {
                live += 1;
                let home = (bucket.hash as usize) & mask;
                assert_eq!(
                    usize::from(bucket.probe_distance),
                    (pos + cap - home) & mask,
                    "slot {pos}: stored probe distance"
                );
            }
            let next_pos = (pos + 1) & mask;
            let next = &buckets[next_pos];
            if occupied(next) {
                let limit = if occupied(bucket) {
                    u32::from(bucket.probe_distance) + 1
                } else {
                    0
                };
                assert!(
                    u32::from(next.probe_distance) <= limit,
                    "slot {next_pos}: distance {} after limit {limit}",
                    next.probe_distance
                );
            }
        }
        assert_eq!(live, expected_live, "live entry count");
    }

    #[test]
    fn test_c7_4_tombstone_reuse_keeps_later_entries_reachable() {
        // Capacity 8, tombstone mode. Home 0: key 1 @0 d0, key 2 @1 d1,
        // key 5 @2 d2. Home 1: key 3 @3 d2.
        let strategy = RobinHoodStrategy::new(64, false, false);
        let config = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: false,
        };
        let mut ctx = RobinHoodContext::default();
        let mut buckets: Vec<HashBucket<u32, u32>> = vec![HashBucket::default(); 8];
        for (k, h) in [(1u32, 0u64), (2, 0), (5, 0), (3, 1)] {
            assert_eq!(
                strategy
                    .insert(&mut ctx, &mut buckets, k, k * 10, h, &config)
                    .unwrap(),
                None
            );
        }

        // Removing key 2 turns slot 1 into a tombstone; key 5 behind it must
        // stay reachable.
        assert_eq!(strategy.remove(&mut ctx, &mut buckets, &2, 0, &config), Some(20));
        assert_eq!(strategy.get(&ctx, &buckets, &5, 0, &config), Some(&50));

        // Key 4 (home 1) reaches the tombstone at distance 0, below the
        // tombstone's distance 1. Taking the slot there would make key 5
        // (home 0, distance 2) unreachable: a probe for it would stop at slot 1.
        assert_eq!(
            strategy
                .insert(&mut ctx, &mut buckets, 4, 40, 1, &config)
                .unwrap(),
            None
        );
        for (k, h) in [(1u32, 0u64), (5, 0), (3, 1), (4, 1)] {
            assert_eq!(
                strategy.get(&ctx, &buckets, &k, h, &config),
                Some(&(k * 10)),
                "key {k}"
            );
        }
        assert_eq!(strategy.get(&ctx, &buckets, &2, 0, &config), None);
        check_robin_hood_layout(&buckets, 4);
    }

    #[test]
    fn test_c7_4_backward_shift_moves_tombstones_and_later_entries() {
        // Capacity 8, all keys home 0: key 10 @0 d0, 11 @1 d1, 12 @2 d2, 13 @3 d3.
        let strategy = RobinHoodStrategy::new(64, false, true);
        let shift = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: true,
        };
        let tombstone = RobinHoodConfig {
            backward_shift: false,
            ..shift.clone()
        };
        let mut ctx = RobinHoodContext::default();
        let mut buckets: Vec<HashBucket<u32, u32>> = vec![HashBucket::default(); 8];
        for k in 10..14u32 {
            strategy
                .insert(&mut ctx, &mut buckets, k, k, 0, &shift)
                .unwrap();
        }

        // Tombstone at slot 2, then backward-shift removal of the entry at
        // slot 0: key 11, the tombstone and key 13 must each move back one slot.
        assert_eq!(strategy.remove(&mut ctx, &mut buckets, &12, 0, &tombstone), Some(12));
        assert_eq!(strategy.remove(&mut ctx, &mut buckets, &10, 0, &shift), Some(10));

        assert_eq!(strategy.get(&ctx, &buckets, &11, 0, &shift), Some(&11));
        assert_eq!(strategy.get(&ctx, &buckets, &13, 0, &shift), Some(&13));
        assert_eq!(strategy.get(&ctx, &buckets, &10, 0, &shift), None);
        assert_eq!(strategy.get(&ctx, &buckets, &12, 0, &shift), None);
        assert!(buckets[1].is_deleted() && buckets[1].probe_distance == 1);
        assert!(buckets[3].key.is_none() && !buckets[3].is_deleted());
        check_robin_hood_layout(&buckets, 2);
    }

    #[test]
    fn test_c7_4_probe_stats_match_live_layout() {
        // Displacements and both removal modes must keep the running sums
        // equal to the distances actually stored in the slice.
        let strategy = RobinHoodStrategy::new(64, false, true);
        let shift = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: true,
        };
        let tombstone = RobinHoodConfig {
            backward_shift: false,
            ..shift.clone()
        };
        let hash_of = |k: u32| u64::from(2 - k % 3);
        let mut ctx = RobinHoodContext::default();
        let mut buckets: Vec<HashBucket<u32, u32>> = vec![HashBucket::default(); 32];
        // Homes cycle 2, 1, 0, so later keys with earlier homes displace
        // earlier keys with later homes.
        for k in 0..12u32 {
            strategy
                .insert(&mut ctx, &mut buckets, k, k, hash_of(k), &shift)
                .unwrap();
        }
        assert!(ctx.eviction_count > 0, "key pattern must exercise displacement");
        for k in [1u32, 4, 7] {
            assert_eq!(
                strategy.remove(&mut ctx, &mut buckets, &k, hash_of(k), &tombstone),
                Some(k)
            );
        }
        for k in [0u32, 5] {
            assert_eq!(
                strategy.remove(&mut ctx, &mut buckets, &k, hash_of(k), &shift),
                Some(k)
            );
        }

        let live: Vec<f64> = buckets
            .iter()
            .filter(|b| b.key.is_some() && !b.is_deleted())
            .map(|b| f64::from(b.probe_distance))
            .collect();
        assert_eq!(live.len(), 7);
        let n = live.len() as f64;
        let mean = live.iter().sum::<f64>() / n;
        let variance = live.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;

        assert_eq!(ctx.total_probe_distance as f64, live.iter().sum::<f64>());
        let stats =
            <RobinHoodStrategy as CollisionResolutionStrategy<u32, u32>>::probe_stats(&strategy, &ctx);
        assert!(
            (stats.average_probe_distance - mean).abs() < 1e-9,
            "average {} vs layout mean {mean}",
            stats.average_probe_distance
        );
        assert!(
            (stats.variance - variance).abs() < 1e-9,
            "variance {} vs layout variance {variance}",
            stats.variance
        );
    }

    #[test]
    fn test_c7_4_create_and_resize_storage_use_power_of_two_lengths() {
        let cfg = StandardStorageConfig {
            initial_capacity: 0,
            growth_factor: 2.0,
        };
        let lens: Vec<usize> = [0usize, 1, 2, 3, 5, 17]
            .iter()
            .map(|&c| {
                let storage: FastVec<HashBucket<u32, u32>> =
                    StandardStorageStrategy::create_storage(c, &cfg);
                storage.len()
            })
            .collect();
        assert_eq!(lens, vec![0, 1, 2, 4, 8, 32]);

        let strategy = RobinHoodStrategy::new(64, false, true);
        let config = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: true,
        };
        let mut ctx = RobinHoodContext::default();
        let mut storage: FastVec<HashBucket<u32, u32>> =
            StandardStorageStrategy::create_storage(3, &cfg);
        for k in 0..3u32 {
            let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
            strategy
                .insert(&mut ctx, buckets, k, k, u64::from(k) * 7, &config)
                .unwrap();
        }
        StandardStorageStrategy::resize_storage(&mut storage, 5, &cfg).unwrap();
        assert_eq!(storage.len(), 8);
        for k in 0..3u32 {
            let buckets = StandardStorageStrategy::get_buckets(&storage);
            assert_eq!(
                strategy.get(&ctx, buckets, &k, u64::from(k) * 7, &config),
                Some(&k)
            );
        }
    }

    /// Drives `RobinHoodStrategy` over `StandardStorageStrategy` storage the
    /// way a map would (grow or purge tombstones before the table fills,
    /// sometimes shrink) and checks every result against
    /// `std::collections::HashMap`, with 40 keys forced onto 5 home slots.
    /// `mode`: 0 = backward shift only, 1 = tombstones only, 2 = random mix.
    fn run_robin_hood_differential(seed: u64, mode: u8, ops: usize, full_check_every: usize) {
        use std::collections::HashMap;

        const KEYS: u32 = 40;
        let hash_of = |k: u32| u64::from(k % 5) * 3;
        let strategy = RobinHoodStrategy::new(u16::MAX, false, true);
        let config = |backward_shift| RobinHoodConfig {
            max_probe_distance: u16::MAX,
            variance_reduction: false,
            backward_shift,
        };
        let storage_cfg = StandardStorageConfig {
            initial_capacity: 8,
            growth_factor: 2.0,
        };
        let mut storage: FastVec<HashBucket<u32, u64>> =
            StandardStorageStrategy::create_storage(8, &storage_cfg);
        let mut ctx = RobinHoodContext::default();
        let mut model: HashMap<u32, u64> = HashMap::new();
        let mut rng = seed;

        for step in 0..ops {
            let key = (splitmix64(&mut rng) % u64::from(KEYS)) as u32;
            let hash = hash_of(key);
            match splitmix64(&mut rng) % 10 {
                0..=4 => {
                    // Grow, or purge tombstones, before the table fills.
                    let cap = storage.len();
                    let used = storage
                        .as_slice()
                        .iter()
                        .filter(|b| b.key.is_some() || b.is_deleted())
                        .count();
                    if (used + 1) * 4 > cap * 3 {
                        let new_cap = if (model.len() + 1) * 2 > cap { cap * 2 } else { cap };
                        StandardStorageStrategy::resize_storage(&mut storage, new_cap, &storage_cfg)
                            .unwrap();
                    }
                    let value = splitmix64(&mut rng);
                    let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
                    let got = strategy
                        .insert(&mut ctx, buckets, key, value, hash, &config(true))
                        .unwrap();
                    assert_eq!(
                        got,
                        model.insert(key, value),
                        "seed {seed} mode {mode} step {step}: insert {key}"
                    );
                }
                5..=7 => {
                    let backward_shift = match mode {
                        0 => true,
                        1 => false,
                        _ => splitmix64(&mut rng) & 1 == 0,
                    };
                    let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
                    let got = strategy.remove(&mut ctx, buckets, &key, hash, &config(backward_shift));
                    assert_eq!(
                        got,
                        model.remove(&key),
                        "seed {seed} mode {mode} step {step}: remove {key}"
                    );
                }
                8 => {
                    // Shrink (or purge) to the smallest table with load <= 1/2.
                    let new_cap = (model.len() * 2).max(8);
                    StandardStorageStrategy::resize_storage(&mut storage, new_cap, &storage_cfg)
                        .unwrap();
                }
                _ => {
                    let buckets = StandardStorageStrategy::get_buckets(&storage);
                    assert_eq!(
                        strategy.get(&ctx, buckets, &key, hash, &config(true)),
                        model.get(&key),
                        "seed {seed} mode {mode} step {step}: get {key}"
                    );
                }
            }

            check_robin_hood_layout(StandardStorageStrategy::get_buckets(&storage), model.len());
            if step % full_check_every == 0 || step + 1 == ops {
                let buckets = StandardStorageStrategy::get_buckets(&storage);
                for k in 0..KEYS {
                    assert_eq!(
                        strategy.get(&ctx, buckets, &k, hash_of(k), &config(true)),
                        model.get(&k),
                        "seed {seed} mode {mode} step {step}: get {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_c7_4_robin_hood_matches_std_hashmap_under_collisions() {
        let (seeds, ops, full_check_every) = if cfg!(miri) { (1u64, 150, 25) } else { (4, 2000, 1) };
        for mode in 0..3u8 {
            for seed in 0..seeds {
                run_robin_hood_differential(seed * 3 + u64::from(mode) + 1, mode, ops, full_check_every);
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // 2^18 buckets: too slow under Miri
    fn test_c7_4_rehash_rejects_u16_probe_distance_overflow() {
        // 65,537 entries whose hashes share their low 17 bits. In a 2^18 table
        // they split between two home slots (32,769 and 32,768 entries). In a
        // 2^17 table they all share home slot 0, so the last one would sit at
        // distance 65,536, beyond what `probe_distance: u16` can hold.
        const N: usize = 65_537;
        let cfg = StandardStorageConfig {
            initial_capacity: 0,
            growth_factor: 2.0,
        };
        let mut storage: FastVec<HashBucket<u32, u32>> =
            StandardStorageStrategy::create_storage(1 << 18, &cfg);
        assert_eq!(storage.len(), 1 << 18);
        {
            let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
            for i in 0..N {
                let distance = i / 2;
                let bucket = &mut buckets[(i % 2) * (1 << 17) + distance];
                bucket.hash = (i as u64) << 17;
                bucket.key = Some(i as u32);
                bucket.value = Some(i as u32);
                bucket.probe_distance = distance as u16;
            }
        }
        let strategy = RobinHoodStrategy::new(u16::MAX, false, true);
        let config = RobinHoodConfig {
            max_probe_distance: u16::MAX,
            variance_reduction: false,
            backward_shift: true,
        };
        let ctx = RobinHoodContext::default();
        let spot_check = |storage: &FastVec<HashBucket<u32, u32>>| {
            for i in [0usize, 1, N - 2, N - 1] {
                let buckets = StandardStorageStrategy::get_buckets(storage);
                assert_eq!(
                    strategy.get(&ctx, buckets, &(i as u32), (i as u64) << 17, &config),
                    Some(&(i as u32)),
                    "key {i}"
                );
            }
        };
        spot_check(&storage);

        assert!(
            StandardStorageStrategy::resize_storage(&mut storage, 1 << 17, &cfg).is_err(),
            "a rehash that needs a probe distance above u16::MAX must fail"
        );
        assert_eq!(storage.len(), 1 << 18, "a failed rehash must leave the table unchanged");
        spot_check(&storage);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // 2^18 buckets: too slow under Miri
    fn test_c7_4_rehash_over_65536_entries_keeps_every_entry() {
        // 70,000 entries, one per home slot of a 2^17 table. Odd keys have hash
        // bit 17 set, so doubling the table moves them to a new home slot.
        const N: u32 = 70_000;
        let hash_of = |k: u32| u64::from(k) | (u64::from(k & 1) << 17);
        let cfg = StandardStorageConfig {
            initial_capacity: 0,
            growth_factor: 2.0,
        };
        let mut storage: FastVec<HashBucket<u32, u32>> =
            StandardStorageStrategy::create_storage(1 << 17, &cfg);
        {
            let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
            for k in 0..N {
                let bucket = &mut buckets[k as usize];
                bucket.hash = hash_of(k);
                bucket.key = Some(k);
                bucket.value = Some(k);
            }
        }
        StandardStorageStrategy::resize_storage(&mut storage, 1 << 18, &cfg).unwrap();
        assert_eq!(storage.len(), 1 << 18);
        check_robin_hood_layout(StandardStorageStrategy::get_buckets(&storage), N as usize);

        let strategy = RobinHoodStrategy::new(u16::MAX, false, true);
        let config = RobinHoodConfig {
            max_probe_distance: u16::MAX,
            variance_reduction: false,
            backward_shift: true,
        };
        let ctx = RobinHoodContext::default();
        let buckets = StandardStorageStrategy::get_buckets(&storage);
        for k in 0..N {
            assert_eq!(strategy.get(&ctx, buckets, &k, hash_of(k), &config), Some(&k), "key {k}");
        }
    }

    #[test]
    fn test_max_linear_probe_distance_matches_robin_hood_simulation() {
        let mut rng = 0x1234_5678_u64;
        let rounds = if cfg!(miri) { 60 } else { 3000 };
        for _ in 0..rounds {
            let cap = 1usize << (splitmix64(&mut rng) % 6);
            let n = (splitmix64(&mut rng) as usize) % (cap + 1);
            // Cluster the homes into a window that may wrap around the end.
            let spread = 1 + (splitmix64(&mut rng) as usize) % cap;
            let base = (splitmix64(&mut rng) as usize) % cap;
            let homes: Vec<usize> = (0..n)
                .map(|_| (base + (splitmix64(&mut rng) as usize) % spread) % cap)
                .collect();

            // Robin Hood insertion, tracking only each slot's probe distance.
            let mut slots: Vec<Option<usize>> = vec![None; cap];
            for &home in &homes {
                let mut pos = home;
                let mut distance = 0usize;
                loop {
                    match slots[pos] {
                        None => {
                            slots[pos] = Some(distance);
                            break;
                        }
                        Some(occupant) if distance > occupant => {
                            slots[pos] = Some(distance);
                            distance = occupant;
                        }
                        Some(_) => {}
                    }
                    pos = (pos + 1) % cap;
                    distance += 1;
                }
            }
            let expected = slots.iter().flatten().copied().max().unwrap_or(0);

            let mut counts = vec![0usize; cap];
            for &home in &homes {
                counts[home] += 1;
            }
            assert_eq!(
                max_linear_probe_distance(&counts),
                expected,
                "cap {cap}, homes {homes:?}"
            );
        }
    }

    #[test]
    fn test_c7_4_optimize_layout_drops_tombstones() {
        // `optimize_layout` was a no-op, so only `resize_storage` dropped
        // tombstones. It now rehashes at the current size when there are any.
        let cfg = StandardStorageConfig {
            initial_capacity: 8,
            growth_factor: 2.0,
        };
        let strategy = RobinHoodStrategy::new(64, false, false);
        let tombstone = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: false,
        };
        let mut ctx = RobinHoodContext::default();
        let mut storage: FastVec<HashBucket<u32, u32>> =
            StandardStorageStrategy::create_storage(8, &cfg);
        {
            // Keys 0..4 share home slot 0; removing key 1 leaves a tombstone
            // at slot 1, in front of keys 2 and 3.
            let buckets = StandardStorageStrategy::get_buckets_mut(&mut storage);
            for k in 0..4u32 {
                strategy
                    .insert(&mut ctx, buckets, k, k * 10, 0, &tombstone)
                    .unwrap();
            }
            assert_eq!(strategy.remove(&mut ctx, buckets, &1, 0, &tombstone), Some(10));
            assert!(buckets[1].is_deleted());
        }

        StandardStorageStrategy::optimize_layout(&mut storage, &cfg).unwrap();
        assert_eq!(storage.len(), 8);
        let buckets = StandardStorageStrategy::get_buckets(&storage);
        assert!(
            !buckets.iter().any(|b| b.is_deleted()),
            "optimize_layout must drop tombstones"
        );
        check_robin_hood_layout(buckets, 3);
        for k in [0u32, 2, 3] {
            assert_eq!(
                strategy.get(&ctx, buckets, &k, 0, &tombstone),
                Some(&(k * 10)),
                "key {k}"
            );
        }
        assert_eq!(strategy.get(&ctx, buckets, &1, 0, &tombstone), None);

        // Without tombstones the layout is left as it is.
        let before: Vec<_> = buckets.iter().map(|b| (b.key, b.probe_distance)).collect();
        StandardStorageStrategy::optimize_layout(&mut storage, &cfg).unwrap();
        let after: Vec<_> = StandardStorageStrategy::get_buckets(&storage)
            .iter()
            .map(|b| (b.key, b.probe_distance))
            .collect();
        assert_eq!(after, before);
    }

    #[test]
    fn test_c7_4_non_power_of_two_slices_are_rejected() {
        // Probing masks positions with `len - 1`, which reaches every slot
        // only when `len` is a power of two: a length-5 slice was probed with
        // mask 4, i.e. only slots 0 and 4.
        let strategy = RobinHoodStrategy::new(64, false, true);
        let config = RobinHoodConfig {
            max_probe_distance: 64,
            variance_reduction: false,
            backward_shift: true,
        };
        let mut ctx = RobinHoodContext::default();
        let mut buckets: Vec<HashBucket<u32, u32>> = vec![HashBucket::default(); 5];
        assert!(
            strategy
                .insert(&mut ctx, &mut buckets, 1, 10, 1, &config)
                .is_err(),
            "insert into a length-5 slice must fail"
        );
        assert!(buckets.iter().all(|b| b.key.is_none() && !b.is_deleted()));
        assert_eq!(strategy.get(&ctx, &buckets, &1, 1, &config), None);
        assert_eq!(strategy.remove(&mut ctx, &mut buckets, &1, 1, &config), None);
    }
}
