#pragma once

#include "common.hpp"

#include <list>
#include <span>
#include <unordered_map>
#include <vector>

namespace flint {

constexpr u32 kKvBlockSize = 16; // 16 tokens per paged KV block

// Paged VRAM block allocator + Radix Prefix Tree with LRU eviction.
// Tracks logical block chains mapped to physical block IDs in VRAM.
// Prefix hits reuse cached KV blocks in O(prefix_blocks) without recomputing attention
class RadixKvManager {
public:
    explicit RadixKvManager(u32 total_blocks = 0);

    void init(u32 total_blocks);
    void clear();

    u32 total_blocks() const noexcept { return total_blocks_; }
    u32 free_blocks()  const noexcept { return static_cast<u32>(free_list_.size()); }
    u32 cached_blocks() const noexcept { return static_cast<u32>(hash_to_node_.size()); }

    struct PrefixMatch {
        u32 matched_tokens = 0;         // Multiple of kKvBlockSize (except if exact full match)
        std::vector<i32> block_table;   // Physical block IDs for matched prefix blocks
        u64 last_block_hash = 0;        // Chained hash of last matched full block
    };

    // Match longest cached prefix of `tokens` aligned to kKvBlockSize blocks.
    // Pins matched blocks so LRU eviction will not reclaim them during the active request
    PrefixMatch match_prefix(std::span<const i32> tokens);

    // Allocate a physical block (from free list or LRU-evicting an unpinned cached leaf).
    i32 allocate_block();

    // Commit a completed full block of `kKvBlockSize` tokens into the radix prefix cache.
    // Returns the chained hash of this block (pass as `parent_hash` for the next block).
    u64 insert_block(u64 parent_hash, std::span<const i32> block_tokens, i32 phys_block);

    // Unpin all blocks referenced by a completed request (they remain cached & LRU-evictable).
    void release_blocks(std::span<const i32> block_table, u32 committed_full_blocks);

    static u64 compute_block_hash(u64 parent_hash, std::span<const i32> block_tokens) noexcept;

private:
    struct RadixNode {
        u64 hash = 0;
        u64 parent_hash = 0;
        i32 phys_block = -1;
        u32 ref_count = 0;   // Active request pins
        u32 child_count = 0; // Only leaf nodes (child_count == 0 && ref_count == 0) are evictable
        std::list<u64>::iterator lru_it;
        bool in_lru = false;
    };

    void touch_lru(RadixNode& node);
    void remove_from_lru(RadixNode& node);
    i32 evict_one_leaf();

    u32 total_blocks_ = 0;
    std::vector<i32> free_list_;
    std::unordered_map<u64, RadixNode> hash_to_node_;
    std::list<u64> lru_leaves_; // Front = MRU, Back = LRU evict candidate
};

} // namespace flint
