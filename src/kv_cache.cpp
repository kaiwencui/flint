#include "kv_cache.hpp"

#include <algorithm>

namespace flint {

RadixKvManager::RadixKvManager(u32 total_blocks) {
    if (total_blocks > 0) init(total_blocks);
}

void RadixKvManager::init(u32 total_blocks) {
    total_blocks_ = total_blocks;
    free_list_.clear();
    free_list_.reserve(total_blocks);
    for (u32 i = total_blocks; i > 0; --i) {
        free_list_.push_back(static_cast<i32>(i - 1));
    }
    hash_to_node_.clear();
    lru_leaves_.clear();
}

void RadixKvManager::clear() {
    init(total_blocks_);
}

u64 RadixKvManager::compute_block_hash(u64 parent_hash, std::span<const i32> block_tokens) noexcept {
    return hash_bytes64(block_tokens.data(), block_tokens.size() * sizeof(i32),
                        parent_hash ^ 0xA0761D6478BD642FULL);
}

void RadixKvManager::touch_lru(RadixNode& node) {
    if (node.in_lru) {
        lru_leaves_.erase(node.lru_it);
    }
    lru_leaves_.push_front(node.hash);
    node.lru_it = lru_leaves_.begin();
    node.in_lru = true;
}

void RadixKvManager::remove_from_lru(RadixNode& node) {
    if (node.in_lru) {
        lru_leaves_.erase(node.lru_it);
        node.in_lru = false;
    }
}

i32 RadixKvManager::evict_one_leaf() {
    while (!lru_leaves_.empty()) {
        u64 h = lru_leaves_.back();
        lru_leaves_.pop_back();
        auto it = hash_to_node_.find(h);
        if (it == hash_to_node_.end()) continue;
        it->second.in_lru = false;
        if (it->second.ref_count > 0 || it->second.child_count > 0) continue;

        i32 phys = it->second.phys_block;
        u64 parent_h = it->second.parent_hash;
        hash_to_node_.erase(it);

        if (parent_h != 0) {
            auto pit = hash_to_node_.find(parent_h);
            if (pit != hash_to_node_.end() && pit->second.child_count > 0) {
                pit->second.child_count--;
                if (pit->second.child_count == 0 && pit->second.ref_count == 0) {
                    touch_lru(pit->second);
                }
            }
        }
        return phys;
    }
    return -1;
}

i32 RadixKvManager::allocate_block() {
    if (!free_list_.empty()) {
        i32 b = free_list_.back();
        free_list_.pop_back();
        return b;
    }
    return evict_one_leaf();
}

RadixKvManager::PrefixMatch RadixKvManager::match_prefix(std::span<const i32> tokens) {
    PrefixMatch res;
    // Leave at least 1 token uncached so prefill always produces final logits cleanly
    const size_t max_matchable_tokens = (tokens.size() > 1) ? (tokens.size() - 1) : 0;
    const size_t full_blocks = max_matchable_tokens / kKvBlockSize;

    u64 parent_h = 0;
    for (size_t b = 0; b < full_blocks; ++b) {
        auto slice = tokens.subspan(b * kKvBlockSize, kKvBlockSize);
        u64 h = compute_block_hash(parent_h, slice);
        auto it = hash_to_node_.find(h);
        if (it == hash_to_node_.end()) break;

        RadixNode& node = it->second;
        node.ref_count++;
        remove_from_lru(node);
        res.block_table.push_back(node.phys_block);
        res.matched_tokens += kKvBlockSize;
        res.last_block_hash = h;
        parent_h = h;
    }
    return res;
}

u64 RadixKvManager::insert_block(u64 parent_hash, std::span<const i32> block_tokens, i32 phys_block) {
    u64 h = compute_block_hash(parent_hash, block_tokens);
    auto it = hash_to_node_.find(h);
    if (it != hash_to_node_.end()) {
        // Already cached under another block; keep existing cached node and return free block if distinct
        if (it->second.phys_block != phys_block) {
            free_list_.push_back(phys_block);
        }
        it->second.ref_count++;
        remove_from_lru(it->second);
        return h;
    }

    if (parent_hash != 0) {
        auto pit = hash_to_node_.find(parent_hash);
        if (pit != hash_to_node_.end()) {
            pit->second.child_count++;
            remove_from_lru(pit->second);
        }
    }

    RadixNode node;
    node.hash = h;
    node.parent_hash = parent_hash;
    node.phys_block = phys_block;
    node.ref_count = 1;
    node.child_count = 0;
    node.in_lru = false;
    hash_to_node_.emplace(h, std::move(node));
    return h;
}

void RadixKvManager::release_blocks(std::span<const i32> block_table, u32 committed_full_blocks) {
    // Return uncommitted tail block(s) directly to free list
    for (size_t i = committed_full_blocks; i < block_table.size(); ++i) {
        if (block_table[i] >= 0) {
            free_list_.push_back(block_table[i]);
        }
    }
    // Decrement ref_count on committed nodes (in reverse so leaves become evictable properly)
    for (auto& [h, node] : hash_to_node_) {
        for (size_t i = 0; i < committed_full_blocks && i < block_table.size(); ++i) {
            if (node.phys_block == block_table[i] && node.ref_count > 0) {
                node.ref_count--;
                if (node.ref_count == 0 && node.child_count == 0) {
                    touch_lru(node);
                }
                break;
            }
        }
    }
}

} // namespace flint
