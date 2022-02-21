
## APIs
```c++
template <typename Key,
          typename Value,
          int B              = 16,
          typename Allocator = device_bump_allocator<node_type<Key, Value, B>>>
struct gpu_versioned_btree;
```
### Member functions
```c++
/* Device-side APIs */
//Constructor
gpu_versioned_btree();

// Insert num_keys pairs {keys, values} into the data structure on a given stream.
// Use in_place to chose between in-place and out-of-place insertions
void insert(const Key* keys, const Value* values, const size_type num_keys,
            cudaStream_t stream = 0, bool in_place = true);

// Find num_keys keys from the data structure on a given stream.
// Stores the result into values. Set the concurrent flag to true if concurrent updates are running
void find(const Key* keys, Value* values, const size_type num_keys,
          cudaStream_t stream = 0, bool concurrent = false);

// Looks up the keys in the given timestamp (other arguments are same as above find)
void find(const Key* keys, Value* values, const size_type num_keys,
          size_type timestamp, cudaStream_t stream = 0, bool concurrent = false);

// Erases num_keys keys from the data structure on a given stream.
// Set the concurrent flag to true if concurrent updates are running.
void erase(const Key* keys, const size_type num_keys, cudaStream_t stream = 0, bool concurrent = false);

// Performs num_keys range queries defined by [lower_bound, upper_bound) and the average_range_length on a given stream.
// Stores the RQ results result and the counts into counts. Set counts to nullptr to only compute the RQ.
// Set the concurrent flag to true if concurrent updates are running.
void range_query(const Key* lower_bound, const Key* upper_bound, pair_type* result, size_type* counts,
                 const size_type average_range_length, const size_type num_keys, cudaStream_t stream = 0,
                 bool concurrent = false);

// Performs the range query on a given snapshot defined by timestamp. Other arguments are same as above.
void range_query(const Key* lower_bound, const Key* upper_bound, pair_type* result, size_type* counts,
                 const size_type average_range_length, const size_type num_keys, size_type timestamp,
                 cudaStream_t stream = 0, bool concurrent = false);

// Performs concurrent find and erase on the given stream. Find operations are defined by [find_keys, num_finds].
// Erase operations are defined by [erase_keys, num_erasures]. Stores the find results into find_results.
void concurrent_find_erase(const Key* find_keys, Value* find_results, const size_type num_finds,
                           const Key* erase_keys, const size_type num_erasures, cudaStream_t stream = 0);

// Performs concurrent insertions and range queries on a given stream.
// Insertions defined by keys, values, and num_insertion.
// RQs defined by lower_bound, upper_bound, average_range_length, and num_ranges.
// Stores the RQ results into result
void concurrent_insert_range(const Key* keys, const Value* values, const size_type num_insertion,
                            const Key* lower_bound, const Key* upper_bound, const size_type num_ranges,
                            pair_type* result, const size_type average_range_length,cudaStream_t stream = 0);
// Takes a snapshot from on the give stream. Returns the snapshot handle.
size_type take_snapshot(const cudaStream_t stream = 0);

// Device-side APIs
// Uses a tile to cooperatively insert the pair {key, value} into the tree in-place.
// Uses the device allocator to allocate data structure nodes.
template <typename tile_type, typename DeviceAllocator>
bool cooperative_insert_in_place(const Key& key, const Value& value, const tile_type& tile, DeviceAllocator& allocator);

// Cooperatively ouf-of-place insert the pair {key, value} into the tree.
// Requires a memory reclaimer to reclaim retired tree nodes.
template <typename tile_type, typename DeviceAllocator, typename DeviceReclaimer>
bool cooperative_insert(const Key& key, const Value& value, const tile_type& tile, DeviceAllocator& allocator,
                                    DeviceReclaimer& reclaimer);

// Cooperatively takes a snapshot of the data structure using the given tile.
// Returns the snapshot handle.
template <typename tile_type>
size_type take_snapshot(const tile_type& tile);

// Cooperatively (using the input tile) finds a key in the data structure.
// Uses the allocator to find address of internal pointers.
// Set the concurrent flag if concurrent operations are running.
template <typename tile_type, typename DeviceAllocator>
Value cooperative_find(const Key& key, const tile_type& tile, DeviceAllocator& allocator, bool concurrent = false);

// Find overload the assume concurrent operations
template <typename tile_type, typename DeviceAllocator>
Value concurrent_cooperative_find(const Key& key, const tile_type& tile, DeviceAllocator& allocator, const size_type& timestamp);

// Cooperatively finds a key in the data structure at the given snapshot ID.
// Other arguments are the same as above API.
template <typename tile_type, typename DeviceAllocator>
Value cooperative_find(const Key& key, const tile_type& tile, DeviceAllocator& allocator, const size_type& timestamp,
                      bool concurrent = false)

// Cooperatively (using the input tile) erase a key from the data structure.
// Uses the allocator to find address of internal pointers.
// Set the concurrent flag if concurrent operations are running.
template <typename tile_type, typename DeviceAllocator>
bool cooperative_erase(const Key& key, const tile_type& tile, DeviceAllocator& allocator, bool concurrent = false);

// Cooperatively (using the input tile) erase a key from the data structure using out-of-place technique.
// Requires a reclaimer to reclaim old tree nodes. Other arguments are the same as above API.
template <typename tile_type, typename DeviceAllocator, typename DeviceReclaimer>
bool cooperative_erase(const Key& key, const tile_type& tile, DeviceAllocator& allocator, DeviceReclaimer& reclaimer);

// Cooperatively performs a range query defined by [lower_bound, upper_bound) using the given timestamp.
// Requires the allocator used in building the tree.
// Stores the result into the buffer and returns the count.
template <typename tile_type, typename DeviceAllocator>
size_type concurrent_cooperative_range_query(const Key& lower_bound, const Key& upper_bound, const tile_type& tile,
                                            DeviceAllocator& allocator, const size_type& timestamp, pair_type* buffer = nullptr);
```

