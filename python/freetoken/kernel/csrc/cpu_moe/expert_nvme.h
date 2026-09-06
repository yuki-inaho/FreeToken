// Tier 3: stream MoE expert weights from an FTW checkpoint on NVMe into the pinned host pool.
//
// Why this lives in C++ and not in the Python engine: `decode_submit` only stages an *async*
// D2H of the routing pack and rings a doorbell, so the routed expert ids are not valid
// host-side until the GPU reaches that doorbell. Resolving in Python would mean synchronising
// there and destroying the compute/copy overlap the hybrid backend exists for. The coordinator
// thread, which already wakes on the doorbell and already blocks the GPU on `done[slot]`, is
// the one place a disk stall can legally live: it just makes an existing wait longer.
//
// The residency policy is a straight port of freetoken.tier3.host_pool.HostExpertPool (LRU,
// free slots first, deterministic lowest-slot tie-break) and the reader of
// freetoken.tier3.nvme_store.NvmeExpertStore (aligned O_DIRECT preadv straight into the pinned
// slot -- no bounce buffer, because FTW entry offsets and the packed banks' row bytes are both
// whole 4096-byte blocks). Those Python classes are the executable reference this must match.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace ft_tier3 {

constexpr int kAlign = 4096;
constexpr int32_t kNotResident = -1;

struct Shard {
  int64_t global_off = 0;
  int64_t nbytes = 0;
  std::string path;
};

// One pending read: `nbytes` from `global_off` of the FTW byte region into `dst`.
struct ReadJob {
  int64_t global_off = 0;
  int64_t nbytes = 0;
  uint8_t* dst = nullptr;
};

class ExpertTier3 {
 public:
  // -- geometry, all set once from Python ------------------------------------------------
  int num_layers = 0;
  int num_experts = 0;
  int host_slots = 0;
  int num_banks = 0;
  std::vector<int64_t> feat_bytes;          // [num_banks] per-expert row bytes
  std::vector<const uint64_t*> dst_tbl;     // [num_banks] per-layer host bank base tables
  std::vector<int64_t> entry_off;           // [num_banks * num_layers] FTW global offsets
  std::vector<Shard> shards;

  // -- residency (host_slot_tbl is shared with CpuMoeExecutor::row_of) --------------------
  int32_t* host_slot_tbl = nullptr;         // [num_layers * num_experts], -1 = on NVMe
  std::vector<int32_t> id_of_slot;          // [num_layers * host_slots]
  std::vector<int64_t> usage;               // [num_layers * host_slots] LRU stamps
  std::vector<int64_t> clock;               // [num_layers]
  std::vector<std::vector<int>> free_slots; // [num_layers]

  // -- counters (read back through the executor for [moe-stats]) -------------------------
  std::atomic<int64_t> n_hit{0};
  std::atomic<int64_t> n_fetch{0};
  std::atomic<int64_t> n_bytes{0};
  std::atomic<int64_t> n_stall_ns{0};
  std::atomic<int64_t> n_calls{0};
  // Steps where the pool had no evictable slot left, so an expert could not be made resident
  // and its contribution was DROPPED. This is silent numerical corruption, not a slowdown, and
  // it means the pool is undersized for the step's working set -- the planner's min_host_slots
  // floor (>= the per-step distinct-expert count) exists to make it unreachable. Surfaced so a
  // misconfiguration is observable instead of quietly wrong.
  std::atomic<int64_t> n_exhausted{0};

  ~ExpertTier3() { shutdown(); }

  void init(int layers, int experts, int slots, int readers) {
    num_layers = layers;
    num_experts = experts;
    host_slots = slots;
    id_of_slot.assign((size_t)layers * slots, kNotResident);
    usage.assign((size_t)layers * slots, 0);
    clock.assign(layers, 0);
    free_slots.assign(layers, {});
    for (int l = 0; l < layers; ++l) {
      free_slots[l].reserve(slots);
      for (int s = slots - 1; s >= 0; --s) free_slots[l].push_back(s);  // pop_back -> ascending
    }
    start_readers(readers);
  }

  // Ensure every routed expert of this task is resident in the host pool, reading the misses
  // from NVMe. `excl` names host slots currently under a live UVA gather by the GPU (routing
  // pack row 2) -- they must never be recycled here.
  void resolve(int layer_id, const int32_t* ids, int n_ids, const int32_t* excl, int n_excl) {
    if (host_slot_tbl == nullptr || layer_id < 0 || layer_id >= num_layers) return;
    n_calls.fetch_add(1, std::memory_order_relaxed);
    const int64_t t0 = now_ns();
    const int64_t now = ++clock[layer_id];

    claimed_.clear();
    pending_.clear();
    // Alternate job buffers by generation. A reader that is slow to leave the previous batch
    // can still be inside work() when the next resolve() starts building, and would otherwise
    // index a vector being push_back'd into (reallocation -> dangling read). With two buffers
    // the batch a stale reader can touch is never the one being written.
    build_ = &jobs_buf_[(gen_ + 1) & 1];
    build_->clear();
    for (int i = 0; i < n_excl; ++i) {
      if (excl[i] >= 0) claimed_.push_back(excl[i]);
    }
    const size_t n_locked = claimed_.size();  // exclusions are never publishable, only blocking

    // Pass 1: every RESIDENT routed expert, stamped and claimed before any victim is chosen.
    // Interleaving the two passes made the victim search depend on position in `ids`: a miss at
    // index i could evict the slot of a hit at index j > i, whose stamp was still its old one.
    // That slot is exactly what the GPU may be gathering from over UVA right now -- the routing
    // pack's exclusion list (row 2) was meant to cover this but is never populated, so
    // `n_excl` is always 0 here. Claiming all hits up front closes the window with the
    // information the resolver already has, and incidentally stops the redundant re-read the
    // evicted hit caused when the loop reached it.
    for (int i = 0; i < n_ids; ++i) {
      const int32_t e = ids[i];
      if (e < 0 || e >= num_experts) continue;
      const int32_t slot = host_slot_tbl[(size_t)layer_id * num_experts + e];
      if (slot < 0) continue;
      n_hit.fetch_add(1, std::memory_order_relaxed);
      usage[(size_t)layer_id * host_slots + slot] = now;
      if (!contains(claimed_, slot)) claimed_.push_back(slot);
    }

    // Pass 2: page in whatever is left.
    for (int i = 0; i < n_ids; ++i) {
      const int32_t e = ids[i];
      if (e < 0 || e >= num_experts) continue;
      const size_t key = (size_t)layer_id * num_experts + e;
      if (host_slot_tbl[key] >= 0) continue;  // counted as a hit in pass 1
      if (already_pending(e)) continue;  // the same expert twice in one step: fetch once
      const int32_t victim = claim(layer_id, now);
      if (victim < 0) {                  // see n_exhausted: undersized pool, results are wrong
        n_exhausted.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      n_fetch.fetch_add(1, std::memory_order_relaxed);
      // Publish-clear BEFORE any bytes move: a concurrent reader then sees "not resident" for
      // both experts, which is conservative and safe, never a half-written slot.
      const int32_t old = id_of_slot[(size_t)layer_id * host_slots + victim];
      if (old >= 0) store_release(&host_slot_tbl[(size_t)layer_id * num_experts + old],
                                  kNotResident);
      id_of_slot[(size_t)layer_id * host_slots + victim] = kNotResident;
      usage[(size_t)layer_id * host_slots + victim] = now;
      claimed_.push_back(victim);
      pending_.push_back({e, victim});
      for (int b = 0; b < num_banks; ++b) {
        ReadJob j;
        j.global_off = entry_off[(size_t)b * num_layers + layer_id] +
                       (int64_t)e * feat_bytes[b];
        j.nbytes = feat_bytes[b];
        j.dst = bank_row(b, layer_id, victim);
        build_->push_back(j);
      }
    }
    (void)n_locked;

    if (!build_->empty()) {
      run_jobs();
      // Publish-set only after every byte has landed.
      for (const auto& p : pending_) {
        id_of_slot[(size_t)layer_id * host_slots + p.slot] = p.expert;
        store_release(&host_slot_tbl[(size_t)layer_id * num_experts + p.expert], p.slot);
      }
    }
    n_stall_ns.fetch_add(now_ns() - t0, std::memory_order_relaxed);
  }

  // Publish a prewarmed identity block: slots 0..count-1 of `layer_id` hold experts
  // 0..count-1, already on disk-read into the pool by the loader. Without this the C++ side
  // would still believe those slots are free, hand them out, and overwrite live rows while the
  // residency map still pointed at them.
  void prewarm_identity(int layer_id, int count) {
    if (host_slot_tbl == nullptr || layer_id < 0 || layer_id >= num_layers) return;
    count = std::min(count, host_slots);
    auto& fl = free_slots[layer_id];
    fl.clear();
    for (int s = host_slots - 1; s >= count; --s) fl.push_back(s);
    for (int s = 0; s < count; ++s) {
      id_of_slot[(size_t)layer_id * host_slots + s] = s;
      usage[(size_t)layer_id * host_slots + s] = 0;
      store_release(&host_slot_tbl[(size_t)layer_id * num_experts + s], s);
    }
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& th : readers_) {
      if (th.joinable()) th.join();
    }
    readers_.clear();
    std::lock_guard<std::mutex> lk(fd_mtx_);
    for (auto& kv : fds_) {
      if (kv.second >= 0) ::close(kv.second);
    }
    fds_.clear();
  }

 private:
  struct Pending {
    int32_t expert;
    int32_t slot;
  };

  static int64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
  }

  static void store_release(int32_t* p, int32_t v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
  }

  static bool contains(const std::vector<int32_t>& v, int32_t x) {
    for (int32_t e : v) {
      if (e == x) return true;
    }
    return false;
  }

  bool already_pending(int32_t e) const {
    for (const auto& p : pending_) {
      if (p.expert == e) return true;
    }
    return false;
  }

  uint8_t* bank_row(int bank, int layer_id, int slot) const {
    auto base = reinterpret_cast<uint8_t*>(
        static_cast<uintptr_t>(dst_tbl[bank][layer_id]));
    return base + (int64_t)slot * feat_bytes[bank];
  }

  // Free slots first, then the least-recently-used slot that is neither excluded nor already
  // claimed this call. Ties break to the lowest slot index -- the Python reference does the
  // same, and the two are diffed in tests.
  int32_t claim(int layer_id, int64_t now) {
    auto& fl = free_slots[layer_id];
    while (!fl.empty()) {
      const int32_t s = fl.back();
      fl.pop_back();
      if (!contains(claimed_, s)) {
        usage[(size_t)layer_id * host_slots + s] = now;
        return s;
      }
    }
    int32_t best = -1;
    int64_t best_ts = 0;
    const size_t base = (size_t)layer_id * host_slots;
    for (int s = 0; s < host_slots; ++s) {
      if (contains(claimed_, s)) continue;
      const int64_t ts = usage[base + s];
      if (best < 0 || ts < best_ts) {
        best = s;
        best_ts = ts;
      }
    }
    return best;
  }

  // -- the reader pool -------------------------------------------------------------------

  void start_readers(int n) {
    if (n < 1) n = 1;
    stop_ = false;
    for (int i = 0; i < n; ++i) readers_.emplace_back([this] { reader_loop(); });
  }

  void run_jobs() {
    const int total = (int)build_->size();
    {
      std::lock_guard<std::mutex> lk(mtx_);
      done_jobs_.store(0, std::memory_order_relaxed);
      total_.store(total, std::memory_order_relaxed);
      active_ = build_;
      gen_++;
      // Generation and next-index in ONE word, so a worker cannot claim an index from a batch
      // it is not on (see work_on).
      cursor_.store((uint64_t)gen_ << 32, std::memory_order_release);
    }
    cv_.notify_all();
    // The submitting (coordinator) thread pulls jobs too: at low miss counts that is often the
    // whole batch, and it avoids paying a wakeup for a single read.
    work(gen_);
    std::unique_lock<std::mutex> lk(done_mtx_);
    done_cv_.wait(lk, [&] {
      return done_jobs_.load(std::memory_order_acquire) >= total;
    });
  }

  void reader_loop() {
    uint64_t seen = 0;
    for (;;) {
      const std::vector<ReadJob>* batch;
      uint64_t my_gen;
      {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        my_gen = gen_;
        batch = active_;
      }
      work_on(batch, my_gen);
    }
  }

  void work(uint64_t my_gen) { work_on(active_, my_gen); }

  // Claim (generation, index) as ONE atomic word.
  //
  // Checking the generation and then fetch_add'ing a separate index is not enough: a reader
  // descheduled between the two (16 unpinned readers against 24 spinning MoE workers on 24
  // physical cores) wakes into the NEXT batch and takes index 0 of it while still holding the
  // previous batch's pointer -- which by then is the buffer resolve() is push_back'ing into.
  // It would read a torn job, and credit `done_jobs_` for a batch whose job 0 nobody ran, so
  // the coordinator publishes slots whose bytes never landed. Rare, silent, wrong weights.
  void work_on(const std::vector<ReadJob>* batch, uint64_t my_gen) {
    if (batch == nullptr) return;
    const uint32_t total = (uint32_t)total_.load(std::memory_order_relaxed);
    uint64_t cur = cursor_.load(std::memory_order_acquire);
    for (;;) {
      if ((cur >> 32) != my_gen) return;  // a newer batch owns the cursor: nothing here is ours
      const uint32_t i = (uint32_t)cur;
      if (i >= total) return;
      if (!cursor_.compare_exchange_weak(cur, cur + 1, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
        continue;  // cur was refreshed by the failed CAS
      }
      do_job((*batch)[i]);
      if ((uint32_t)(done_jobs_.fetch_add(1, std::memory_order_acq_rel) + 1) == total) {
        std::lock_guard<std::mutex> lk(done_mtx_);
        done_cv_.notify_all();
      }
      cur = cursor_.load(std::memory_order_acquire);
    }
  }

  // One O_DIRECT fd per shard, SHARED by every reader thread. pread takes an explicit offset
  // and never touches the file position, so there is no shared-position lock to dodge -- and a
  // per-thread cache multiplies the descriptor count by the pool size, which together with the
  // Python readers exhausted a 1024 fd limit.
  int fd_for(const std::string& path) {
    {
      std::lock_guard<std::mutex> lk(fd_mtx_);
      auto it = fds_.find(path);
      if (it != fds_.end()) return it->second;
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    std::lock_guard<std::mutex> lk(fd_mtx_);
    auto it = fds_.find(path);
    if (it != fds_.end()) {           // lost the race: keep the winner, drop ours
      if (fd >= 0) ::close(fd);
      return it->second;
    }
    fds_.emplace(path, fd);
    return fd;
  }

  // One expert row, split at shard boundaries.
  //
  // The FTW entry offset and the shard cut are both 4096-aligned, so when a bank's row bytes
  // are a whole number of blocks (the packed and scale banks) source, length and the
  // page-aligned destination line up and the read DMAs straight into the pinned slot. NVFP4's
  // two tiny *_global banks are 2,560 and 5,120 B and never line up; those read the enclosing
  // aligned window into a thread-local bounce and memcpy the row out. They are 0.3% of the
  // bytes, so the copy is irrelevant -- but doing it unconditionally would not be.
  void do_job(const ReadJob& j) {
    if ((j.global_off % kAlign) != 0 || (j.nbytes % kAlign) != 0 ||
        ((uintptr_t)j.dst % kAlign) != 0) {
      do_job_bounced(j);
      return;
    }
    read_direct(j);
  }

  void read_direct(const ReadJob& j) {
    int64_t pos = j.global_off;
    int64_t left = j.nbytes;
    uint8_t* dst = j.dst;
    for (size_t si = 0; si < shards.size() && left > 0; ++si) {
      const Shard& sh = shards[si];
      if (pos < sh.global_off || pos >= sh.global_off + sh.nbytes) continue;
      const int64_t take = std::min<int64_t>(left, sh.global_off + sh.nbytes - pos);
      const int fd = fd_for(sh.path);
      if (fd < 0) return;
      int64_t got = 0;
      while (got < take) {
        const ssize_t n = ::pread(fd, dst + got, (size_t)(take - got), (off_t)(pos - sh.global_off + got));
        if (n <= 0) return;  // short read: the slot stays unpublished, caller leaves it to CPU
        got += n;
      }
      n_bytes.fetch_add(take, std::memory_order_relaxed);
      pos += take;
      left -= take;
      dst += take;
      si = (size_t)-1;  // rescan from the start for the next piece
    }
  }

  // Aligned-window read + copy, for banks whose row is not a whole block (see do_job).
  void do_job_bounced(const ReadJob& j) {
    thread_local std::vector<uint8_t> buf;
    const int64_t head = j.global_off % kAlign;
    const int64_t span = ((head + j.nbytes + kAlign - 1) / kAlign) * kAlign;
    if ((int64_t)buf.size() < span + kAlign) buf.resize(span + kAlign);
    uint8_t* aligned = buf.data() + ((kAlign - ((uintptr_t)buf.data() % kAlign)) % kAlign);
    ReadJob w;
    w.global_off = j.global_off - head;
    w.nbytes = span;
    w.dst = aligned;
    read_direct(w);
    std::memcpy(j.dst, aligned + head, (size_t)j.nbytes);
  }

  std::unordered_map<std::string, int> fds_;
  std::mutex fd_mtx_;
  std::vector<int32_t> claimed_;
  std::vector<Pending> pending_;
  std::vector<ReadJob> jobs_buf_[2];
  std::vector<ReadJob>* build_ = &jobs_buf_[0];
  const std::vector<ReadJob>* active_ = nullptr;
  std::atomic<int> total_{0};
  std::vector<std::thread> readers_;
  std::mutex mtx_, done_mtx_;
  std::condition_variable cv_, done_cv_;
  std::atomic<uint64_t> cursor_{0};  // (generation << 32) | next job index
  std::atomic<int> done_jobs_{0};
  uint64_t gen_ = 0;
  bool stop_ = false;
};

}  // namespace ft_tier3
