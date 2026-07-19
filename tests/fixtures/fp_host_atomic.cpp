// Host-side std::atomic: not a GPU site.
#include <atomic>
static std::atomic<int> counter{0};
void bump() { counter.fetch_add(1, std::memory_order_relaxed); }
