# Complete Conversation Markdown Dump

```markdown
# Concurrency and Lock-Free Data Structures - Complete Guide

---

## Table of Contents

1. [Lock-Based vs Lock-Free Data Structures](#lock-based-vs-lock-free-data-structures)
2. [Lock-Based vs Lock-Free in Java](#lock-based-vs-lock-free-in-java)
3. [Connection Pool Manager](#connection-pool-manager)
4. [Priority Task Scheduler](#priority-task-scheduler)
5. [Lock-Free Fast Path Explanation](#lock-free-fast-path-explanation)
6. [CAS and Lock-Free Concepts](#cas-and-lock-free-concepts)
7. [CAS in Task Scheduler](#cas-in-task-scheduler)
8. [Multiple Cashiers Analogy](#multiple-cashiers-analogy)
9. [Web Crawler Design](#web-crawler-design)
10. [Pattern Recognition](#pattern-recognition)
11. [Complete List of Concurrency Patterns](#complete-list-of-concurrency-patterns)
12. [Lock-Free Data Structures Reference](#lock-free-data-structures-reference)
13. [Distributed Lock Manager](#distributed-lock-manager)

---

# Lock-Based vs Lock-Free Data Structures

## Lock-Based (Blocking)

### How It Works
- Uses **mutexes, semaphores, or locks** to protect shared data
- Threads must **acquire locks** before accessing data
- Other threads **wait/block** until lock is released

### Example
```c
// Lock-based stack
typedef struct {
    Node* head;
    pthread_mutex_t lock;
} Stack;

void push(Stack* s, int value) {
    pthread_mutex_lock(&s->lock);  // Acquire lock
    Node* node = create_node(value);
    node->next = s->head;
    s->head = node;
    pthread_mutex_unlock(&s->lock); // Release lock
}
```

### Pros & Cons
✅ **Simpler** to design and understand  
✅ **Easier to debug**  
✅ Works well with **low contention**  

❌ **Deadlock** potential  
❌ **Priority inversion**  
❌ **No progress** if lock holder is suspended  
❌ **Poor scalability** under high contention  

---

## Lock-Free (Non-Blocking)

### How It Works
- Uses **atomic operations** (CAS - Compare-And-Swap)
- Multiple threads operate **simultaneously**
- Failed operations **retry** instead of blocking
- **No locks** used

### Example
```c
// Lock-free stack
typedef struct {
    atomic<Node*> head;
} Stack;

void push(Stack* s, int value) {
    Node* node = create_node(value);
    Node* old_head;
    do {
        old_head = s->head.load();
        node->next = old_head;
    } while (!s->head.compare_exchange_weak(old_head, node));
    // Retry until CAS succeeds
}
```

### Pros & Cons
✅ **No deadlocks**  
✅ **System-wide progress** guaranteed  
✅ **Better scalability** under contention  
✅ **Resilient** to thread crashes  

❌ **Much harder** to implement correctly  
❌ **ABA problem** concerns  
❌ **Complex memory management**  
❌ Potential for **livelock/starvation**  

---

## Key Differences

| Aspect | Lock-Based | Lock-Free |
|--------|-----------|-----------|
| **Mechanism** | Mutexes/locks | Atomic CAS operations |
| **Blocking** | Threads block/wait | Threads never block |
| **Progress** | No guarantee | ≥1 thread progresses |
| **Deadlock** | Possible | Impossible |
| **Complexity** | Lower | Much higher |
| **Performance** | Good (low contention) | Better (high contention) |
| **Use Case** | General purpose | High-performance systems |

---

## Progress Guarantees Hierarchy

1. **Blocking**: No progress if lock holder stalls
2. **Obstruction-free**: Progress if running alone
3. **Lock-free**: ≥1 thread always progresses
4. **Wait-free**: Every thread progresses (strongest)

---

## When to Use Each

**Use Lock-Based when:**
- Development speed matters
- Low contention expected
- Simpler codebase preferred

**Use Lock-Free when:**
- High contention scenarios
- Real-time requirements
- Maximum throughput needed
- Thread failures must be tolerated

---

# Lock-Based vs Lock-Free Data Structures in Java

## Lock-Based (Blocking) Approach

### Using `synchronized`
```java
public class LockBasedStack<T> {
    private Node<T> head;
    
    private static class Node<T> {
        final T value;
        Node<T> next;
        
        Node(T value) { this.value = value; }
    }
    
    // synchronized keyword = implicit lock
    public synchronized void push(T value) {
        Node<T> newNode = new Node<>(value);
        newNode.next = head;
        head = newNode;
    }
    
    public synchronized T pop() {
        if (head == null) return null;
        T value = head.value;
        head = head.next;
        return value;
    }
}
```

### Using `ReentrantLock`
```java
import java.util.concurrent.locks.ReentrantLock;

public class ExplicitLockStack<T> {
    private Node<T> head;
    private final ReentrantLock lock = new ReentrantLock();
    
    public void push(T value) {
        lock.lock();  // Acquire lock
        try {
            Node<T> newNode = new Node<>(value);
            newNode.next = head;
            head = newNode;
        } finally {
            lock.unlock();  // Always release
        }
    }
    
    public T pop() {
        lock.lock();
        try {
            if (head == null) return null;
            T value = head.value;
            head = head.next;
            return value;
        } finally {
            lock.unlock();
        }
    }
}
```

---

## Lock-Free (Non-Blocking) Approach

### Using `AtomicReference` and CAS
```java
import java.util.concurrent.atomic.AtomicReference;

public class LockFreeStack<T> {
    private final AtomicReference<Node<T>> head = 
        new AtomicReference<>();
    
    private static class Node<T> {
        final T value;
        final Node<T> next;
        
        Node(T value, Node<T> next) {
            this.value = value;
            this.next = next;
        }
    }
    
    public void push(T value) {
        Node<T> newHead;
        Node<T> oldHead;
        
        do {
            oldHead = head.get();
            newHead = new Node<>(value, oldHead);
            // Retry until CAS succeeds
        } while (!head.compareAndSet(oldHead, newHead));
    }
    
    public T pop() {
        Node<T> oldHead;
        Node<T> newHead;
        
        do {
            oldHead = head.get();
            if (oldHead == null) return null;
            newHead = oldHead.next;
        } while (!head.compareAndSet(oldHead, newHead));
        
        return oldHead.value;
    }
}
```

---

## Real-World Java Examples

### Java's Built-in Lock-Based Collections
```java
import java.util.concurrent.*;

// Uses ReentrantLock internally
BlockingQueue<String> queue = new LinkedBlockingQueue<>();

// Uses synchronized internally
Vector<String> vector = new Vector<>();

// Wrapper that adds synchronization
List<String> syncList = Collections.synchronizedList(new ArrayList<>());
```

### Java's Built-in Lock-Free Collections
```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

// Lock-free queue
ConcurrentLinkedQueue<String> queue = new ConcurrentLinkedQueue<>();

// Lock-free stack
ConcurrentLinkedDeque<String> stack = new ConcurrentLinkedDeque<>();

// Atomic variables (lock-free primitives)
AtomicInteger counter = new AtomicInteger(0);
AtomicReference<String> ref = new AtomicReference<>();
AtomicLong longValue = new AtomicLong(0);
```

---

## Performance Comparison Example

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

public class PerformanceTest {
    
    // Lock-based counter
    static class LockBasedCounter {
        private int count = 0;
        
        public synchronized void increment() {
            count++;
        }
        
        public synchronized int get() {
            return count;
        }
    }
    
    // Lock-free counter
    static class LockFreeCounter {
        private final AtomicInteger count = new AtomicInteger(0);
        
        public void increment() {
            count.incrementAndGet();  // CAS-based
        }
        
        public int get() {
            return count.get();
        }
    }
    
    public static void main(String[] args) throws InterruptedException {
        int threads = 10;
        int iterations = 100_000;
        
        // Test lock-based
        LockBasedCounter lockBased = new LockBasedCounter();
        long start = System.nanoTime();
        runTest(threads, iterations, lockBased::increment);
        long lockTime = System.nanoTime() - start;
        
        // Test lock-free
        LockFreeCounter lockFree = new LockFreeCounter();
        start = System.nanoTime();
        runTest(threads, iterations, lockFree::increment);
        long lockFreeTime = System.nanoTime() - start;
        
        System.out.println("Lock-based: " + lockTime/1_000_000 + "ms");
        System.out.println("Lock-free: " + lockFreeTime/1_000_000 + "ms");
    }
    
    static void runTest(int threads, int iterations, Runnable task) 
            throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch latch = new CountDownLatch(threads);
        
        for (int i = 0; i < threads; i++) {
            executor.submit(() -> {
                for (int j = 0; j < iterations; j++) {
                    task.run();
                }
                latch.countDown();
            });
        }
        
        latch.await();
        executor.shutdown();
    }
}
```

---

## ABA Problem Example

The ABA problem is a common pitfall in lock-free programming:

```java
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicStampedReference;

public class ABAExample {
    
    // Vulnerable to ABA problem
    static class VulnerableStack<T> {
        private final AtomicReference<Node<T>> head = new AtomicReference<>();
        
        // Thread 1 reads A, gets suspended
        // Thread 2 pops A, pops B, pushes A back
        // Thread 1 resumes, CAS succeeds but B is lost!
    }
    
    // Solution: Use stamped reference
    static class SafeStack<T> {
        private final AtomicStampedReference<Node<T>> head = 
            new AtomicStampedReference<>(null, 0);
        
        public void push(T value) {
            Node<T> newHead;
            Node<T> oldHead;
            int[] stampHolder = new int[1];
            
            do {
                oldHead = head.get(stampHolder);
                int oldStamp = stampHolder[0];
                newHead = new Node<>(value, oldHead);
            } while (!head.compareAndSet(oldHead, newHead, 
                                        oldStamp, oldStamp + 1));
        }
    }
}
```

---

## Comparison Table

| Feature | Lock-Based | Lock-Free |
|---------|-----------|-----------|
| **Java Primitives** | `synchronized`, `ReentrantLock` | `AtomicReference`, CAS |
| **Blocking** | Thread waits | Thread retries |
| **Code Complexity** | Simple | Complex |
| **JVM Collections** | `Vector`, `Hashtable` | `ConcurrentLinkedQueue` |
| **Garbage Collection** | Easier | Need careful handling |
| **Debug Difficulty** | Moderate | High |

---

## When to Use in Java

### Use Lock-Based:
```java
// Simple scenarios
synchronized (this) { /* critical section */ }

// Need fairness guarantees
ReentrantLock lock = new ReentrantLock(true);

// Complex operations
lock.lock();
try {
    // Multiple steps atomically
} finally {
    lock.unlock();
}
```

### Use Lock-Free:
```java
// High-performance counters
AtomicInteger counter = new AtomicInteger();

// Simple data structures with high contention
ConcurrentLinkedQueue<Task> tasks = new ConcurrentLinkedQueue<>();

// Single-variable updates
AtomicReference<Config> config = new AtomicReference<>();
config.updateAndGet(old -> new Config(old));
```

**General Rule**: Start with lock-based (simpler), optimize to lock-free only if profiling shows contention issues.

---

# Connection Pool Manager - Problem Explanation

## What is a Connection Pool?

Imagine you're building an application that needs to talk to a database. Every time you want to query the database, you need a **connection** (think of it like a phone line to the database).

### The Problem Without a Pool:

```
Request 1 comes in → Open new DB connection → Query → Close connection
Request 2 comes in → Open new DB connection → Query → Close connection
Request 3 comes in → Open new DB connection → Query → Close connection
...
```

**Issues:**
- Opening/closing connections is **expensive** (network handshake, authentication, etc.)
- Database has **limited connections** (e.g., PostgreSQL might allow 100 max)
- Each connection uses **memory and resources**
- If 1000 requests come at once, you try to open 1000 connections → **database crashes**

### The Solution: Connection Pool

```
Application Startup → Create pool of 10 connections → Keep them open

Request 1 → Borrow connection #3 → Query → Return to pool
Request 2 → Borrow connection #7 → Query → Return to pool
Request 3 → Borrow connection #3 (reused!) → Query → Return to pool
```

**Benefits:**
- Connections are **reused** (no overhead of opening/closing)
- **Limit** concurrent database operations (protect DB from overload)
- **Faster** response times (connection already established)

---

## Real-World Context

### Where You'll See This:

1. **Database Drivers**
   - HikariCP (Java's fastest connection pool)
   - Apache DBCP
   - C3P0

2. **HTTP Client Libraries**
   - Apache HttpClient connection pooling
   - OkHttp connection pool

3. **Message Queue Clients**
   - RabbitMQ channel pools
   - Kafka producer connection pools

4. **Cloud Services**
   - Redis connection pools
   - S3 client connection pools

### At Rubrix (Data Infrastructure Company):

Imagine you're building a system that:
- Ingests data from multiple sources
- Processes 10,000 requests/second
- Writes to PostgreSQL database
- Each write needs a connection

Without pooling:
```
10,000 requests/sec × open connection (50ms) = system crawls
Database: "I only allow 100 connections!" = crash
```

With pooling:
```
Pool of 20 connections → Reused efficiently
Excess requests wait → No database overload
System stable and fast ✓
```

---

## The Problem Statement

**Design and implement a thread-safe Connection Pool that:**

1. Maintains a **fixed number** of connections (e.g., 10 connections max)
2. **Reuses** connections instead of creating new ones
3. **Blocks** threads when all connections are in use (until one becomes available)
4. Supports **timeout** (don't wait forever if system is overloaded)
5. Properly handles **connection lifecycle** (creation, validation, cleanup)
6. Is **thread-safe** for concurrent access by multiple threads
7. Provides **monitoring** (how many connections in use, available, etc.)

---

## Requirements Breakdown

### Functional Requirements:

1. **acquire()** - Get a connection from pool
   - If available → return immediately
   - If all busy → wait (block) until one is free
   - Support timeout (e.g., wait max 5 seconds)

2. **release()** - Return connection to pool
   - Mark as available
   - Wake up waiting threads

3. **Size Management:**
   - Start with N connections
   - Never exceed max size
   - Option to grow/shrink (advanced)

4. **Connection Health:**
   - Validate connections before giving them out
   - Handle broken connections
   - Replace dead connections

### Non-Functional Requirements:

1. **Thread Safety** - Multiple threads can acquire/release concurrently
2. **No Deadlocks** - System should never hang
3. **Fairness** - Threads get connections in order (no starvation)
4. **Performance** - Low overhead for acquire/release
5. **Monitoring** - Track pool statistics

---

## Key Challenges & Edge Cases

### Challenge 1: What if all connections are in use?

```
Pool has 3 connections
Thread-A: acquired connection #1
Thread-B: acquired connection #2  
Thread-C: acquired connection #3
Thread-D: tries to acquire → MUST WAIT

When Thread-A releases connection #1:
Thread-D: wakes up and gets connection #1
```

**Solution:** Use **Condition Variables** to wait/signal

---

### Challenge 2: Connection dies while in use

```
Thread-1: acquires connection
Thread-1: executes query
Network blip: connection breaks
Thread-1: returns broken connection to pool

Thread-2: acquires same connection
Thread-2: tries to use it → FAILS!
```

**Solution:** Validate connections before handing them out, replace broken ones

---

### Challenge 3: Thread holds connection forever

```
Thread-1: acquires connection
Thread-1: starts long-running query (30 seconds)
Threads 2-20: all waiting for connections
System becomes unresponsive
```

**Solution:** 
- Implement **timeout** on acquire
- Consider **connection leak detection** (warn if held too long)

---

### Challenge 4: Thread crashes while holding connection

```
Thread-1: acquires connection
Thread-1: throws exception and dies
Connection never returned to pool
Pool size shrinks over time
```

**Solution:** Use **try-finally** pattern, automatic cleanup

---

### Challenge 5: Thundering herd

```
Pool has 1 connection available
100 threads waiting

Connection becomes available
All 100 threads wake up and compete
Only 1 wins, 99 go back to sleep
Inefficient!
```

**Solution:** Use `signal()` instead of `signalAll()` (wake one thread at a time)

---

## Design Considerations

### Data Structure Choice:

```
Option 1: Queue (LinkedList)
✓ FIFO fairness
✓ Simple add/remove
✓ Our choice

Option 2: Stack
✓ Better cache locality (reuse recent connections)
✗ No fairness guarantees

Option 3: Set/Map
✓ Fast lookup by ID
✗ No ordering
```

### Lock Strategy:

```
Option 1: Single Lock (ReentrantLock)
✓ Simple
✓ Sufficient for most cases
✓ Our choice

Option 2: Lock-Free (CAS)
✓ Better under high contention
✗ Complex implementation
✗ Harder to implement blocking

Option 3: Read-Write Lock
✗ Not applicable (no read-only operations)
```

### Waiting Strategy:

```
Option 1: Block with Condition Variable
✓ Efficient (no busy waiting)
✓ OS-level sleep
✓ Our choice

Option 2: Busy Waiting (spin lock)
✗ Wastes CPU
✗ Not suitable for long waits

Option 3: Timed Polling
✗ Delayed response
✗ Still wastes some CPU
```

---

## High-Level Design

```
┌─────────────────────────────────────┐
│     Connection Pool Manager          │
│                                      │
│  ┌────────────────────────────────┐ │
│  │   Available Connections        │ │
│  │   Queue<Connection>            │ │
│  │   [Conn1, Conn2, Conn3]        │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │   In-Use Tracking              │ │
│  │   Set<Connection>              │ │
│  │   {Conn4, Conn5}               │ │
│  └────────────────────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │   Lock + Condition Variable    │ │
│  │   - Protects shared state      │ │
│  │   - Signals availability       │ │
│  └────────────────────────────────┘ │
└─────────────────────────────────────┘

Thread 1 (acquire)                Thread 2 (release)
     │                                  │
     ├─> Lock                           ├─> Lock
     │                                  │
     ├─> Check available?               ├─> Return to available
     │                                  │
     ├─> If empty: wait()               ├─> Signal waiting threads
     │                                  │
     ├─> If available: take it          ├─> Unlock
     │                                  │
     ├─> Unlock                         └─> Done
     │
     └─> Use connection
```

---

## API Design

```java
public class ConnectionPool {
    
    // Constructor
    public ConnectionPool(int poolSize, ConnectionFactory factory)
    
    // Core operations
    public Connection acquire() throws InterruptedException
    public Connection acquire(long timeout, TimeUnit unit) throws InterruptedException
    public void release(Connection connection)
    
    // Lifecycle
    public void shutdown()
    
    // Monitoring
    public int getAvailableConnections()
    public int getActiveConnections()
    public int getTotalConnections()
    public PoolStats getStats()
}
```

---

## Example Usage Scenario

```java
// Application startup
ConnectionPool pool = new ConnectionPool(10, new PostgresConnectionFactory());

// Request handler thread
public void handleRequest(Request request) {
    Connection conn = null;
    try {
        // Acquire connection (blocks if none available)
        conn = pool.acquire(5, TimeUnit.SECONDS);
        
        // Use connection
        ResultSet rs = conn.executeQuery("SELECT * FROM users WHERE id = ?", request.userId);
        
        // Process results...
        
    } catch (TimeoutException e) {
        // Pool exhausted - system overloaded
        return "Service unavailable - try again later";
    } finally {
        // Always return to pool
        if (conn != null) {
            pool.release(conn);
        }
    }
}

// Application shutdown
pool.shutdown();
```

---

# Connection Pool Manager - Step-by-Step Implementation

## Step 1: Define the Connection Interface

First, let's create what a "Connection" looks like. In real life, this would be a database connection, but we'll create a simple interface for testing.

```java
// Represents a database connection
public interface Connection {
    void executeQuery(String query);
    boolean isValid();
    void close();
    int getId();
}

// Mock implementation for testing
class MockConnection implements Connection {
    private final int id;
    private boolean closed = false;
    private boolean broken = false;
    
    public MockConnection(int id) {
        this.id = id;
        System.out.println("  [CREATED] Connection-" + id);
    }
    
    @Override
    public void executeQuery(String query) {
        if (closed) {
            throw new IllegalStateException("Connection is closed");
        }
        if (broken) {
            throw new RuntimeException("Connection is broken");
        }
        System.out.println("  [QUERY] Connection-" + id + ": " + query);
        // Simulate query execution
        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    @Override
    public boolean isValid() {
        return !closed && !broken;
    }
    
    @Override
    public void close() {
        closed = true;
        System.out.println("  [CLOSED] Connection-" + id);
    }
    
    @Override
    public int getId() {
        return id;
    }
    
    // For testing - simulate connection failure
    public void breakConnection() {
        this.broken = true;
        System.out.println("  [BROKEN] Connection-" + id);
    }
}

// Factory to create connections
interface ConnectionFactory {
    Connection create();
}

class MockConnectionFactory implements ConnectionFactory {
    private int connectionIdCounter = 0;
    
    @Override
    public Connection create() {
        return new MockConnection(connectionIdCounter++);
    }
}
```

**Key Points:**
- `Connection` interface abstracts the actual connection type
- `isValid()` checks if connection is still usable
- `ConnectionFactory` allows different connection types (Postgres, MySQL, etc.)

---

## Step 2: Basic Pool Structure

Now let's build the core pool with just the essential parts:

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;

public class ConnectionPool {
    
    // Configuration
    private final int maxPoolSize;
    private final ConnectionFactory factory;
    
    // Pool state
    private final Queue<Connection> availableConnections;
    private final Set<Connection> inUseConnections;
    
    // Thread safety
    private final Lock lock;
    private final Condition connectionAvailable;
    
    // Lifecycle
    private volatile boolean isShutdown = false;
    
    public ConnectionPool(int poolSize, ConnectionFactory factory) {
        this.maxPoolSize = poolSize;
        this.factory = factory;
        this.availableConnections = new LinkedList<>();
        this.inUseConnections = new HashSet<>();
        this.lock = new ReentrantLock();
        this.connectionAvailable = lock.newCondition();
        
        // Initialize pool with connections
        initializePool();
    }
    
    private void initializePool() {
        System.out.println("Initializing pool with " + maxPoolSize + " connections...");
        for (int i = 0; i < maxPoolSize; i++) {
            Connection conn = factory.create();
            availableConnections.offer(conn);
        }
        System.out.println("Pool initialized.\n");
    }
}
```

**Key Points:**
- `availableConnections`: Queue of connections ready to use (FIFO for fairness)
- `inUseConnections`: Set of connections currently borrowed
- `lock` + `connectionAvailable`: For thread-safe waiting/signaling
- Initialize all connections upfront (eager initialization)

---

## Step 3: Implement acquire() - Basic Version

```java
public Connection acquire() throws InterruptedException {
    lock.lock();
    try {
        // Check if pool is shut down
        if (isShutdown) {
            throw new IllegalStateException("Connection pool is shut down");
        }
        
        // Wait until a connection is available
        while (availableConnections.isEmpty()) {
            System.out.println(Thread.currentThread().getName() + 
                " waiting for connection...");
            connectionAvailable.await();
        }
        
        // Get connection from available queue
        Connection connection = availableConnections.poll();
        
        // Move to in-use set
        inUseConnections.add(connection);
        
        System.out.println(Thread.currentThread().getName() + 
            " acquired Connection-" + connection.getId() + 
            " (available: " + availableConnections.size() + ")");
        
        return connection;
        
    } finally {
        lock.unlock();
    }
}
```

**What's Happening:**
1. **Lock acquisition**: Only one thread can modify pool state at a time
2. **Wait loop**: If no connections available, thread sleeps (releases lock)
3. **Condition.await()**: Thread goes to sleep, other threads can proceed
4. **Wake up**: When another thread calls `signal()`, this thread wakes up
5. **Get connection**: Remove from available, add to in-use
6. **Unlock**: Release lock so other threads can proceed

**Why `while` loop instead of `if`?**
```java
// ❌ WRONG - Spurious wakeups can happen
if (availableConnections.isEmpty()) {
    connectionAvailable.await();
}

// ✓ CORRECT - Always recheck condition
while (availableConnections.isEmpty()) {
    connectionAvailable.await();
}
```

---

## Step 4: Implement release()

```java
public void release(Connection connection) {
    if (connection == null) {
        throw new IllegalArgumentException("Connection cannot be null");
    }
    
    lock.lock();
    try {
        // Verify this connection was actually from our pool
        if (!inUseConnections.remove(connection)) {
            throw new IllegalStateException(
                "Connection-" + connection.getId() + 
                " was not borrowed from this pool");
        }
        
        // Return to available queue
        availableConnections.offer(connection);
        
        System.out.println(Thread.currentThread().getName() + 
            " released Connection-" + connection.getId() + 
            " (available: " + availableConnections.size() + ")");
        
        // Wake up ONE waiting thread
        connectionAvailable.signal();
        
    } finally {
        lock.unlock();
    }
}
```

**Key Points:**
- **Validation**: Ensure connection was actually borrowed from this pool
- **Remove from in-use**: Mark as no longer borrowed
- **Add to available**: Back to the queue
- **Signal one thread**: Wake up one waiting thread (not all)

**Why `signal()` instead of `signalAll()`?**
```java
signal()     → Wakes 1 thread (efficient)
signalAll()  → Wakes ALL threads (thundering herd)

If 100 threads waiting and 1 connection available:
- signal(): Wake 1, that 1 gets connection ✓
- signalAll(): Wake 100, 99 go back to sleep ✗ (wasteful)
```

---

## Step 5: Add Timeout Support

```java
public Connection acquire(long timeout, TimeUnit unit) throws InterruptedException {
    long nanosTimeout = unit.toNanos(timeout);
    long deadline = System.nanoTime() + nanosTimeout;
    
    lock.lock();
    try {
        if (isShutdown) {
            throw new IllegalStateException("Connection pool is shut down");
        }
        
        // Wait with timeout
        while (availableConnections.isEmpty()) {
            if (nanosTimeout <= 0) {
                // Timeout expired
                System.out.println(Thread.currentThread().getName() + 
                    " TIMEOUT waiting for connection");
                throw new TimeoutException("Timeout acquiring connection after " + 
                    timeout + " " + unit);
            }
            
            System.out.println(Thread.currentThread().getName() + 
                " waiting for connection (timeout: " + 
                TimeUnit.NANOSECONDS.toMillis(nanosTimeout) + "ms)");
            
            // Wait for specified time
            nanosTimeout = connectionAvailable.awaitNanos(nanosTimeout);
        }
        
        // Got a connection
        Connection connection = availableConnections.poll();
        inUseConnections.add(connection);
        
        System.out.println(Thread.currentThread().getName() + 
            " acquired Connection-" + connection.getId());
        
        return connection;
        
    } finally {
        lock.unlock();
    }
}
```

**Key Points:**
- `awaitNanos()` returns remaining wait time
- Automatically handles spurious wakeups
- Throws `TimeoutException` if time expires

---

## Step 6: Add Connection Validation

What if a connection breaks while in the pool?

```java
public Connection acquire(long timeout, TimeUnit unit) throws InterruptedException {
    long nanosTimeout = unit.toNanos(timeout);
    
    lock.lock();
    try {
        if (isShutdown) {
            throw new IllegalStateException("Connection pool is shut down");
        }
        
        while (true) {
            // Wait for available connection
            while (availableConnections.isEmpty()) {
                if (nanosTimeout <= 0) {
                    throw new TimeoutException("Timeout acquiring connection");
                }
                nanosTimeout = connectionAvailable.awaitNanos(nanosTimeout);
            }
            
            // Get connection
            Connection connection = availableConnections.poll();
            
            // ⭐ VALIDATE CONNECTION
            if (!connection.isValid()) {
                System.out.println("  [INVALID] Connection-" + connection.getId() + 
                    " is broken, creating new one");
                
                // Close broken connection
                connection.close();
                
                // Create replacement
                Connection newConnection = factory.create();
                availableConnections.offer(newConnection);
                
                // Try again
                continue;
            }
            
            // Connection is valid
            inUseConnections.add(connection);
            
            System.out.println(Thread.currentThread().getName() + 
                " acquired Connection-" + connection.getId());
            
            return connection;
        }
        
    } finally {
        lock.unlock();
    }
}
```

**Key Points:**
- Validate before handing out
- Replace broken connections
- Continue loop to try again

---

## Step 7: Add Monitoring & Statistics

```java
// Add to ConnectionPool class

// Statistics
private long totalAcquires = 0;
private long totalReleases = 0;
private long totalTimeouts = 0;
private long totalWaitTimeMs = 0;

public static class PoolStats {
    public final int totalConnections;
    public final int availableConnections;
    public final int activeConnections;
    public final long totalAcquires;
    public final long totalReleases;
    public final long totalTimeouts;
    public final double averageWaitTimeMs;
    
    public PoolStats(int total, int available, int active, 
                     long acquires, long releases, long timeouts, double avgWait) {
        this.totalConnections = total;
        this.availableConnections = available;
        this.activeConnections = active;
        this.totalAcquires = acquires;
        this.totalReleases = releases;
        this.totalTimeouts = timeouts;
        this.averageWaitTimeMs = avgWait;
    }
    
    @Override
    public String toString() {
        return String.format(
            "PoolStats{total=%d, available=%d, active=%d, " +
            "acquires=%d, releases=%d, timeouts=%d, avgWait=%.2fms}",
            totalConnections, availableConnections, activeConnections,
            totalAcquires, totalReleases, totalTimeouts, averageWaitTimeMs
        );
    }
}

public PoolStats getStats() {
    lock.lock();
    try {
        int total = availableConnections.size() + inUseConnections.size();
        int available = availableConnections.size();
        int active = inUseConnections.size();
        double avgWait = totalAcquires == 0 ? 0 : 
            (double) totalWaitTimeMs / totalAcquires;
        
        return new PoolStats(total, available, active, 
            totalAcquires, totalReleases, totalTimeouts, avgWait);
    } finally {
        lock.unlock();
    }
}

public int getAvailableCount() {
    lock.lock();
    try {
        return availableConnections.size();
    } finally {
        lock.unlock();
    }
}

public int getActiveCount() {
    lock.lock();
    try {
        return inUseConnections.size();
    } finally {
        lock.unlock();
    }
}
```

---

## Step 8: Add Shutdown

```java
public void shutdown() {
    lock.lock();
    try {
        if (isShutdown) {
            return;
        }
        
        System.out.println("\nShutting down connection pool...");
        
        isShutdown = true;
        
        // Wake up all waiting threads
        connectionAvailable.signalAll();
        
        // Close all available connections
        for (Connection conn : availableConnections) {
            conn.close();
        }
        availableConnections.clear();
        
        // Warn about in-use connections
        if (!inUseConnections.isEmpty()) {
            System.out.println("WARNING: " + inUseConnections.size() + 
                " connections still in use during shutdown");
        }
        
        System.out.println("Pool shut down.\n");
        
    } finally {
        lock.unlock();
    }
}
```

---

# Priority Task Scheduler - Interview Format

## Initial Discussion

**Interviewer**: "Design a task scheduler that can execute tasks with different priorities and support delayed execution. Multiple threads should be able to submit tasks concurrently."

**Me**: "Interesting! Let me clarify the requirements:

1. **Priorities**: How do priorities work? Lower number = higher priority, or vice versa?
2. **Delays**: Can tasks be scheduled to run in the future (like 'run this in 5 seconds')?
3. **Execution**: Do we need a thread pool to execute tasks, or just schedule them?
4. **Priority vs Delay**: If Task A has higher priority but scheduled later, and Task B has lower priority but ready now, which runs first?
5. **Cancellation**: Should we support canceling scheduled tasks?"

**Interviewer**: "Good questions. Let's say:
- Lower number = higher priority
- Yes, support delays (run in X milliseconds)
- You need to actually execute them with a thread pool
- Time takes precedence - execute tasks when they're ready, then by priority
- Let's skip cancellation for now"

**Me**: "Perfect. So if I understand correctly:
```
Task A: priority=1, delay=5000ms  (high priority, run in 5 seconds)
Task B: priority=10, delay=0ms    (low priority, run immediately)

Task B executes first (ready now), then Task A (ready in 5 seconds)
```
Is that right?"

**Interviewer**: "Exactly."

---

## High-Level Design Discussion

**Me**: "Here's my approach:

### Core Components:
```
TaskScheduler
├── PriorityQueue (ordered by time, then priority)
├── Scheduler Thread (monitors queue, dispatches ready tasks)
├── Executor Pool (actually runs the tasks)
├── Lock + Condition (for thread-safe coordination)
└── schedule() method
```

### Why these choices?

**PriorityQueue**: 
- Automatically keeps tasks ordered
- Need custom comparator: sort by execution time first, then priority

**Scheduler Thread**:
- Continuously monitors the queue
- Waits until next task is ready
- Dispatches to executor pool

**Executor Pool**:
- Separate execution from scheduling
- Allows parallel task execution

**Lock + Condition**:
- Protect queue from concurrent access
- Wake up scheduler when new task added

Does this architecture make sense?"

**Interviewer**: "Yes. Why separate scheduler thread from executor threads?"

**Me**: "Great question! Separation of concerns:

```
Without separation:
Thread 1: schedule(task) → waits 5 seconds → executes
Thread 2: schedule(task) → blocked waiting for Thread 1
❌ Scheduling is blocked by execution

With separation:
Thread 1: schedule(task) → returns immediately ✓
Scheduler: monitors queue → dispatches when ready
Executor: runs tasks in parallel
✓ Scheduling never blocks on execution
```

Also, the scheduler is a **single point of coordination** - simpler than multiple threads fighting over the queue."

---

## Step 1: Task Structure

**Interviewer**: "Let's start coding. What does a Task look like?"

**Me**:
```java
class Task implements Comparable<Task> {
    final String id;
    final Runnable runnable;
    final int priority;
    final long executeAtMs;
    
    Task(String id, Runnable runnable, int priority, long delayMs) {
        this.id = id;
        this.runnable = runnable;
        this.priority = priority;
        this.executeAtMs = System.currentTimeMillis() + delayMs;
    }
    
    @Override
    public int compareTo(Task other) {
        // First compare by execution time
        int timeCompare = Long.compare(this.executeAtMs, other.executeAtMs);
        if (timeCompare != 0) {
            return timeCompare;
        }
        // If same time, compare by priority
        return Integer.compare(this.priority, other.priority);
    }
}
```

**Key decisions**:
- Store absolute execution time (not relative delay)
- Implement `Comparable` for PriorityQueue ordering
- Time takes precedence over priority in comparator"

**Interviewer**: "Why store absolute time instead of delay?"

**Me**: "Because we need to know **when** to execute, not just **how long** to wait:

```
Current time: 1000ms
Task scheduled with 5000ms delay

Storing delay: 5000
- Later: Is it ready? Need to track when it was scheduled
- Complicated!

Storing absolute time: 1000 + 5000 = 6000
- Later: Is it ready? currentTime >= 6000
- Simple comparison!
```
"

---

## Step 2: Basic Structure

**Me**: "Now the scheduler class:

```java
public class TaskScheduler {
    private final PriorityQueue<Task> taskQueue;
    private final Lock lock;
    private final Condition taskAvailable;
    private final ExecutorService executorPool;
    private final Thread schedulerThread;
    private volatile boolean shutdown = false;
    
    public TaskScheduler(int workerThreads) {
        this.taskQueue = new PriorityQueue<>();
        this.lock = new ReentrantLock();
        this.taskAvailable = lock.newCondition();
        this.executorPool = Executors.newFixedThreadPool(workerThreads);
        
        this.schedulerThread = new Thread(this::scheduleLoop);
        this.schedulerThread.start();
    }
    
    private void scheduleLoop() {
        // We'll implement this next
    }
}
```

**Why volatile for shutdown?**
- Read by scheduler thread, written by main thread
- Need visibility guarantee across threads
- volatile ensures changes are seen immediately"

---

## Step 3: The schedule() Method

**Interviewer**: "Implement the schedule method - how do users submit tasks?"

**Me**:
```java
public void schedule(String taskId, Runnable task, int priority, long delayMs) {
    lock.lock();
    try {
        if (shutdown) {
            throw new IllegalStateException("Scheduler is shut down");
        }
        
        Task newTask = new Task(taskId, task, priority, delayMs);
        taskQueue.offer(newTask);
        
        // Wake up scheduler - might need to reorder
        taskAvailable.signal();
        
    } finally {
        lock.unlock();
    }
}
```

**Critical point**: Why do we signal() after adding?"

**Interviewer**: "Good question - why?"

**Me**: "Because the scheduler might be sleeping! Let me trace through:

```
Time 0: Queue is empty
        Scheduler: "No tasks, I'll sleep indefinitely"
        
Time 1: User calls schedule(task, delay=5000ms)
        Adds to queue
        WITHOUT signal: Scheduler still sleeping! ❌
        WITH signal: Scheduler wakes up, sees new task ✓

Time 2: Scheduler: "Oh! New task at time 6000, I'll wake up then"
```

Even more important - what if new task should run SOONER?

```
Time 0: Queue has Task A (runs at time 10000)
        Scheduler: "I'll wake up at 10000"
        
Time 1: User schedules Task B (runs at time 2000)
        Task B is now at head of queue
        WITHOUT signal: Scheduler sleeps until 10000 ❌
        WITH signal: Scheduler wakes up, sees Task B should run sooner ✓
```
"

---

## Step 4: The Scheduler Loop - Core Logic

**Interviewer**: "Now the tricky part - implement the scheduler loop that monitors the queue."

**Me**: "This is where it gets interesting. The scheduler needs to:
1. Wait if queue is empty
2. Check the next task's time
3. Sleep until that time
4. Wake up and dispatch the task

Let me build it step by step:

```java
private void scheduleLoop() {
    while (!shutdown) {
        lock.lock();
        try {
            // Step 1: Wait for tasks
            while (taskQueue.isEmpty() && !shutdown) {
                taskAvailable.await();  // Sleep indefinitely
            }
            
            if (shutdown) break;
            
            // Step 2: Check next task
            Task nextTask = taskQueue.peek();  // Don't remove yet
            long now = System.currentTimeMillis();
            long waitTime = nextTask.executeAtMs - now;
            
            // Step 3: Is it ready?
            if (waitTime <= 0) {
                // Ready to execute!
                taskQueue.poll();  // Now remove it
                executeTask(nextTask);
            } else {
                // Not ready yet, wait
                taskAvailable.await(waitTime, TimeUnit.MILLISECONDS);
            }
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            break;
        } finally {
            lock.unlock();
        }
    }
}
```

Let me explain the key parts:"

**Interviewer**: "Walk me through the wait logic - why peek() then poll()?"

**Me**: "Excellent catch! This is subtle:

```java
// Why not just poll() immediately?
Task task = taskQueue.poll();  // ❌ Removed from queue
if (not ready yet) {
    // Oops, we removed it but didn't execute it!
    // Need to put it back? Messy!
}

// Better: peek() first
Task task = taskQueue.peek();  // Still in queue ✓
if (ready) {
    taskQueue.poll();  // Only remove when ready
    execute();
}
```

Also important - what happens during the wait?"

**Interviewer**: "What do you mean?"

**Me**: "While waiting, the lock is released! So:

```
Time 0: Next task runs at 10000ms (10 seconds from now)
        Scheduler: await(10000) → releases lock and sleeps

Time 1: Another thread schedules urgent task (runs at 1000ms)
        Can acquire lock ✓ (scheduler released it)
        Adds task
        Signals scheduler
        
Time 2: Scheduler wakes up from signal
        Loops back, sees new urgent task at head
        Dispatches it immediately
```

The `await(timeout)` is interruptible - new tasks can wake it up!"

---

## Step 5: Executing Tasks

**Interviewer**: "What happens in executeTask()? Should we execute with the lock held?"

**Me**: "Absolutely NOT! This is critical:

```java
// ❌ WRONG - Execute while holding lock
private void executeTask(Task task) {
    task.runnable.run();  // Could take seconds/minutes!
    // Lock held entire time - nothing else can schedule!
}

// ✓ CORRECT - Execute outside lock
private void executeTask(Task task) {
    // We're still holding lock here from caller
    // Need to release it before executing
}
```

Let me fix the scheduleLoop:

```java
if (waitTime <= 0) {
    taskQueue.poll();
    
    // Submit to executor OUTSIDE the lock
    lock.unlock();  // Release early
    executorPool.submit(() -> {
        task.runnable.run();
    });
    lock.lock();  // Reacquire for finally block
}
```

Wait, that's awkward. Better approach:

```java
private void scheduleLoop() {
    while (!shutdown) {
        Task taskToExecute = null;
        
        lock.lock();
        try {
            while (taskQueue.isEmpty() && !shutdown) {
                taskAvailable.await();
            }
            
            if (shutdown) break;
            
            Task nextTask = taskQueue.peek();
            long waitTime = nextTask.executeAtMs - System.currentTimeMillis();
            
            if (waitTime <= 0) {
                taskToExecute = taskQueue.poll();
            } else {
                taskAvailable.await(waitTime, TimeUnit.MILLISECONDS);
            }
            
        } finally {
            lock.unlock();
        }
        
        // Execute OUTSIDE the lock
        if (taskToExecute != null) {
            executorPool.submit(taskToExecute.runnable);
        }
    }
}
```

Now the task execution happens outside the lock - scheduler can continue working while tasks run."

---

# CAS and Lock-Free Concepts - Deep Dive

## The Core Question

"If lock-free uses CAS, and CAS can fail and retry, isn't that just as bad as waiting for a lock? What's the real advantage?"

## Analogy 1: The Bathroom

### With Locks (Traditional)

```
Office bathroom with 1 stall, 10 people need to use it

Person 1: Enters, locks door
Person 2: Tries door → LOCKED → sits in waiting room, reads magazine
Person 3: Tries door → LOCKED → sits in waiting room, reads magazine
Person 4-10: All sit in waiting room

Person 1: Finishes, unlocks door, goes to reception desk
Reception: "Next person please!" (calls Person 2)
Person 2: Wakes up from magazine, walks to bathroom, enters

Problem: People 2-10 are SITTING IDLE, doing nothing productive
```

**Key point**: When you can't get the lock, you **go to sleep** (blocked). The operating system puts you in a waiting queue, and you do NOTHING until woken up.

### With CAS (Lock-Free)

```
10 people, 1 door, but NO lock on door

Person 1: Opens door, enters
Person 2: Tries door → Person 1 inside → "OK, I'll check again in 0.0001 seconds"
         (keeps walking around, checks again)
Person 3: Tries door → occupied → "OK, I'll check again in 0.0001 seconds"

Person 1: Exits
Person 2: Tries door → empty! → enters immediately
Person 3: Tries door → Person 2 inside → checks again soon

Problem: Some people make multiple attempts
BUT: Nobody is sitting idle! Everyone keeps trying
```

**Key point**: When CAS fails, you **don't go to sleep**. You immediately try again. No waiting, no context switching.

---

## Analogy 2: The Checkout Line

### With Locks (Single Cashier)

```
Grocery store, 1 cashier, 10 customers

Customer 1: Being served at register
Customers 2-10: Standing in line, WAITING

Timeline:
Minute 1: Customer 1 [██████]
Minute 2: Customer 2          [██████]
Minute 3: Customer 3                   [██████]
...

Total time: 10 minutes (serialized)
```

### With CAS (Self-Checkout Kiosks)

```
Grocery store, 5 self-checkout kiosks, 10 customers

Customer 1: Kiosk 1 [██]
Customer 2: Kiosk 2 [██]
Customer 3: Kiosk 3 [██]
Customer 4: Kiosk 4 [██]
Customer 5: Kiosk 5 [██]
Customer 6: Waits for any kiosk... Kiosk 1 free! [██]
Customer 7: Waits briefly... Kiosk 2 free! [██]
...

Total time: 2 minutes (parallel!)

Yes, customers 6-10 had to wait briefly,
but multiple customers served at once!
```

**Key point**: Lock-free allows **parallelism**. Multiple operations can succeed simultaneously.

---

## What Actually Happens: Operating System Level

### With Locks

```
Thread 1: Tries to acquire lock
  ↓
Lock already held by Thread 2
  ↓
Thread 1: "I need to wait"
  ↓
Operating System: "OK, I'll put you to sleep"
  ↓
1. Remove Thread 1 from CPU
2. Save Thread 1's state (registers, stack pointer, etc.)
3. Put Thread 1 in BLOCKED queue
4. Find another thread to run
5. Load that thread's state
6. Resume that thread

[This is called CONTEXT SWITCH - takes 1-10 microseconds!]

Later...
  ↓
Thread 2: Releases lock
  ↓
Operating System: "Who's waiting? Ah, Thread 1"
  ↓
1. Remove Thread 1 from BLOCKED queue
2. Put Thread 1 in RUNNABLE queue
3. Eventually, schedule Thread 1 on CPU
4. Load Thread 1's state back
5. Thread 1 resumes

[Another CONTEXT SWITCH - another 1-10 microseconds!]
```

**Cost**: ~10-20 microseconds per lock contention (context switches are expensive!)

### With CAS

```
Thread 1: Tries CAS
  ↓
CAS fails (another thread modified the value)
  ↓
Thread 1: "OK, let me try again"
  ↓
1. Re-read the value (from cache or memory)
2. Compute new value
3. Try CAS again

[Thread 1 NEVER leaves the CPU!]
[No context switch!]
[No OS involvement!]

Maybe fails 2-3 times, then succeeds
```

**Cost**: ~0.1-0.5 microseconds per retry (just re-reading and retrying)

**Comparison**:
- Lock contention: 10-20 microseconds (context switch)
- CAS retry: 0.1-0.5 microseconds (just retry)
- **CAS is 20-100× faster!**

---

## Visual: What Happens to Your Thread

### With Lock (Thread Blocks)

```
Thread-1 trying to acquire lock:

CPU Timeline:
[Thread-1 running]
  ↓
  Tries lock → held by Thread-2
  ↓
[Thread-1 BLOCKED - not running, just sitting in queue]
[CPU runs other threads instead]
  ↓
  (waiting... waiting... waiting...)
  ↓
  Lock released by Thread-2
  ↓
  OS wakes up Thread-1
  ↓
[Thread-1 running again]

Time thread was blocked: 5-100+ microseconds
Thread did ZERO work during this time
```

### With CAS (Thread Keeps Running)

```
Thread-1 trying CAS:

CPU Timeline:
[Thread-1 running]
  ↓
  Tries CAS → fails
  ↓
[Thread-1 STILL RUNNING]
  Retry #1 → fails
[Thread-1 STILL RUNNING]
  Retry #2 → fails
[Thread-1 STILL RUNNING]
  Retry #3 → succeeds!
  ↓
[Thread-1 continues]

Time thread was blocked: ZERO
Thread stayed on CPU the whole time
```

**Key insight**: With locks, your thread goes to sleep. With CAS, your thread keeps running.

---

## Why CAS Retries Are Fast

### CAS Retry (Nanoseconds)

```
Retry loop:
1. Read current value from memory/cache     [~5 nanoseconds]
2. Compute new value                        [~1 nanosecond]
3. Execute CAS instruction                  [~10 nanoseconds]
4. Check if succeeded                       [~1 nanosecond]

Total per retry: ~17 nanoseconds

If it fails 5 times:
5 retries × 17 ns = 85 nanoseconds = 0.085 microseconds
```

### Lock Block (Microseconds)

```
Failed lock acquisition:
1. Try to acquire lock                      [~10 nanoseconds]
2. Lock is held, notify OS                  [~100 nanoseconds]
3. Context switch out                       [~5,000 nanoseconds]
4. Wait in queue                            [~10,000+ nanoseconds]
5. Wake up notification                     [~100 nanoseconds]
6. Context switch in                        [~5,000 nanoseconds]

Total: ~20,000 nanoseconds = 20 microseconds (minimum!)
```

**Comparison**:
- 5 CAS retries: 0.085 microseconds
- 1 lock block: 20 microseconds
- **Lock is 235× slower!**

Even if CAS fails 100 times (rare!), it's still faster than blocking once.

---

## The Parallelism Advantage

### With Locks (Serialized)

```
3 threads, 100 available connections

Thread-1: lock.lock() → acquire connection → unlock
          [████████]
Thread-2:              lock.lock() → acquire → unlock
                       [████████]
Thread-3:                           lock.lock() → acquire → unlock
                                    [████████]

Total time: 24 microseconds (3 × 8μs)

Even though 97 connections are available,
only 1 thread can proceed at a time!
```

### With CAS (Parallel)

```
3 threads, 100 available connections

Thread-1: CAS → success [██]
Thread-2: CAS → success [██]  (same time!)
Thread-3: CAS → success [██]  (same time!)

Total time: 2 microseconds (all parallel)

All 3 threads operate simultaneously!
```

**Visual**:

```
Lock-based:
Thread-1: [████████]
Thread-2:           [████████]
Thread-3:                     [████████]
          └─ Serialized (one after another)

Lock-free:
Thread-1: [██]
Thread-2: [██]
Thread-3: [██]
          └─ Parallel (all at once)
```

**This is the real advantage**: Operations that don't actually conflict can proceed in parallel.

---

## Real Example: Connection Pool

### Scenario: 10 threads, 100 connections available

### With Lock

```
Thread-1: Wants connection
  ↓
  lock.lock() [✓ acquired]
  ↓
  connections.remove(conn1) [takes 50ns]
  ↓
  lock.unlock()

Thread-2: Wants connection (while Thread-1 has lock)
  ↓
  lock.lock() [✗ held by Thread-1]
  ↓
  BLOCKED - goes to sleep [20,000ns wasted]
  ↓
  Thread-1 unlocks
  ↓
  OS wakes up Thread-2 [context switch: 5,000ns]
  ↓
  lock.lock() [✓ acquired]
  ↓
  connections.remove(conn2) [50ns]
  ↓
  lock.unlock()

Thread-3-10: All blocked waiting their turn...

Total time for 10 threads: ~250,000 nanoseconds (0.25 milliseconds)
```

### With CAS

```
Thread-1: Wants connection
  ↓
  CAS(remove conn1) [✓ success] [100ns]

Thread-2: Wants connection (at same time!)
  ↓
  CAS(remove conn2) [✓ success] [100ns]

Thread-3: Wants connection (at same time!)
  ↓
  CAS(remove conn3) [maybe fails once] [retry: 20ns]
  ↓
  CAS(remove conn3) [✓ success] [100ns]

Thread-4-10: All succeed in parallel, maybe 1-2 retries each

Total time for 10 threads: ~500 nanoseconds (0.0005 milliseconds)

500× FASTER than locks!
```

**Why so much faster?**
1. No blocking (threads stay on CPU)
2. No context switches (no OS involvement)
3. Parallel execution (multiple succeed simultaneously)

---

## When CAS Can Be Worse

### Bad Case for CAS

```
1000 threads, 1 connection available

Thread-1:   CAS → success! [gets connection]
Thread-2:   CAS → fail (conn taken) → retry → fail → retry → fail...
Thread-3:   CAS → fail → retry → fail → retry → fail...
Thread-4:   CAS → fail → retry → fail → retry → fail...
...
Thread-1000: CAS → fail → retry → fail → retry → fail...

All 999 threads spinning, wasting CPU!

Eventually Thread-1 returns connection:
Thread-2: CAS → success!
Thread-3-1000: Still spinning...

Wasted CPU cycles on 999 threads!
```

**With locks in this case**:
```
Thread-1: lock → gets connection
Thread-2-1000: BLOCKED (sleeping, not wasting CPU)

Thread-1 returns connection
OS wakes up Thread-2 (only one thread)
Thread-2: gets connection
Thread-3-1000: Still sleeping

Only 1-2 threads active at a time
```

**When locks are better**:
- Resource almost always exhausted (pool usually empty)
- Extreme contention (100+ threads on 1 resource)
- Want to block/sleep (not waste CPU)

**When CAS is better**:
- Resource usually available (pool has items)
- Moderate contention (2-20 threads)
- Want maximum throughput
- Operations are fast

---

## Simple Mental Model

### Locks = Taking Turns

```
Like a single-lane road with traffic light

Car 1: [green light] → drives through
Cars 2-10: [red light] → STOP and WAIT

Everyone takes turns, one at a time
Safe, orderly, but SLOW
```

### CAS = Multi-Lane Highway

```
Like a 5-lane highway

Car 1: Lane 1 → drives
Car 2: Lane 2 → drives (at same time!)
Car 3: Lane 3 → drives (at same time!)
Car 4: Lane 4 → drives (at same time!)
Car 5: Lane 5 → drives (at same time!)
Car 6: Looks for open lane... Lane 1 free! → drives

Some cars might need to change lanes (retry)
But multiple cars drive at once
MUCH FASTER overall
```

---

## The Bottom Line

### Locks:
```
✗ Thread blocks (goes to sleep)
✗ Context switch (expensive: ~20μs)
✗ OS involvement (kernel calls)
✗ Serialized (one at a time)
✓ Efficient when waiting long time
✓ Don't waste CPU when blocked
```

### CAS (Lock-Free):
```
✓ Thread never blocks (stays running)
✓ No context switch (fast: ~0.1μs per retry)
✓ No OS involvement (pure CPU operation)
✓ Parallel (multiple succeed at once)
✗ Wastes CPU if too much contention
✗ Can livelock in extreme cases
```

### Why CAS is Usually Faster:

1. **No blocking** = no sleeping = no context switches
   - Context switch: 20 microseconds
   - CAS retry: 0.1 microseconds
   - **200× faster per retry**

2. **Parallelism** = multiple operations succeed simultaneously
   - Locks: One at a time
   - CAS: Many at once
   - **N× faster (N = number of threads)**

3. **Stay on CPU** = no OS overhead
   - Locks: Kernel calls, scheduling, queues
   - CAS: Pure CPU instruction
   - **100× less overhead**

### The Math:

```
10 threads acquiring connections:

Lock-based:
- Each thread: 20μs context switch
- Serialized: 10 × 20μs = 200μs
- Total: 200 microseconds

CAS-based:
- Each thread: 0.1μs × 3 retries = 0.3μs
- Parallel: max(0.3μs) = 0.3μs
- Total: 0.3 microseconds

Speedup: 200 / 0.3 = 666× faster!
```

---

# Complete List of Concurrency Patterns

## Pattern 1: Producer-Consumer Queue

**Description**: Items flow through a queue from producers to consumers

**Structure**:
```
Producers → [Queue] → Consumers
```

**Key characteristics**:
- Work items/tasks added to queue
- Consumers process items
- Decoupling between production and consumption
- Usually need blocking when empty/full

**Problems using this pattern**:
1. ✓ **Batch Processor with Auto-Flush**
2. ✓ **Priority Task Scheduler**
3. ✓ **Multi-Stage Pipeline** (multiple queues)
4. ✓ **Producer-Consumer with Multiple Queues**
5. ✓ **Thread Pool with Dynamic Sizing**
6. ✓ **Async Log Writer**

**Data structures**:
```java
// Lock-based
LinkedBlockingQueue<Task>
ArrayBlockingQueue<Task>

// Lock-free
ConcurrentLinkedQueue<Task>

// Priority
PriorityBlockingQueue<Task>
```

**Template code**:
```java
class ProducerConsumer<T> {
    private final BlockingQueue<T> queue;
    
    void produce(T item) {
        queue.put(item);  // Blocks if full
    }
    
    void consume() {
        T item = queue.take();  // Blocks if empty
        process(item);
    }
}
```

---

## Pattern 2: Resource Pool

**Description**: Fixed set of reusable resources, borrow and return

**Structure**:
```
Available Resources ⇄ In-Use Resources
```

**Key characteristics**:
- Resources are reused, not consumed
- Circular flow: acquire → use → release
- Usually fixed pool size
- Need blocking when exhausted

**Problems using this pattern**:
1. ✓ **Connection Pool Manager**
2. ✓ **Thread Pool with Dynamic Sizing** (hybrid with queue)
3. Object pool (generic)
4. Memory pool

**Data structures**:
```java
Queue<Resource> available;      // ConcurrentLinkedQueue
Set<Resource> inUse;            // ConcurrentHashMap.newKeySet()
AtomicInteger availableCount;
Lock + Condition for blocking
```

**Template code**:
```java
class ResourcePool<R> {
    private final ConcurrentLinkedQueue<R> available;
    private final Set<R> inUse;
    private final Lock lock;
    private final Condition notEmpty;
    
    R acquire() throws InterruptedException {
        // Fast path: lock-free
        R resource = available.poll();
        if (resource != null) {
            inUse.add(resource);
            return resource;
        }
        
        // Slow path: blocking
        lock.lock();
        try {
            while (available.isEmpty()) {
                notEmpty.await();
            }
            resource = available.poll();
            inUse.add(resource);
            return resource;
        } finally {
            lock.unlock();
        }
    }
    
    void release(R resource) {
        inUse.remove(resource);
        available.offer(resource);
        
        lock.lock();
        try {
            notEmpty.signal();
        } finally {
            lock.unlock();
        }
    }
}
```

---

## Pattern 3: Cache/Map with Eviction

**Description**: Key-value storage with eviction policies

**Structure**:
```
Key → Value (with metadata: access time, TTL, frequency)
+ Eviction policy (LRU, LFU, TTL)
```

**Key characteristics**:
- Random access by key
- In-place updates
- Eviction based on policy
- Often read-heavy

**Problems using this pattern**:
1. ✓ **TTL Cache with Size Limits**
2. ✓ **Write-Behind Cache**
3. ✓ **Multi-Level Cache**
4. LRU Cache
5. Session cache
6. Memoization cache

**Data structures**:
```java
ConcurrentHashMap<K, CacheEntry<V>>
LinkedHashMap (for LRU ordering)
ReadWriteLock (if custom eviction)
```

**Template code**:
```java
class TTLCache<K, V> {
    private final ConcurrentHashMap<K, CacheEntry<V>> cache;
    
    static class CacheEntry<V> {
        final V value;
        final long expiryTime;
        volatile long lastAccess;
    }
    
    V get(K key) {
        CacheEntry<V> entry = cache.get(key);
        if (entry != null && !entry.isExpired()) {
            entry.lastAccess = System.currentTimeMillis();
            return entry.value;
        }
        cache.remove(key);  // Expired
        return null;
    }
    
    void put(K key, V value, long ttlMs) {
        cache.put(key, new CacheEntry<>(value, 
            System.currentTimeMillis() + ttlMs));
        evictIfNeeded();
    }
}
```

---

## Pattern 4: Rate Limiting / Token Bucket

**Description**: Control rate of operations using tokens

**Structure**:
```
Token Bucket → Tokens refilled periodically
Request → Consumes token if available
```

**Key characteristics**:
- Counter-based (tokens available)
- Time-based refilling
- Allow/deny decisions
- No queue of work

**Problems using this pattern**:
1. ✓ **Rate Limiter**
2. ✓ **Semaphore-Based Rate Limiter**
3. API throttling
4. Request quota management
5. Traffic shaping

**Data structures**:
```java
AtomicInteger tokens;
AtomicLong lastRefillTime;
Lock for refill coordination
```

**Template code**:
```java
class RateLimiter {
    private final AtomicInteger tokens;
    private final AtomicLong lastRefill;
    private final int maxTokens;
    private final long refillIntervalMs;
    
    boolean tryAcquire() {
        refillTokens();
        
        // CAS loop to consume token
        while (true) {
            int current = tokens.get();
            if (current <= 0) {
                return false;  // Rate limited
            }
            if (tokens.compareAndSet(current, current - 1)) {
                return true;  // Allowed
            }
        }
    }
    
    private void refillTokens() {
        long now = System.currentTimeMillis();
        long last = lastRefill.get();
        
        if (now - last > refillIntervalMs) {
            if (lastRefill.compareAndSet(last, now)) {
                tokens.set(maxTokens);
            }
        }
    }
}
```

---

## Pattern 5: Metrics/Statistics Aggregation

**Description**: Collect and aggregate data from multiple threads

**Structure**:
```
Multiple threads → Update counters/histograms → Read aggregated results
```

**Key characteristics**:
- Many writers, few readers
- Aggregate operations (sum, average, percentiles)
- High write frequency
- Eventually consistent reads OK

**Problems using this pattern**:
1. ✓ **Real-Time Metrics Aggregator**
2. ✓ **Event Counter with Time Windows**
3. Performance monitoring
4. Request statistics
5. System health metrics

**Data structures**:
```java
LongAdder (high contention counters)
AtomicLong (low/medium contention)
ConcurrentHashMap (per-metric storage)
```

**Template code**:
```java
class MetricsAggregator {
    private final LongAdder requestCount = new LongAdder();
    private final LongAdder errorCount = new LongAdder();
    private final LongAdder totalLatency = new LongAdder();
    
    // High-frequency updates
    void recordRequest(long latencyMs, boolean error) {
        requestCount.increment();
        totalLatency.add(latencyMs);
        if (error) {
            errorCount.increment();
        }
    }
    
    // Infrequent reads
    Metrics getMetrics() {
        long requests = requestCount.sum();
        long errors = errorCount.sum();
        long latency = totalLatency.sum();
        
        return new Metrics(
            requests,
            errors,
            requests > 0 ? latency / requests : 0  // avg latency
        );
    }
}
```

---

## Pattern 6: Deduplication / Set Membership

**Description**: Track unique items, detect duplicates

**Structure**:
```
Item → Check if seen → Add if new → Process if new
```

**Key characteristics**:
- Set-based membership testing
- Add if absent
- Often used in crawlers, stream processing

**Problems using this pattern**:
1. ✓ **Concurrent Data Deduplicator**
2. Web crawler (visited URLs)
3. Event deduplication
4. Unique visitor tracking

**Data structures**:
```java
ConcurrentHashMap.newKeySet()
Bloom filter + set (for very large scale)
```

**Template code**:
```java
class Deduplicator<T> {
    private final Set<T> seen = ConcurrentHashMap.newKeySet();
    
    boolean processIfNew(T item) {
        // Returns true if item was added (not present before)
        if (seen.add(item)) {
            process(item);
            return true;
        }
        return false;  // Duplicate
    }
}
```

---

## Pattern 7: Read-Write Split

**Description**: Optimize for read-heavy workloads with separate read/write handling

**Structure**:
```
Many Readers (concurrent) + Few Writers (exclusive)
```

**Key characteristics**:
- Many concurrent readers
- Writers need exclusive access
- Read-heavy workload (90%+ reads)
- Use ReadWriteLock

**Problems using this pattern**:
1. ✓ **Reader-Writer Lock for Config**
2. Configuration management
3. Read-mostly cache
4. Feature flags
5. Routing tables

**Data structures**:
```java
ReadWriteLock
ReentrantReadWriteLock
StampedLock (optimistic reads)
```

**Template code**:
```java
class ConfigManager {
    private Config config;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    
    // Many threads can read simultaneously
    Config getConfig() {
        lock.readLock().lock();
        try {
            return config;  // Read-only, no copy needed
        } finally {
            lock.readLock().unlock();
        }
    }
    
    // Only one writer at a time
    void updateConfig(Config newConfig) {
        lock.writeLock().lock();
        try {
            this.config = newConfig;
        } finally {
            lock.writeLock().unlock();
        }
    }
}
```

---

## Pattern 8: Barrier/Phaser (Wait for All)

**Description**: Multiple threads wait for each other at synchronization point

**Structure**:
```
Thread-1 ──┐
Thread-2 ──┤─→ [Barrier] ─→ All continue together
Thread-3 ──┘
```

**Key characteristics**:
- Fixed or dynamic number of parties
- All threads wait until all arrive
- Multi-phase computation
- Cyclic (can reuse)

**Problems using this pattern**:
1. ✓ **Barrier for Parallel Processing**
2. MapReduce (wait for all mappers before reduce)
3. Parallel matrix computation
4. Multi-phase algorithms
5. Coordinated batch processing

**Data structures**:
```java
CyclicBarrier
Phaser (more flexible)
CountDownLatch (one-time)
```

**Template code**:
```java
// CyclicBarrier example
class ParallelProcessor {
    private final CyclicBarrier barrier;
    
    public ParallelProcessor(int numThreads) {
        this.barrier = new CyclicBarrier(numThreads, () -> {
            System.out.println("All threads reached barrier!");
        });
    }
    
    void processPhase() throws Exception {
        // Phase 1: Each thread does work
        doWork();
        
        // Wait for all threads
        barrier.await();
        
        // Phase 2: All threads continue together
        doMoreWork();
    }
}
```

---

## Pattern 9: Fork-Join / Divide-and-Conquer

**Description**: Recursively divide work, process in parallel, merge results

**Structure**:
```
        [Task]
       /      \
   [Sub-1]  [Sub-2]
    /  \      /  \
  [..][..]  [..][..]
       \      /
    [Merge Results]
```

**Key characteristics**:
- Recursive task decomposition
- Work-stealing for load balancing
- Merge/combine results
- CPU-intensive parallel computation

**Problems using this pattern**:
1. ✓ **Concurrent Result Aggregator** (when results are independent)
2. Parallel sorting (merge sort, quicksort)
3. Parallel tree traversal
4. Parallel array processing
5. Recursive algorithms

**Data structures**:
```java
ForkJoinPool
RecursiveTask<T>
RecursiveAction
```

**Template code**:
```java
class ParallelSum extends RecursiveTask<Long> {
    private final int[] array;
    private final int start, end;
    private static final int THRESHOLD = 1000;
    
    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // Base case: compute directly
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += array[i];
            }
            return sum;
        } else {
            // Recursive case: split
            int mid = (start + end) / 2;
            
            ParallelSum left = new ParallelSum(array, start, mid);
            ParallelSum right = new ParallelSum(array, mid, end);
            
            left.fork();   // Async execute left
            long rightResult = right.compute();  // Execute right
            long leftResult = left.join();       // Wait for left
            
            return leftResult + rightResult;
        }
    }
}
```

---

## Pattern 10: Master-Worker

**Description**: One coordinator distributes work to many workers

**Structure**:
```
         [Master]
        /   |   \
    [W-1] [W-2] [W-3]
```

**Key characteristics**:
- Central coordinator
- Workers are independent
- Master assigns work
- Workers report results

**Problems using this pattern**:
1. ✓ **Threa
