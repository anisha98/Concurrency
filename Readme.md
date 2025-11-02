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
1. ✓ **Thread Pool with Dynamic Sizing**
2. Distributed computation coordinator
3. Load balancer
4. Job scheduler

**Data structures**:
```java
BlockingQueue<Task> (work queue)
ExecutorService (worker pool)
CompletionService (for results)
```

**Template code**:
```java
class MasterWorker {
    private final ExecutorService workers;
    private final BlockingQueue<Task> workQueue;
    private final CompletionService<Result> completionService;
    
    public MasterWorker(int numWorkers) {
        this.workers = Executors.newFixedThreadPool(numWorkers);
        this.workQueue = new LinkedBlockingQueue<>();
        this.completionService = new ExecutorCompletionService<>(workers);
    }
    
    // Master: distribute work
    void submitTasks(List<Task> tasks) {
        for (Task task : tasks) {
            completionService.submit(() -> processTask(task));
        }
    }
    
    // Master: collect results
    List<Result> collectResults(int numTasks) throws Exception {
        List<Result> results = new ArrayList<>();
        for (int i = 0; i < numTasks; i++) {
            results.add(completionService.take().get());
        }
        return results;
    }
}
```

---

## Pattern 11: Copy-on-Write / Immutable State

**Description**: Create new version on modification, readers see consistent snapshot

**Structure**:
```
Thread-1 (reader) → sees version 1
Thread-2 (writer) → creates version 2
Thread-3 (reader) → sees version 1 (still)
Thread-4 (reader) → sees version 2 (after switch)
```

**Key characteristics**:
- Immutable objects
- Atomic reference swap
- No locks for readers
- Copy entire structure on write

**Problems using this pattern**:
1. Configuration updates
2. Read-heavy lists/sets
3. Event listeners
4. Snapshot isolation

**Data structures**:
```java
AtomicReference<ImmutableState>
CopyOnWriteArrayList
CopyOnWriteArraySet
```

**Template code**:
```java
class ImmutableConfig {
    private final AtomicReference<Config> configRef;
    
    // Read (no lock!)
    Config getConfig() {
        return configRef.get();  // Atomic read
    }
    
    // Write (creates new copy)
    void updateConfig(Function<Config, Config> updater) {
        configRef.updateAndGet(old -> {
            // Create new immutable config
            return updater.apply(old);
        });
    }
}

// CopyOnWriteArrayList example
class EventBus {
    private final CopyOnWriteArrayList<Listener> listeners = 
        new CopyOnWriteArrayList<>();
    
    void register(Listener listener) {
        listeners.add(listener);  // Creates new copy
    }
    
    void fireEvent(Event event) {
        // Iteration sees consistent snapshot, no ConcurrentModificationException
        for (Listener listener : listeners) {
            listener.onEvent(event);
        }
    }
}
```

**When to use**:
- ✓ Read-heavy (99%+ reads)
- ✓ Small collections
- ✗ Write-heavy (copying is expensive)
- ✗ Large collections (memory overhead)

---

## Pattern 12: Coordination / Lock Management

**Description**: Manage locks on multiple resources, prevent deadlock

**Structure**:
```
Resources: [R1, R2, R3]
Lock graph: Track who holds what, who waits for what
Deadlock detection: Cycle detection in wait graph
```

**Key characteristics**:
- Multiple resource locks
- Deadlock prevention/detection
- Lock ordering
- Timeout-based acquisition

**Problems using this pattern**:
1. ✓ **Distributed Lock Manager**
2. Database transaction manager
3. Resource allocation
4. Dining philosophers

**Data structures**:
```java
ConcurrentHashMap<Resource, LockInfo>
Graph structure for deadlock detection
```

---

## Pattern Mapping to Problems

| Problem | Primary Pattern | Secondary Pattern |
|---------|----------------|-------------------|
| **Batch Processor with Auto-Flush** | Producer-Consumer | Rate Limiting (time-based) |
| **Connection Pool Manager** | Resource Pool | - |
| **TTL Cache with Size Limits** | Cache/Map | - |
| **Priority Task Scheduler** | Producer-Consumer | - |
| **Rate Limiter** | Rate Limiting | - |
| **Multi-Stage Pipeline** | Producer-Consumer | Pipeline (multiple queues) |
| **Write-Behind Cache** | Cache/Map | Producer-Consumer (write queue) |
| **Distributed Lock Manager** | Coordination | - |
| **Real-Time Metrics Aggregator** | Metrics/Statistics | - |
| **Producer-Consumer Multi-Queue** | Producer-Consumer | Master-Worker (routing) |
| **Concurrent Data Deduplicator** | Deduplication | - |
| **Multi-Level Cache** | Cache/Map | Read-Write Split |
| **Thread Pool Dynamic Sizing** | Master-Worker | Resource Pool |
| **Concurrent Result Aggregator** | Fork-Join | Barrier (if waiting for all) |
| **Reader-Writer Config** | Read-Write Split | Copy-on-Write |
| **Barrier for Parallel Processing** | Barrier/Phaser | - |

---

## Pattern Selection Flowchart

```
What are you building?

├─ Work flows through system?
│  └─ Producer-Consumer Queue
│
├─ Reusable resources (borrow/return)?
│  └─ Resource Pool
│
├─ Key-value storage?
│  └─ Cache/Map
│
├─ Controlling rate of operations?
│  └─ Rate Limiting
│
├─ Collecting statistics?
│  └─ Metrics/Statistics
│
├─ Tracking unique items?
│  └─ Deduplication
│
├─ Many readers, few writers?
│  └─ Read-Write Split or Copy-on-Write
│
├─ Wait for multiple threads?
│  └─ Barrier/Phaser
│
├─ Divide work recursively?
│  └─ Fork-Join
│
├─ Central coordinator + workers?
│  └─ Master-Worker
│
└─ Managing multiple locks?
   └─ Coordination/Lock Management
```

---

## Which Patterns Need Lock-Free?

| Pattern | Lock-Free Helps? | Why |
|---------|------------------|-----|
| Producer-Consumer | Medium | Queue operations benefit, but need blocking when empty |
| Resource Pool | Medium | Fast path yes, but need blocking when exhausted |
| Cache/Map | Yes | `ConcurrentHashMap` is perfect |
| Rate Limiting | Yes | `AtomicInteger` for token counter |
| Metrics/Statistics | Yes | `LongAdder` for high-contention counters |
| Deduplication | Yes | `ConcurrentHashMap.newKeySet()` |
| Read-Write Split | No | `ReadWriteLock` is the point |
| Barrier/Phaser | No | Need coordination, locks appropriate |
| Fork-Join | Yes | `ForkJoinPool` uses work-stealing (lock-free) |
| Master-Worker | Medium | Work queue can be lock-free |
| Copy-on-Write | Yes | `AtomicReference` + immutable state |
| Coordination | No | Managing locks, locks appropriate |

---

# Lock-Free Data Structures Reference

## Java Built-In Lock-Free Data Structures

### Atomic Variables (java.util.concurrent.atomic)

```java
// Single values
AtomicBoolean     // boolean
AtomicInteger     // int
AtomicLong        // long
AtomicReference<T> // any object reference

// Arrays
AtomicIntegerArray
AtomicLongArray
AtomicReferenceArray<T>

// Field updaters (advanced)
AtomicIntegerFieldUpdater
AtomicLongFieldUpdater
AtomicReferenceFieldUpdater

// Special purpose
LongAdder         // Better than AtomicLong for high contention
LongAccumulator   // Customizable accumulation
DoubleAdder       // For doubles
DoubleAccumulator
```

### AtomicInteger / AtomicLong

**Use for:** Counters, IDs, statistics

```java
AtomicInteger counter = new AtomicInteger(0);

// Read
int value = counter.get();

// Write
counter.set(10);

// Increment/Decrement
int newValue = counter.incrementAndGet();  // ++i
int oldValue = counter.getAndIncrement();  // i++
counter.decrementAndGet();                 // --i
counter.getAndDecrement();                 // i--

// Add
counter.addAndGet(5);    // i += 5, return new value
counter.getAndAdd(5);    // i += 5, return old value

// CAS
boolean success = counter.compareAndSet(5, 10);  // if (i == 5) i = 10

// Update with function
counter.updateAndGet(x -> x * 2);  // i = i * 2
counter.accumulateAndGet(5, (x, y) -> x + y);  // i = i + 5
```

**When to use:**
- ✓ Simple counters (pages crawled, requests processed)
- ✓ Sequence generators (ID generation)
- ✓ Statistics (success/failure counts)
- ✓ Flags with numeric meaning

---

### AtomicBoolean

**Use for:** Flags, states, one-time initialization

```java
AtomicBoolean flag = new AtomicBoolean(false);

// Read/Write
boolean value = flag.get();
flag.set(true);

// CAS (most useful)
boolean wasSet = flag.compareAndSet(false, true);

// Get and set
boolean old = flag.getAndSet(true);
```

**When to use:**
- ✓ Shutdown flags
- ✓ One-time initialization
- ✓ Simple state toggles
- ✓ Circuit breaker states

---

### AtomicReference<T>

**Use for:** Object references, complex state

```java
AtomicReference<String> ref = new AtomicReference<>("initial");

// Read/Write
String value = ref.get();
ref.set("new value");

// CAS
boolean success = ref.compareAndSet("old", "new");

// Update with function
ref.updateAndGet(old -> old + " modified");

// Get and update
String old = ref.getAndUpdate(x -> x + " suffix");
```

**When to use:**
- ✓ Configuration objects (swap entire config atomically)
- ✓ Immutable state updates
- ✓ Head/tail pointers in lock-free data structures
- ✓ Cached computed values

**Important caveat - ABA problem:**
```java
// Problem scenario
AtomicReference<Node> head = new AtomicReference<>(nodeA);

// Thread 1: Read head (gets A)
Node old = head.get();  // A

// Thread 2: Remove A, remove B, add A back
head.set(nodeB);
head.set(nodeA);  // Same object!

// Thread 1: CAS succeeds but A might be in inconsistent state
head.compareAndSet(old, newNode);  // Succeeds! (But dangerous)

// Solution: Use AtomicStampedReference
AtomicStampedReference<Node> head = new AtomicStampedReference<>(nodeA, 0);

int[] stampHolder = new int[1];
Node old = head.get(stampHolder);
int oldStamp = stampHolder[0];

// CAS with stamp - fails if stamp changed
head.compareAndSet(old, newNode, oldStamp, oldStamp + 1);
```

---

### LongAdder / DoubleAdder

**Use for:** High-contention counters

```java
// Replace this (under HIGH contention):
AtomicLong counter = new AtomicLong();
counter.incrementAndGet();  // Many CAS retries under contention

// With this:
LongAdder counter = new LongAdder();
counter.increment();  // Distributes contention across cells

// Read (slightly more expensive)
long total = counter.sum();
```

**How it works:**
```
AtomicLong (single variable):
Thread-1: [CAS on same variable]
Thread-2: [CAS on same variable] → retry
Thread-3: [CAS on same variable] → retry
Thread-4: [CAS on same variable] → retry

LongAdder (multiple cells):
Thread-1: [CAS on cell-0] ✓
Thread-2: [CAS on cell-1] ✓ (different cell!)
Thread-3: [CAS on cell-2] ✓
Thread-4: [CAS on cell-3] ✓

When reading: sum all cells
```

**When to use:**
- ✓ Very high contention (16+ threads)
- ✓ Write-heavy workloads (many increments)
- ✓ Don't need exact value frequently
- ✗ Don't use if you need to read often (sum is expensive)

---

### ConcurrentHashMap<K, V>

**Use for:** Thread-safe key-value storage

```java
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

// Basic operations (lock-free for different keys)
map.put("key", 1);
Integer value = map.get("key");
map.remove("key");

// Atomic operations (very useful!)
map.putIfAbsent("key", 1);  // Only if not present

map.computeIfAbsent("key", k -> expensiveComputation(k));  // Lazy init

map.computeIfPresent("key", (k, v) -> v + 1);  // Update if exists

map.compute("key", (k, v) -> v == null ? 1 : v + 1);  // Upsert

map.merge("key", 1, Integer::sum);  // Increment or set to 1

// Replace operations
map.replace("key", 1, 2);  // CAS: if value is 1, set to 2

// Bulk operations (parallel internally)
map.forEach((k, v) -> process(k, v));
map.search(Long.MAX_VALUE, (k, v) -> v > 100 ? k : null);
Integer sum = map.reduceValues(Long.MAX_VALUE, Integer::sum);
```

**When to use:**
- ✓ Shared caches
- ✓ Visited sets (deduplication)
- ✓ Frequency maps
- ✓ Any concurrent map needs

**Special feature - KeySet as Set:**
```java
// Lock-free concurrent set!
Set<String> set = ConcurrentHashMap.newKeySet();

set.add("item");  // Lock-free!
boolean isNew = set.add("item");  // Returns false if duplicate

// Perfect for:
Set<String> visitedUrls = ConcurrentHashMap.newKeySet();
if (visitedUrls.add(url)) {
    // New URL, crawl it
}
```

---

### ConcurrentLinkedQueue<E>

**Use for:** Unbounded lock-free queue

```java
ConcurrentLinkedQueue<String> queue = new ConcurrentLinkedQueue<>();

// Add (always succeeds, unbounded)
queue.offer("item");  // Lock-free CAS
queue.add("item");    // Same as offer

// Remove
String item = queue.poll();  // null if empty, lock-free CAS

// Peek
String head = queue.peek();  // Don't remove

// Check
boolean empty = queue.isEmpty();
int size = queue.size();  // O(n)! Not constant time
```

**When to use:**
- ✓ Task queues (unbounded)
- ✓ Event queues
- ✓ Connection pools (available connections)
- ✓ High-throughput producer-consumer

**When NOT to use:**
- ✗ Need blocking when empty (use `BlockingQueue`)
- ✗ Need bounded size (use `ArrayBlockingQueue`)
- ✗ Need frequent size checks (size is O(n))

---

### ConcurrentLinkedDeque<E>

**Use for:** Lock-free double-ended queue

```java
ConcurrentLinkedDeque<String> deque = new ConcurrentLinkedDeque<>();

// Add to front
deque.offerFirst("first");
deque.addFirst("first");

// Add to back
deque.offerLast("last");
deque.addLast("last");

// Remove from front
String first = deque.pollFirst();

// Remove from back
String last = deque.pollLast();
```

**When to use:**
- ✓ Work-stealing queues
- ✓ LRU cache (add to front, remove from back)
- ✓ Undo/redo stacks

---

### ConcurrentSkipListMap<K, V> / ConcurrentSkipListSet<E>

**Use for:** Sorted concurrent map/set

```java
// Replace TreeMap + lock with:
ConcurrentSkipListMap<Integer, String> map = new ConcurrentSkipListMap<>();

map.put(1, "one");
map.put(3, "three");
map.put(2, "two");

// Sorted operations
String first = map.firstEntry().getValue();  // "one"
String last = map.lastEntry().getValue();    // "three"

// Range queries
SortedMap<Integer, String> subMap = map.subMap(1, 3);

// For set:
ConcurrentSkipListSet<Integer> set = new ConcurrentSkipListSet<>();
set.add(3);
set.add(1);
set.add(2);
Integer first = set.first();  // 1 (sorted!)
```

**When to use:**
- ✓ Need sorted data structure
- ✓ Priority queues (but see `PriorityBlockingQueue`)
- ✓ Range queries
- ✓ Task scheduler (sorted by time)

---

### Blocking Collections (Hybrid: Lock-Free + Locks)

```java
// ArrayBlockingQueue - bounded, array-based
BlockingQueue<String> queue = new ArrayBlockingQueue<>(100);
queue.put("item");  // Blocks if full
String item = queue.take();  // Blocks if empty

// LinkedBlockingQueue - optionally bounded, linked nodes
BlockingQueue<String> queue = new LinkedBlockingQueue<>(100);

// PriorityBlockingQueue - unbounded, sorted
BlockingQueue<Task> queue = new PriorityBlockingQueue<>();

// SynchronousQueue - no storage, direct handoff
BlockingQueue<String> queue = new SynchronousQueue<>();
queue.put("item");  // Blocks until another thread takes it

// DelayQueue - elements available after delay
DelayQueue<DelayedTask> queue = new DelayQueue<>();

// LinkedTransferQueue - performance optimized
TransferQueue<String> queue = new LinkedTransferQueue<>();
```

---

## Complete Replacement Guide

### Counters and Flags

```java
// Simple counter
int count;                    → AtomicInteger
long count;                   → AtomicLong

// High-contention counter
AtomicLong counter;           → LongAdder

// Boolean flag
boolean flag;                 → AtomicBoolean

// Floating point counter
double sum;                   → DoubleAdder
```

### Collections

```java
// Map
HashMap + Lock                → ConcurrentHashMap
TreeMap + Lock                → ConcurrentSkipListMap

// Set
HashSet + Lock                → ConcurrentHashMap.newKeySet()
TreeSet + Lock                → ConcurrentSkipListSet

// Queue
LinkedList + Lock             → ConcurrentLinkedQueue
ArrayDeque + Lock             → ConcurrentLinkedDeque

// Priority Queue
PriorityQueue + Lock          → PriorityBlockingQueue
                              → ConcurrentSkipListSet (if don't need blocking)

// Blocking Queue
LinkedList + Lock + Condition → LinkedBlockingQueue
Array + Lock + Condition      → ArrayBlockingQueue
```

### Object References

```java
// Simple reference
T reference;                  → AtomicReference<T>

// With version/stamp (ABA protection)
T reference + int version;   → AtomicStampedReference<T>

// With boolean mark
T reference + boolean mark;   → AtomicMarkableReference<T>

// Array of references
T[] array;                    → AtomicReferenceArray<T>
```

---

## Quick Decision Matrix

| Need | Use This | Why |
|------|----------|-----|
| **Counter** | `AtomicInteger/Long` | Simple, fast |
| **High-contention counter** | `LongAdder` | Distributes contention |
| **Flag** | `AtomicBoolean` | Simple state |
| **Object reference** | `AtomicReference<T>` | Immutable updates |
| **Map** | `ConcurrentHashMap` | Most common, great performance |
| **Set** | `ConcurrentHashMap.newKeySet()` | Built on ConcurrentHashMap |
| **Queue** | `ConcurrentLinkedQueue` | Unbounded, lock-free |
| **Deque** | `ConcurrentLinkedDeque` | Double-ended |
| **Sorted map** | `ConcurrentSkipListMap` | Sorted access |
| **Sorted set** | `ConcurrentSkipListSet` | Sorted access |
| **Blocking queue** | `LinkedBlockingQueue` | Need blocking |
| **Bounded queue** | `ArrayBlockingQueue` | Fixed capacity |
| **Priority queue** | `PriorityBlockingQueue` | Sorted + blocking |

---

## Common Patterns in Interviews

### Pattern 1: Visited Set
```java
Set<String> visited = ConcurrentHashMap.newKeySet();
if (visited.add(url)) {
    // New URL, process it
}
```

### Pattern 2: Counter
```java
AtomicInteger counter = new AtomicInteger(0);
counter.incrementAndGet();
```

### Pattern 3: Cache
```java
ConcurrentHashMap<K, V> cache = new ConcurrentHashMap<>();
cache.computeIfAbsent(key, k -> expensiveCompute(k));
```

### Pattern 4: Task Queue
```java
ConcurrentLinkedQueue<Task> tasks = new ConcurrentLinkedQueue<>();
tasks.offer(task);  // Add
Task t = tasks.poll();  // Remove
```

### Pattern 5: Configuration
```java
AtomicReference<Config> config = new AtomicReference<>(initialConfig);
config.updateAndGet(old -> new Config(old, newSettings));
```

### Pattern 6: Rate Limiter
```java
AtomicInteger tokens = new AtomicInteger(MAX_TOKENS);
if (tokens.getAndDecrement() > 0) {
    // Allow request
} else {
    tokens.incrementAndGet();  // Restore token
    // Deny request
}
```

---

## Summary Cheat Sheet

```java
// MOST COMMON (memorize these)
AtomicInteger               // Counters
AtomicLong                  // Large counters
AtomicBoolean               // Flags
AtomicReference<T>          // Object references
ConcurrentHashMap<K,V>      // Maps
ConcurrentHashMap.newKeySet() // Sets
ConcurrentLinkedQueue<E>    // Queues
LongAdder                   // High-contention counters

// LESS COMMON (know they exist)
ConcurrentSkipListMap<K,V>  // Sorted map
ConcurrentSkipListSet<E>    // Sorted set
ConcurrentLinkedDeque<E>    // Double-ended queue
LongAccumulator             // Custom accumulation

// BLOCKING (hybrid, not pure lock-free)
LinkedBlockingQueue<E>      // Bounded queue with blocking
ArrayBlockingQueue<E>       // Array-based bounded queue
PriorityBlockingQueue<E>    // Priority + blocking
```

---

# Distributed Lock Manager - Interview Deep Dive

## Problem Introduction

**Interviewer**: "Design a distributed lock manager. Multiple threads need to acquire locks on different resources. Ensure no deadlocks and handle concurrent access safely."

**Me**: "Great! Let me clarify the requirements:

**Basic scenario**:
```
Thread-1 needs: [Database, FileSystem]
Thread-2 needs: [Database, Network]
Thread-3 needs: [FileSystem, Network]

All trying to acquire at the same time - how do we prevent deadlock?
```

**Questions**:
1. **Single vs Multiple Resources**: Can a thread acquire multiple locks at once?
2. **Deadlock Handling**: Prevent or detect deadlocks?
3. **Fairness**: Should locks be granted in FIFO order?
4. **Timeout**: Should lock acquisition have a timeout?
5. **Reentrancy**: Can the same thread acquire the same lock multiple times?
6. **Distributed**: Is this actually distributed (across machines) or just multi-threaded?

Let me assume:
- Multiple locks per thread (the hard part!)
- Prevent deadlocks (better than detecting)
- Timeouts supported
- Single machine (multi-threaded, not truly distributed)
- Non-reentrant for simplicity

Does this sound right?"

**Interviewer**: "Yes, focus on preventing deadlocks with multiple locks per thread. Start with a basic design."

---

## Understanding the Deadlock Problem

**Me**: "First, let me explain why this is hard. The classic deadlock scenario:

```
Time 0:
Thread-1: Acquires Lock-A, wants Lock-B
Thread-2: Acquires Lock-B, wants Lock-A

Thread-1: Holds A, waiting for B ─┐
                                   ├─→ DEADLOCK!
Thread-2: Holds B, waiting for A ─┘

Both threads wait forever!
```

**The four conditions for deadlock** (Coffman conditions):
1. **Mutual Exclusion**: Only one thread can hold a lock
2. **Hold and Wait**: Thread holds lock while waiting for another
3. **No Preemption**: Can't force thread to release lock
4. **Circular Wait**: Thread-1 → Lock-A → Thread-2 → Lock-B → Thread-1

**Prevention strategies**:
1. **Lock Ordering**: Always acquire locks in consistent order (break circular wait)
2. **Timeout**: Give up after waiting too long (break hold and wait)
3. **Try-Lock-All**: Acquire all or none (break hold and wait)
4. **Deadlock Detection**: Find cycles and break them (detect circular wait)

I'll implement multiple approaches."

---

## Approach 1: Lock Ordering (Simplest)

**Me**: "The simplest solution - always acquire locks in sorted order:

```java
public class LockManagerV1 {
    
    // Map of resource ID to its lock
    private final ConcurrentHashMap<String, ReentrantLock> locks;
    
    public LockManagerV1() {
        this.locks = new ConcurrentHashMap<>();
    }
    
    // Acquire multiple locks
    public boolean acquireLocks(List<String> resourceIds) 
            throws InterruptedException {
        
        // KEY: Sort resources to ensure consistent order
        List<String> sorted = new ArrayList<>(resourceIds);
        Collections.sort(sorted);
        
        // Acquire in order
        for (String resourceId : sorted) {
            ReentrantLock lock = locks.computeIfAbsent(
                resourceId, 
                id -> new ReentrantLock()
            );
            lock.lockInterruptibly();
        }
        
        return true;
    }
    
    // Release locks
    public void releaseLocks(List<String> resourceIds) {
        for (String resourceId : resourceIds) {
            ReentrantLock lock = locks.get(resourceId);
            if (lock != null && lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }
}
```

**Why this works**:

```
Thread-1 wants: [Database, FileSystem]
Thread-2 wants: [FileSystem, Database]

Without sorting:
Thread-1: Lock Database → Wait for FileSystem
Thread-2: Lock FileSystem → Wait for Database
DEADLOCK!

With sorting (alphabetical):
Both sort to: [Database, FileSystem]

Thread-1: Lock Database → Lock FileSystem ✓
Thread-2: Wait for Database → Eventually gets both ✓
NO DEADLOCK!
```

---

## Approach 2: Timeout-Based Acquisition

**Me**: "Let's add timeout support:

```java
public class LockManagerV2 {
    
    private final ConcurrentHashMap<String, ReentrantLock> locks;
    
    public LockManagerV2() {
        this.locks = new ConcurrentHashMap<>();
    }
    
    public boolean acquireLocks(List<String> resourceIds, long timeoutMs) 
            throws InterruptedException {
        
        // Sort for consistent ordering
        List<String> sorted = new ArrayList<>(resourceIds);
        Collections.sort(sorted);
        
        List<ReentrantLock> acquired = new ArrayList<>();
        long deadline = System.currentTimeMillis() + timeoutMs;
        
        try {
            for (String resourceId : sorted) {
                long remaining = deadline - System.currentTimeMillis();
                
                if (remaining <= 0) {
                    // Timeout expired
                    System.out.println(Thread.currentThread().getName() + 
                        " TIMEOUT acquiring " + resourceId);
                    return false;
                }
                
                ReentrantLock lock = locks.computeIfAbsent(
                    resourceId, 
                    id -> new ReentrantLock()
                );
                
                // Try with remaining time
                if (!lock.tryLock(remaining, TimeUnit.MILLISECONDS)) {
                    System.out.println(Thread.currentThread().getName() + 
                        " TIMEOUT on " + resourceId);
                    return false;
                }
                
                acquired.add(lock);
                System.out.println(Thread.currentThread().getName() + 
                    " acquired " + resourceId);
            }
            
            return true;
            
        } finally {
            // Always release acquired locks if we didn't get all
            if (acquired.size() < sorted.size()) {
                System.out.println(Thread.currentThread().getName() + 
                    " releasing " + acquired.size() + " locks (partial acquisition)");
                for (ReentrantLock lock : acquired) {
                    lock.unlock();
                }
            }
        }
    }
    
    public void releaseLocks(List<String> resourceIds) {
        for (String resourceId : resourceIds) {
            ReentrantLock lock = locks.get(resourceId);
            if (lock != null && lock.isHeldByCurrentThread()) {
                lock.unlock();
                System.out.println(Thread.currentThread().getName() + 
                    " released " + resourceId);
            }
        }
    }
}
```

---

## Approach 3: All-or-Nothing with Retry

**Me**: "Let's add exponential backoff retry:

```java
public class LockManagerV3 {
    
    private final ConcurrentHashMap<String, ReentrantLock> locks;
    private final int maxRetries;
    private final long baseBackoffMs;
    
    public LockManagerV3(int maxRetries, long baseBackoffMs) {
        this.locks = new ConcurrentHashMap<>();
        this.maxRetries = maxRetries;
        this.baseBackoffMs = baseBackoffMs;
    }
    
    public boolean acquireLocksWithRetry(List<String> resourceIds) 
            throws InterruptedException {
        
        for (int attempt = 0; attempt < maxRetries; attempt++) {
            if (tryAcquireAll(resourceIds)) {
                return true;
            }
            
            // Exponential backoff
            long backoff = baseBackoffMs * (1L << attempt);  // 2^attempt
            System.out.println(Thread.currentThread().getName() + 
                " retry attempt " + attempt + ", backing off " + backoff + "ms");
            Thread.sleep(backoff);
        }
        
        return false;  // Failed after all retries
    }
    
    private boolean tryAcquireAll(List<String> resourceIds) 
            throws InterruptedException {
        
        List<String> sorted = new ArrayList<>(resourceIds);
        Collections.sort(sorted);
        
        List<ReentrantLock> acquired = new ArrayList<>();
        
        try {
            for (String resourceId : sorted) {
                ReentrantLock lock = locks.computeIfAbsent(
                    resourceId, 
                    id -> new ReentrantLock()
                );
                
                // Try immediately, don't wait
                if (!lock.tryLock()) {
                    // Failed to acquire, release all and return false
                    return false;
                }
                
                acquired.add(lock);
            }
            
            // Success - acquired all locks!
            return true;
            
        } finally {
            // Release if didn't get all
            if (acquired.size() < sorted.size()) {
                for (ReentrantLock lock : acquired) {
                    lock.unlock();
                }
            }
        }
    }
    
    public void releaseLocks(List<String> resourceIds) {
        for (String resourceId : resourceIds) {
            ReentrantLock lock = locks.get(resourceId);
            if (lock != null && lock.isHeldByCurrentThread()) {
                lock.unlock();
            }
        }
    }
}
```

---

## Approach 4: Deadlock Detection (Advanced)

**Me**: "We can build a wait-for graph and detect cycles:

```java
public class LockManagerV4 {
    
    private final ConcurrentHashMap<String, LockInfo> locks;
    private final ConcurrentHashMap<String, Set<String>> waitGraph;
    
    static class LockInfo {
        final ReentrantLock lock = new ReentrantLock();
        volatile String holder;  // Thread ID holding lock
        final Set<String> waiters = ConcurrentHashMap.newKeySet();
    }
    
    public LockManagerV4() {
        this.locks = new ConcurrentHashMap<>();
        this.waitGraph = new ConcurrentHashMap<>();
    }
    
    public boolean acquireLocks(List<String> resourceIds, long timeoutMs) 
            throws InterruptedException {
        
        String threadId = Thread.currentThread().getName();
        List<String> sorted = new ArrayList<>(resourceIds);
        Collections.sort(sorted);
        
        List<String> acquired = new ArrayList<>();
        
        try {
            for (String resourceId : sorted) {
                LockInfo info = locks.computeIfAbsent(
                    resourceId, 
                    id -> new LockInfo()
                );
                
                // Check for potential deadlock BEFORE waiting
                if (wouldCauseDeadlock(threadId, resourceId)) {
                    System.out.println("❌ DEADLOCK DETECTED! " + threadId + 
                        " cannot acquire " + resourceId);
                    return false;
                }
                
                // Add to wait graph
                info.waiters.add(threadId);
                waitGraph.computeIfAbsent(threadId, k -> ConcurrentHashMap.newKeySet())
                    .add(resourceId);
                
                // Try to acquire
                if (!info.lock.tryLock(timeoutMs, TimeUnit.MILLISECONDS)) {
                    // Timeout
                    info.waiters.remove(threadId);
                    waitGraph.get(threadId).remove(resourceId);
                    return false;
                }
                
                // Acquired!
                info.holder = threadId;
                info.waiters.remove(threadId);
                waitGraph.get(threadId).remove(resourceId);
                acquired.add(resourceId);
                
                System.out.println("✓ " + threadId + " acquired " + resourceId);
            }
            
            return true;
            
        } catch (InterruptedException e) {
            // Cleanup
            for (String resourceId : acquired) {
                releaseLock(resourceId);
            }
            throw e;
        }
    }
    
    private boolean wouldCauseDeadlock(String threadId, String resourceId) {
        LockInfo info = locks.get(resourceId);
        if (info == null || info.holder == null) {
            return false;  // Not held, can't cause deadlock
        }
        
        // Check if there's a path from resource holder back to this thread
        return hasPath(info.holder, threadId, new HashSet<>());
    }
    
    private boolean hasPath(String from, String to, Set<String> visited) {
        if (from.equals(to)) {
            return true;  // Cycle found!
        }
        
        if (visited.contains(from)) {
            return false;  // Already checked
        }
        
        visited.add(from);
        
        // Check what resources 'from' thread is waiting for
        Set<String> waitingFor = waitGraph.get(from);
        if (waitingFor == null) {
            return false;
        }
        
        // For each resource, check who holds it
        for (String resource : waitingFor) {
            LockInfo info = locks.get(resource);
            if (info != null && info.holder != null) {
                if (hasPath(info.holder, to, visited)) {
                    return true;  // Found path through this holder
                }
            }
        }
        
        return false;
    }
    
    private void releaseLock(String resourceId) {
        LockInfo info = locks.get(resourceId);
        if (info != null) {
            info.holder = null;
            info.lock.unlock();
        }
    }
    
    public void releaseLocks(List<String> resourceIds) {
        String threadId = Thread.currentThread().getName();
        for (String resourceId : resourceIds) {
            LockInfo info = locks.get(resourceId);
            if (info != null && threadId.equals(info.holder)) {
                releaseLock(resourceId);
                System.out.println("✓ " + threadId + " released " + resourceId);
            }
        }
    }
}
```

---

## Production-Ready Implementation

**Me**: "Here's a complete, production-ready version:

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;

public class DistributedLockManager {
    
    // Lock registry
    private final ConcurrentHashMap<String, LockInfo> locks;
    
    // Configuration
    private final long defaultTimeoutMs;
    private final int maxRetries;
    
    // Statistics
    private final AtomicLong successfulAcquisitions = new AtomicLong(0);
    private final AtomicLong failedAcquisitions = new AtomicLong(0);
    private final AtomicLong deadlocksDetected = new AtomicLong(0);
    
    static class LockInfo {
        final ReentrantLock lock;
        volatile String holder;
        final AtomicLong acquisitions = new AtomicLong(0);
        final Set<String> waiters = ConcurrentHashMap.newKeySet();
        
        LockInfo() {
            this.lock = new ReentrantLock(true);  // Fair lock
        }
    }
    
    public DistributedLockManager(long defaultTimeoutMs, int maxRetries) {
        this.locks = new ConcurrentHashMap<>();
        this.defaultTimeoutMs = defaultTimeoutMs;
        this.maxRetries = maxRetries;
    }
    
    /**
     * Acquire multiple locks atomically with retry
     */
    public boolean acquireLocks(List<String> resourceIds) 
            throws InterruptedException {
        return acquireLocksWithTimeout(resourceIds, defaultTimeoutMs);
    }
    
    public boolean acquireLocksWithTimeout(List<String> resourceIds, long timeoutMs) 
            throws InterruptedException {
        
        if (resourceIds == null || resourceIds.isEmpty()) {
            throw new IllegalArgumentException("Resource IDs cannot be empty");
        }
        
        // Try with exponential backoff
        long baseBackoff = 10;  // Start with 10ms
        
        for (int attempt = 0; attempt < maxRetries; attempt++) {
            if (tryAcquireAllLocks(resourceIds, timeoutMs)) {
                successfulAcquisitions.incrementAndGet();
                return true;
            }
            
            // Failed, back off before retry
            if (attempt < maxRetries - 1) {
                long backoff = baseBackoff * (1L << attempt);
                System.out.println(Thread.currentThread().getName() + 
                    " retry " + attempt + ", backing off " + backoff + "ms");
                Thread.sleep(backoff);
            }
        }
        
        failedAcquisitions.incrementAndGet();
        return false;
    }
    
    private boolean tryAcquireAllLocks(List<String> resourceIds, long timeoutMs) 
            throws InterruptedException {
        
        String threadId = Thread.currentThread().getName();
        
        // Sort for consistent ordering (deadlock prevention)
        List<String> sorted = new ArrayList<>(resourceIds);
        Collections.sort(sorted);
        
        List<String> acquired = new ArrayList<>();
        long deadline = System.currentTimeMillis() + timeoutMs;
        
        try {
            for (String resourceId : sorted) {
                long remaining = deadline - System.currentTimeMillis();
                
                if (remaining <= 0) {
                    System.out.println(threadId + " timeout expired");
                    return false;
                }
                
                LockInfo info = locks.computeIfAbsent(
                    resourceId, 
                    id -> new LockInfo()
                );
                
                // Try to acquire with remaining time
                if (!info.lock.tryLock(remaining, TimeUnit.MILLISECONDS)) {
                    System.out.println(threadId + " timeout on " + resourceId);
                    return false;
                }
                
                // Acquired!
                info.holder = threadId;
                info.acquisitions.incrementAndGet();
                acquired.add(resourceId);
                
                System.out.println(threadId + " acquired " + resourceId);
            }
            
            // Success - acquired all locks
            return true;
            
        } finally {
            // If didn't acquire all, release what we got
            if (acquired.size() < sorted.size()) {
                System.out.println(threadId + " releasing " + acquired.size() + 
                    " locks (partial acquisition)");
                    
                for (String resourceId : acquired) {
                    LockInfo info = locks.get(resourceId);
                    if (info != null) {
                        info.holder = null;
                        info.lock.unlock();
                    }
                }
            }
        }
    }
    
    /**
     * Release locks
     */
    public void releaseLocks(List<String> resourceIds) {
        String threadId = Thread.currentThread().getName();
        
        for (String resourceId : resourceIds) {
            LockInfo info = locks.get(resourceId);
            
            if (info != null && info.lock.isHeldByCurrentThread()) {
                info.holder = null;
                info.lock.unlock();
                System.out.println(threadId + " released " + resourceId);
            }
        }
    }
    
    /**
     * Check if thread holds lock
     */
    public boolean holdsLock(String resourceId) {
        LockInfo info = locks.get(resourceId);
        return info != null && info.lock.isHeldByCurrentThread();
    }
    
    /**
     * Get lock statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalLocks", locks.size());
        stats.put("successfulAcquisitions", successfulAcquisitions.get());
        stats.put("failedAcquisitions", failedAcquisitions.get());
        stats.put("deadlocksDetected", deadlocksDetected.get());
        
        // Per-lock stats
        Map<String, Long> perLock = new HashMap<>();
        for (Map.Entry<String, LockInfo> entry : locks.entrySet()) {
            perLock.put(entry.getKey(), entry.getValue().acquisitions.get());
        }
        stats.put("perLockAcquisitions", perLock);
        
        return stats;
    }
    
    /**
     * Get current lock holders
     */
    public Map<String, String> getCurrentHolders() {
        Map<String, String> holders = new HashMap<>();
        for (Map.Entry<String, LockInfo> entry : locks.entrySet()) {
            if (entry.getValue().holder != null) {
                holders.put(entry.getKey(), entry.getValue().holder);
            }
        }
        return holders;
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        DistributedLockManager manager = new DistributedLockManager(5000, 3);
        
        ExecutorService executor = Executors.newFixedThreadPool(5);
        CountDownLatch latch = new CountDownLatch(5);
        
        // Simulate concurrent lock acquisitions
        for (int i = 0; i < 5; i++) {
            final int threadNum = i;
            executor.submit(() -> {
                try {
                    List<String> resources = Arrays.asList(
                        "resource-" + (threadNum % 3),
                        "resource-" + ((threadNum + 1) % 3)
                    );
                    
                    if (manager.acquireLocks(resources)) {
                        System.out.println(Thread.currentThread().getName() + 
                            " doing work...");
                        Thread.sleep(100);  // Simulate work
                        manager.releaseLocks(resources);
                    } else {
                        System.out.println(Thread.currentThread().getName() + 
                            " failed to acquire locks");
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await();
        executor.shutdown();
        
        System.out.println("\n=== Statistics ===");
        manager.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
    }
}
```

---

## Summary: Key Takeaways

**Me**: "To wrap up:

### Core Concepts:
1. **Deadlock happens** when circular wait exists
2. **Prevention**: Lock ordering (simplest)
3. **Detection**: Wait-for graph + cycle detection (complex)
4. **Timeout**: Fail fast, retry with backoff
5. **All-or-nothing**: Acquire all or release all

### Best Practices:
```java
✓ Always sort resources before acquiring
✓ Use try-finally for release
✓ Support timeout for bounded waiting
✓ Add retry with exponential backoff
✓ Track statistics for monitoring
✓ Handle InterruptedException properly
✓ Consider fairness (ReentrantLock(true))
```

### When to Use:
- Multiple resource coordination
- Database transaction management
- Resource allocation systems
- Distributed systems (with external coordinator)

### Interview Strategy:
1. Start with simple lock ordering
2. Add timeout if asked
3. Discuss deadlock detection if time permits
4. Show awareness of distributed challenges
5. Talk about testing and edge cases

The key insight: **Prevention (lock ordering) is simpler and faster than detection (graph cycles)**. Use detection only when ordering isn't possible."

---

**END OF DOCUMENT**

# Remaining Concurrency Design Problems - Interview Format

---

## Table of Contents

1. [Batch Processor with Auto-Flush](#batch-processor-with-auto-flush)
2. [TTL Cache with Size Limits](#ttl-cache-with-size-limits)
3. [Rate Limiter](#rate-limiter)
4. [Multi-Stage Pipeline with Backpressure](#multi-stage-pipeline-with-backpressure)
5. [Write-Behind Cache](#write-behind-cache)
6. [Real-Time Metrics Aggregator](#real-time-metrics-aggregator)
7. [Producer-Consumer with Multiple Queues](#producer-consumer-with-multiple-queues)
8. [Concurrent Data Deduplicator](#concurrent-data-deduplicator)
9. [Multi-Level Cache](#multi-level-cache)
10. [Thread Pool with Dynamic Sizing](#thread-pool-with-dynamic-sizing)
11. [Concurrent Result Aggregator](#concurrent-result-aggregator)

---

# Batch Processor with Auto-Flush

## Problem Introduction

**Interviewer**: "Design a batch processor that collects items and processes them in batches. It should auto-flush when the batch reaches a certain size OR after a timeout period."

**Me**: "Great! Let me clarify:

**Scenario**:
```
Items arrive: I1, I2, I3, I4, I5, I6...
Batch size: 5
Timeout: 2 seconds

Scenario 1: Items arrive fast
I1, I2, I3, I4, I5 → FLUSH (size reached)
I6, I7, I8, I9, I10 → FLUSH (size reached)

Scenario 2: Items arrive slowly
I1, I2, I3 → ... 2 seconds pass → FLUSH (timeout)
I4 → ... 2 seconds pass → FLUSH (timeout)
```

**Questions**:
1. **Batch size and timeout**: Both configurable?
2. **Concurrent producers**: Multiple threads adding items?
3. **Processing**: Synchronous or async? Who processes?
4. **Ordering**: Must items be processed in order?
5. **Failure handling**: What if batch processing fails?
6. **Shutdown**: How to handle items in buffer during shutdown?

**Assumptions**:
- Both size and timeout triggers
- Multiple concurrent producers
- Async processing (background thread)
- Order preserved within batch
- Simple error logging
- Flush all on shutdown

Does this sound right?"

**Interviewer**: "Yes, exactly. Start with basic implementation."

---

## High-Level Design

**Me**: "Here's my approach:

### Components:
```
BatchProcessor
├── Buffer (stores items until batch ready)
├── Lock + Condition (thread-safe add)
├── Flusher Thread (monitors timeout)
├── Batch Handler (processes batches)
└── Statistics
```

### Two triggers:
1. **Size trigger**: When buffer.size() >= batchSize
2. **Time trigger**: When lastFlushTime + timeout < now

### Key challenge:
**How to handle timeout-based flushing while allowing concurrent adds?**

**Solution**: Background thread that wakes up periodically to check timeout."

---

## Step 1: Basic Structure

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;

public class BatchProcessor<T> {
    
    // Configuration
    private final int batchSize;
    private final long flushIntervalMs;
    private final BatchHandler<T> handler;
    
    // Buffer
    private final List<T> buffer;
    private final Lock lock;
    private final Condition notEmpty;
    
    // State
    private long lastFlushTime;
    private volatile boolean shutdown = false;
    
    // Background flusher
    private final Thread flusher;
    
    // Statistics
    private final AtomicLong itemsProcessed = new AtomicLong(0);
    private final AtomicLong batchesProcessed = new AtomicLong(0);
    
    public interface BatchHandler<T> {
        void processBatch(List<T> batch);
    }
    
    public BatchProcessor(int batchSize, long flushIntervalMs, BatchHandler<T> handler) {
        this.batchSize = batchSize;
        this.flushIntervalMs = flushIntervalMs;
        this.handler = handler;
        
        this.buffer = new ArrayList<>(batchSize);
        this.lock = new ReentrantLock();
        this.notEmpty = lock.newCondition();
        
        this.lastFlushTime = System.currentTimeMillis();
        
        // Start background flusher
        this.flusher = new Thread(this::flushPeriodically, "BatchFlusher");
        this.flusher.setDaemon(true);
        this.flusher.start();
    }
    
    private void flushPeriodically() {
        // Will implement next
    }
}
```

---

## Step 2: The add() Method

```java
public void add(T item) throws InterruptedException {
    if (item == null) {
        throw new IllegalArgumentException("Item cannot be null");
    }
    
    lock.lock();
    try {
        if (shutdown) {
            throw new IllegalStateException("Processor is shut down");
        }
        
        buffer.add(item);
        System.out.println(Thread.currentThread().getName() + 
            " added item. Buffer size: " + buffer.size());
        
        // Check if batch is full
        if (buffer.size() >= batchSize) {
            flush();  // Flush immediately
        }
        
        // Signal flusher thread
        notEmpty.signal();
        
    } finally {
        lock.unlock();
    }
}
```

**Key decision**: Call `flush()` while holding lock vs outside lock?

```java
// Option 1: Flush while holding lock (BLOCKING)
if (buffer.size() >= batchSize) {
    flush();  // Other threads blocked during processing!
}

// Option 2: Flush outside lock (BETTER)
List<T> toFlush = null;
lock.lock();
try {
    if (buffer.size() >= batchSize) {
        toFlush = new ArrayList<>(buffer);
        buffer.clear();
        lastFlushTime = System.currentTimeMillis();
    }
} finally {
    lock.unlock();
}

if (toFlush != null) {
    processBatch(toFlush);  // Outside lock!
}
```

**Better implementation**:

```java
public void add(T item) throws InterruptedException {
    if (item == null) {
        throw new IllegalArgumentException("Item cannot be null");
    }
    
    List<T> toFlush = null;
    
    lock.lock();
    try {
        if (shutdown) {
            throw new IllegalStateException("Processor is shut down");
        }
        
        buffer.add(item);
        System.out.println(Thread.currentThread().getName() + 
            " added item. Buffer size: " + buffer.size());
        
        // Check if batch is full
        if (buffer.size() >= batchSize) {
            toFlush = new ArrayList<>(buffer);
            buffer.clear();
            lastFlushTime = System.currentTimeMillis();
            System.out.println("Size trigger: Flushing batch of " + toFlush.size());
        }
        
        notEmpty.signal();
        
    } finally {
        lock.unlock();
    }
    
    // Process outside lock
    if (toFlush != null) {
        processBatch(toFlush);
    }
}

private void processBatch(List<T> batch) {
    try {
        handler.processBatch(batch);
        itemsProcessed.addAndGet(batch.size());
        batchesProcessed.incrementAndGet();
    } catch (Exception e) {
        System.err.println("Error processing batch: " + e.getMessage());
    }
}
```

---

## Step 3: Periodic Flusher Thread

```java
private void flushPeriodically() {
    while (!shutdown) {
        try {
            lock.lock();
            try {
                // Wait until timeout or notified
                while (!shutdown && buffer.isEmpty()) {
                    notEmpty.await(flushIntervalMs, TimeUnit.MILLISECONDS);
                }
                
                if (shutdown) {
                    break;
                }
                
                // Check if timeout elapsed
                long now = System.currentTimeMillis();
                long elapsed = now - lastFlushTime;
                
                if (elapsed >= flushIntervalMs && !buffer.isEmpty()) {
                    System.out.println("Timeout trigger: Flushing batch of " + buffer.size());
                    List<T> toFlush = new ArrayList<>(buffer);
                    buffer.clear();
                    lastFlushTime = now;
                    
                    // Process outside lock
                    lock.unlock();
                    try {
                        processBatch(toFlush);
                    } finally {
                        lock.lock();
                    }
                }
                
            } finally {
                lock.unlock();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            break;
        }
    }
}
```

**Key point**: Process batch outside lock to avoid blocking producers!

---

## Step 4: Shutdown and Force Flush

```java
public void forceFlush() {
    List<T> toFlush = null;
    
    lock.lock();
    try {
        if (!buffer.isEmpty()) {
            toFlush = new ArrayList<>(buffer);
            buffer.clear();
            lastFlushTime = System.currentTimeMillis();
            System.out.println("Force flush: " + toFlush.size() + " items");
        }
    } finally {
        lock.unlock();
    }
    
    if (toFlush != null) {
        processBatch(toFlush);
    }
}

public void shutdown() throws InterruptedException {
    lock.lock();
    try {
        shutdown = true;
        notEmpty.signalAll();
    } finally {
        lock.unlock();
    }
    
    // Wait for flusher thread
    flusher.join();
    
    // Final flush
    forceFlush();
    
    System.out.println("Batch processor shut down");
}

public Map<String, Long> getStatistics() {
    Map<String, Long> stats = new HashMap<>();
    stats.put("itemsProcessed", itemsProcessed.get());
    stats.put("batchesProcessed", batchesProcessed.get());
    
    lock.lock();
    try {
        stats.put("itemsInBuffer", (long) buffer.size());
    } finally {
        lock.unlock();
    }
    
    return stats;
}
```

---

## Complete Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;
import java.util.concurrent.atomic.*;

public class BatchProcessor<T> {
    
    // Configuration
    private final int batchSize;
    private final long flushIntervalMs;
    private final BatchHandler<T> handler;
    
    // Buffer
    private final List<T> buffer;
    private final Lock lock;
    private final Condition notEmpty;
    
    // State
    private long lastFlushTime;
    private volatile boolean shutdown = false;
    
    // Background flusher
    private final Thread flusher;
    
    // Statistics
    private final AtomicLong itemsProcessed = new AtomicLong(0);
    private final AtomicLong batchesProcessed = new AtomicLong(0);
    private final AtomicLong sizeTriggers = new AtomicLong(0);
    private final AtomicLong timeTriggers = new AtomicLong(0);
    
    public interface BatchHandler<T> {
        void processBatch(List<T> batch);
    }
    
    public BatchProcessor(int batchSize, long flushIntervalMs, BatchHandler<T> handler) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        if (flushIntervalMs <= 0) {
            throw new IllegalArgumentException("Flush interval must be positive");
        }
        if (handler == null) {
            throw new IllegalArgumentException("Handler cannot be null");
        }
        
        this.batchSize = batchSize;
        this.flushIntervalMs = flushIntervalMs;
        this.handler = handler;
        
        this.buffer = new ArrayList<>(batchSize);
        this.lock = new ReentrantLock();
        this.notEmpty = lock.newCondition();
        
        this.lastFlushTime = System.currentTimeMillis();
        
        this.flusher = new Thread(this::flushPeriodically, "BatchFlusher");
        this.flusher.setDaemon(true);
        this.flusher.start();
    }
    
    public void add(T item) throws InterruptedException {
        if (item == null) {
            throw new IllegalArgumentException("Item cannot be null");
        }
        
        List<T> toFlush = null;
        
        lock.lock();
        try {
            if (shutdown) {
                throw new IllegalStateException("Processor is shut down");
            }
            
            buffer.add(item);
            
            // Check if batch is full
            if (buffer.size() >= batchSize) {
                toFlush = new ArrayList<>(buffer);
                buffer.clear();
                lastFlushTime = System.currentTimeMillis();
                sizeTriggers.incrementAndGet();
                System.out.println(Thread.currentThread().getName() + 
                    " - Size trigger: Flushing batch of " + toFlush.size());
            }
            
            notEmpty.signal();
            
        } finally {
            lock.unlock();
        }
        
        // Process outside lock
        if (toFlush != null) {
            processBatch(toFlush);
        }
    }
    
    private void flushPeriodically() {
        while (!shutdown) {
            try {
                List<T> toFlush = null;
                
                lock.lock();
                try {
                    // Wait until timeout or notified
                    while (!shutdown && buffer.isEmpty()) {
                        notEmpty.await(flushIntervalMs, TimeUnit.MILLISECONDS);
                    }
                    
                    if (shutdown) {
                        break;
                    }
                    
                    // Check if timeout elapsed
                    long now = System.currentTimeMillis();
                    long elapsed = now - lastFlushTime;
                    
                    if (elapsed >= flushIntervalMs && !buffer.isEmpty()) {
                        toFlush = new ArrayList<>(buffer);
                        buffer.clear();
                        lastFlushTime = now;
                        timeTriggers.incrementAndGet();
                        System.out.println("Timeout trigger: Flushing batch of " + toFlush.size());
                    }
                    
                } finally {
                    lock.unlock();
                }
                
                // Process outside lock
                if (toFlush != null) {
                    processBatch(toFlush);
                }
                
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
    
    private void processBatch(List<T> batch) {
        try {
            handler.processBatch(batch);
            itemsProcessed.addAndGet(batch.size());
            batchesProcessed.incrementAndGet();
        } catch (Exception e) {
            System.err.println("Error processing batch: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public void forceFlush() {
        List<T> toFlush = null;
        
        lock.lock();
        try {
            if (!buffer.isEmpty()) {
                toFlush = new ArrayList<>(buffer);
                buffer.clear();
                lastFlushTime = System.currentTimeMillis();
                System.out.println("Force flush: " + toFlush.size() + " items");
            }
        } finally {
            lock.unlock();
        }
        
        if (toFlush != null) {
            processBatch(toFlush);
        }
    }
    
    public void shutdown() throws InterruptedException {
        lock.lock();
        try {
            shutdown = true;
            notEmpty.signalAll();
        } finally {
            lock.unlock();
        }
        
        // Wait for flusher thread
        flusher.join();
        
        // Final flush
        forceFlush();
        
        System.out.println("Batch processor shut down");
    }
    
    public Map<String, Long> getStatistics() {
        Map<String, Long> stats = new HashMap<>();
        stats.put("itemsProcessed", itemsProcessed.get());
        stats.put("batchesProcessed", batchesProcessed.get());
        stats.put("sizeTriggers", sizeTriggers.get());
        stats.put("timeTriggers", timeTriggers.get());
        
        lock.lock();
        try {
            stats.put("itemsInBuffer", (long) buffer.size());
        } finally {
            lock.unlock();
        }
        
        return stats;
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        BatchProcessor<String> processor = new BatchProcessor<>(
            5,      // Batch size
            2000,   // Flush every 2 seconds
            batch -> {
                System.out.println(">>> PROCESSING BATCH: " + batch);
                try {
                    Thread.sleep(100); // Simulate processing
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        );
        
        // Scenario 1: Fast arrival (size trigger)
        System.out.println("\n=== Scenario 1: Fast Arrival ===");
        for (int i = 0; i < 12; i++) {
            processor.add("item-" + i);
            Thread.sleep(50);
        }
        
        Thread.sleep(3000);
        
        // Scenario 2: Slow arrival (timeout trigger)
        System.out.println("\n=== Scenario 2: Slow Arrival ===");
        for (int i = 20; i < 23; i++) {
            processor.add("item-" + i);
            Thread.sleep(500);
        }
        
        Thread.sleep(3000);
        
        System.out.println("\n=== Statistics ===");
        processor.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
        
        processor.shutdown();
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Let me discuss the gotchas:

### Pitfall 1: Processing While Holding Lock

```java
// ❌ WRONG - Blocks all producers during processing
lock.lock();
try {
    if (buffer.size() >= batchSize) {
        handler.processBatch(buffer);  // Long operation under lock!
        buffer.clear();
    }
} finally {
    lock.unlock();
}

// ✓ CORRECT - Process outside lock
List<T> toFlush = null;
lock.lock();
try {
    if (buffer.size() >= batchSize) {
        toFlush = new ArrayList<>(buffer);
        buffer.clear();
    }
} finally {
    lock.unlock();
}

if (toFlush != null) {
    handler.processBatch(toFlush);  // Outside lock
}
```

### Pitfall 2: Race Between Size and Time Triggers

```java
// Problem scenario:
Thread-1 (producer): Adds 5th item → triggers size flush
Thread-2 (flusher): Wakes up → sees empty buffer (already flushed)

// Solution: Check if buffer is empty before flushing
if (elapsed >= flushIntervalMs && !buffer.isEmpty()) {
    // Safe to flush
}
```

### Pitfall 3: Lost Items on Shutdown

```java
// ❌ WRONG - Items in buffer are lost
public void shutdown() {
    shutdown = true;
    // Buffer has items but never flushed!
}

// ✓ CORRECT - Final flush
public void shutdown() throws InterruptedException {
    shutdown = true;
    flusher.join();
    forceFlush();  // Flush remaining items
}
```

### Pitfall 4: Time Drift

```java
// Problem: If items arrive frequently, timeout never triggers!
add(item1) → lastFlushTime = T0
add(item2) → lastFlushTime = T0 (not updated)
add(item3) → lastFlushTime = T0 (not updated)
... timeout should trigger but size trigger happens first

// This is actually OK - size trigger takes precedence
// But be aware of the behavior
```

### Pitfall 5: Handler Throws Exception

```java
// ❌ WRONG - Exception kills flusher thread
private void processBatch(List<T> batch) {
    handler.processBatch(batch);  // Throws exception
    // Thread dies!
}

// ✓ CORRECT - Catch and log
private void processBatch(List<T> batch) {
    try {
        handler.processBatch(batch);
    } catch (Exception e) {
        System.err.println("Error processing batch: " + e);
        // Thread continues
    }
}
```

### Pitfall 6: Buffer Capacity Growth

```java
// Problem: ArrayList grows unbounded if items arrive faster than processing
buffer = new ArrayList<>();  // Starts at 10, doubles when full

// Solution: Use bounded buffer with backpressure
if (buffer.size() >= MAX_BUFFER_SIZE) {
    // Option 1: Block
    bufferNotFull.await();
    
    // Option 2: Drop
    System.err.println("Buffer full, dropping item");
    return;
    
    // Option 3: Flush immediately
    forceFlush();
}
```
"

---

## Lock-Free Optimization

**Me**: "Can we optimize with lock-free structures?

### Current Bottleneck: Single Lock

```java
// All producers contend on same lock
Thread-1: lock.lock() → add → unlock
Thread-2: lock.lock() → BLOCKED
Thread-3: lock.lock() → BLOCKED
```

### Option 1: Lock-Free Queue (Partial Solution)

```java
public class LockFreeBatchProcessor<T> {
    
    private final ConcurrentLinkedQueue<T> buffer;
    private final AtomicInteger bufferSize;
    private final int batchSize;
    
    public void add(T item) {
        buffer.offer(item);  // Lock-free!
        int size = bufferSize.incrementAndGet();  // Lock-free!
        
        if (size >= batchSize) {
            // Try to flush
            tryFlush();
        }
    }
    
    private void tryFlush() {
        // Problem: How to atomically drain queue?
        // Multiple threads might try to flush simultaneously!
        
        // Solution: Use atomic flag
        if (flushing.compareAndSet(false, true)) {
            try {
                List<T> batch = new ArrayList<>();
                T item;
                while (batch.size() < batchSize && (item = buffer.poll()) != null) {
                    batch.add(item);
                    bufferSize.decrementAndGet();
                }
                
                if (!batch.isEmpty()) {
                    processBatch(batch);
                }
            } finally {
                flushing.set(false);
            }
        }
    }
}
```

**Problem**: Timeout-based flushing still needs coordination!

### Option 2: Sharded Buffers

```java
public class ShardedBatchProcessor<T> {
    
    private final BatchProcessor<T>[] shards;
    
    public ShardedBatchProcessor(int numShards, int batchSize, long timeout, BatchHandler<T> handler) {
        this.shards = new BatchProcessor[numShards];
        for (int i = 0; i < numShards; i++) {
            shards[i] = new BatchProcessor<>(batchSize, timeout, handler);
        }
    }
    
    public void add(T item) throws InterruptedException {
        // Route to shard based on thread ID
        int shard = (int) (Thread.currentThread().getId() % shards.length);
        shards[shard].add(item);  // Less contention per shard!
    }
}
```

**Trade-off**:
- ✓ Less contention per shard
- ✗ More complex
- ✗ Batches are per-shard (might be smaller)

### Recommendation

**For this problem, locks are appropriate** because:
1. Timeout-based flushing needs coordination anyway
2. Batch creation requires atomic buffer drain
3. Lock is held briefly (just copying list)
4. Processing happens outside lock

**Lock-free helps** only if:
- Pure size-based flushing (no timeout)
- Acceptable to have multiple concurrent flushes
- Can tolerate imprecise batch sizes
"

---

## Interview Follow-Up Questions

**Q1: What if batch processing fails? Should we retry?**

```java
private void processBatch(List<T> batch) {
    int maxRetries = 3;
    for (int attempt = 0; attempt < maxRetries; attempt++) {
        try {
            handler.processBatch(batch);
            return;  // Success
        } catch (Exception e) {
            if (attempt == maxRetries - 1) {
                // Failed all retries
                sendToDeadLetterQueue(batch);
            } else {
                Thread.sleep(1000 * (1 << attempt));  // Exponential backoff
            }
        }
    }
}
```

**Q2: How to handle backpressure if items arrive too fast?**

```java
public void add(T item) throws InterruptedException {
    lock.lock();
    try {
        // Wait if buffer is too large
        while (buffer.size() >= MAX_BUFFER_SIZE) {
            bufferNotFull.await();  // Block producer
        }
        
        buffer.add(item);
        
        if (buffer.size() >= batchSize) {
            // Flush and signal
            flush();
            bufferNotFull.signalAll();
        }
    } finally {
        lock.unlock();
    }
}
```

**Q3: How to test this?**

```java
@Test
public void testSizeTrigger() throws InterruptedException {
    List<List<String>> batches = new ArrayList<>();
    BatchProcessor<String> processor = new BatchProcessor<>(
        3, 10000, batches::add
    );
    
    processor.add("1");
    processor.add("2");
    processor.add("3");  // Should trigger
    
    Thread.sleep(100);
    assertEquals(1, batches.size());
    assertEquals(Arrays.asList("1", "2", "3"), batches.get(0));
}

@Test
public void testTimeoutTrigger() throws InterruptedException {
    List<List<String>> batches = new ArrayList<>();
    BatchProcessor<String> processor = new BatchProcessor<>(
        100, 500, batches::add
    );
    
    processor.add("1");
    processor.add("2");
    
    Thread.sleep(600);  // Wait for timeout
    
    assertEquals(1, batches.size());
    assertEquals(Arrays.asList("1", "2"), batches.get(0));
}
```

---

# TTL Cache with Size Limits

## Problem Introduction

**Interviewer**: "Design a thread-safe cache with TTL (time-to-live) and size limits. Items expire after a certain time and the cache has a maximum size."

**Me**: "Interesting! Let me clarify:

**Requirements**:
```
Cache with:
1. TTL: Items expire after X milliseconds
2. Size limit: Max N items
3. Eviction: Remove expired or least-recently-used items
4. Thread-safe: Multiple threads reading/writing

Example:
cache.put("key1", "value1", 5000);  // Expires in 5 seconds
cache.get("key1");  // Returns "value1"
... 6 seconds later ...
cache.get("key1");  // Returns null (expired)
```

**Questions**:
1. **Eviction policy**: LRU, LFU, or just expiry-based?
2. **Expiry check**: On access or background cleanup?
3. **Size enforcement**: Block when full or evict?
4. **Read-heavy or write-heavy**: Optimize for reads?
5. **TTL**: Per-item or global default?

**Assumptions**:
- LRU + TTL eviction
- Check expiry on access + background cleanup
- Evict automatically when full
- Read-heavy (use ReadWriteLock)
- Per-item TTL with default

Sound good?"

**Interviewer**: "Perfect. Start with the design."

---

## High-Level Design

**Me**: "Here's the approach:

### Data Structure:
```
TTLCache
├── ConcurrentHashMap<K, CacheEntry<V>>
│   └── CacheEntry: value, expiryTime, lastAccess
├── LinkedHashMap for LRU ordering (or manual doubly-linked list)
├── ReadWriteLock (many readers, few writers)
├── Cleanup thread (removes expired entries)
└── Statistics
```

### Eviction Strategy:
1. **On put()**: If size > maxSize, evict LRU
2. **On get()**: Check if expired, remove if yes
3. **Background**: Periodic cleanup of expired entries

### Key Design Decision:
**Use ConcurrentHashMap or Lock-based?**

```
Option 1: ConcurrentHashMap
✓ Built-in thread safety
✓ Lock striping (good concurrency)
✗ Can't easily maintain LRU order

Option 2: HashMap + ReadWriteLock
✓ Can use LinkedHashMap for LRU
✓ Readers don't block each other
✗ Writers block readers

Hybrid: ConcurrentHashMap + separate LRU tracking
```

I'll show both approaches."

---

## Approach 1: ConcurrentHashMap-Based

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class TTLCache<K, V> {
    
    static class CacheEntry<V> {
        final V value;
        final long expiryTime;
        volatile long lastAccessTime;
        
        CacheEntry(V value, long ttlMs) {
            this.value = value;
            this.expiryTime = System.currentTimeMillis() + ttlMs;
            this.lastAccessTime = System.currentTimeMillis();
        }
        
        boolean isExpired() {
            return System.currentTimeMillis() > expiryTime;
        }
    }
    
    private final ConcurrentHashMap<K, CacheEntry<V>> cache;
    private final int maxSize;
    private final long defaultTtlMs;
    
    // For LRU tracking (approximation)
    private final ConcurrentLinkedQueue<K> accessOrder;
    
    // Cleanup
    private final ScheduledExecutorService cleaner;
    
    // Statistics
    private final AtomicLong hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);
    private final AtomicLong evictions = new AtomicLong(0);
    
    public TTLCache(int maxSize, long defaultTtlMs) {
        this.cache = new ConcurrentHashMap<>();
        this.maxSize = maxSize;
        this.defaultTtlMs = defaultTtlMs;
        this.accessOrder = new ConcurrentLinkedQueue<>();
        
        // Periodic cleanup
        this.cleaner = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "CacheCleaner");
            t.setDaemon(true);
            return t;
        });
        
        cleaner.scheduleAtFixedRate(
            this::cleanupExpired,
            1000,
            1000,
            TimeUnit.MILLISECONDS
        );
    }
    
    public void put(K key, V value) {
        put(key, value, defaultTtlMs);
    }
    
    public void put(K key, V value, long ttlMs) {
        if (key == null || value == null) {
            throw new IllegalArgumentException("Key and value cannot be null");
        }
        
        // Check size limit
        if (cache.size() >= maxSize && !cache.containsKey(key)) {
            evictOne();
        }
        
        CacheEntry<V> entry = new CacheEntry<>(value, ttlMs);
        cache.put(key, entry);
        accessOrder.offer(key);  // Track for LRU
        
        System.out.println("Put: " + key + " (TTL: " + ttlMs + "ms)");
    }
    
    public V get(K key) {
        CacheEntry<V> entry = cache.get(key);
        
        if (entry == null) {
            misses.incrementAndGet();
            return null;
        }
        
        if (entry.isExpired()) {
            // Expired, remove it
            cache.remove(key);
            evictions.incrementAndGet();
            misses.incrementAndGet();
            System.out.println("Get: " + key + " - EXPIRED");
            return null;
        }
        
        // Update access time
        entry.lastAccessTime = System.currentTimeMillis();
        accessOrder.offer(key);  // Move to end (approximate LRU)
        hits.incrementAndGet();
        
        return entry.value;
    }
    
    public void remove(K key) {
        if (cache.remove(key) != null) {
            System.out.println("Removed: " + key);
        }
    }
    
    private void evictOne() {
        // Evict least recently used (approximate)
        K keyToEvict = accessOrder.poll();
        
        while (keyToEvict != null) {
            if (cache.remove(keyToEvict) != null) {
                evictions.incrementAndGet();
                System.out.println("Evicted (LRU): " + keyToEvict);
                return;
            }
            // Key already removed, try next
            keyToEvict = accessOrder.poll();
        }
        
        // Fallback: remove any key
        K anyKey = cache.keys().nextElement();
        if (anyKey != null) {
            cache.remove(anyKey);
            evictions.incrementAndGet();
            System.out.println("Evicted (any): " + anyKey);
        }
    }
    
    private void cleanupExpired() {
        int cleaned = 0;
        
        Iterator<Map.Entry<K, CacheEntry<V>>> iterator = cache.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<K, CacheEntry<V>> entry = iterator.next();
            if (entry.getValue().isExpired()) {
                iterator.remove();
                cleaned++;
                evictions.incrementAndGet();
            }
        }
        
        if (cleaned > 0) {
            System.out.println("Cleanup: removed " + cleaned + " expired entries");
        }
    }
    
    public void shutdown() {
        cleaner.shutdown();
    }
    
    public Map<String, Long> getStatistics() {
        Map<String, Long> stats = new HashMap<>();
        stats.put("size", (long) cache.size());
        stats.put("hits", hits.get());
        stats.put("misses", misses.get());
        stats.put("evictions", evictions.get());
        
        long total = hits.get() + misses.get();
        stats.put("hitRate", total == 0 ? 0 : (hits.get() * 100 / total));
        
        return stats;
    }
}
```

**Problem with this approach**: Access order queue is not perfect LRU (duplicates possible).

---

## Approach 2: LinkedHashMap with ReadWriteLock (Better LRU)

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.locks.*;
import java.util.concurrent.atomic.*;

public class TTLCacheV2<K, V> {
    
    static class CacheEntry<V> {
        final V value;
        final long expiryTime;
        
        CacheEntry(V value, long ttlMs) {
            this.value = value;
            this.expiryTime = System.currentTimeMillis() + ttlMs;
        }
        
        boolean isExpired() {
            return System.currentTimeMillis() > expiryTime;
        }
    }
    
    // LRU map with access-order
    private final LinkedHashMap<K, CacheEntry<V>> cache;
    private final int maxSize;
    private final long defaultTtlMs;
    
    // Read-write lock for better concurrency
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    
    // Cleanup
    private final ScheduledExecutorService cleaner;
    
    // Statistics
    private final AtomicLong hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);
    private final AtomicLong evictions = new AtomicLong(0);
    
    public TTLCacheV2(int maxSize, long defaultTtlMs) {
        this.maxSize = maxSize;
        this.defaultTtlMs = defaultTtlMs;
        
        // LinkedHashMap with access-order and automatic eviction
        this.cache = new LinkedHashMap<K, CacheEntry<V>>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<K, CacheEntry<V>> eldest) {
                boolean shouldRemove = size() > maxSize;
                if (shouldRemove) {
                    evictions.incrementAndGet();
                    System.out.println("Auto-evicted (LRU): " + eldest.getKey());
                }
                return shouldRemove;
            }
        };
        
        // Periodic cleanup
        this.cleaner = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "CacheCleaner");
            t.setDaemon(true);
            return t;
        });
        
        cleaner.scheduleAtFixedRate(
            this::cleanupExpired,
            1000,
            1000,
            TimeUnit.MILLISECONDS
        );
    }
    
    public void put(K key, V value) {
        put(key, value, defaultTtlMs);
    }
    
    public void put(K key, V value, long ttlMs) {
        if (key == null || value == null) {
            throw new IllegalArgumentException("Key and value cannot be null");
        }
        
        lock.writeLock().lock();  // Exclusive write
        try {
            CacheEntry<V> entry = new CacheEntry<>(value, ttlMs);
            cache.put(key, entry);  // LinkedHashMap handles LRU
            System.out.println("Put: " + key + " (TTL: " + ttlMs + "ms, size: " + cache.size() + ")");
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public V get(K key) {
        lock.readLock().lock();  // Multiple readers OK
        try {
            CacheEntry<V> entry = cache.get(key);
            
            if (entry == null) {
                misses.incrementAndGet();
                return null;
            }
            
            if (entry.isExpired()) {
                misses.incrementAndGet();
                
                // Need write lock to remove
                lock.readLock().unlock();
                lock.writeLock().lock();
                try {
                    // Double-check after acquiring write lock
                    entry = cache.get(key);
                    if (entry != null && entry.isExpired()) {
                        cache.remove(key);
                        evictions.incrementAndGet();
                        System.out.println("Get: " + key + " - EXPIRED");
                    }
                    return null;
                } finally {
                    lock.readLock().lock();
                    lock.writeLock().unlock();
                }
            }
            
            hits.incrementAndGet();
            return entry.value;
            
        } finally {
            lock.readLock().unlock();
        }
    }
    
    public void remove(K key) {
        lock.writeLock().lock();
        try {
            if (cache.remove(key) != null) {
                System.out.println("Removed: " + key);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    private void cleanupExpired() {
        lock.writeLock().lock();
        try {
            Iterator<Map.Entry<K, CacheEntry<V>>> iterator = cache.entrySet().iterator();
            int cleaned = 0;
            
            while (iterator.hasNext()) {
                Map.Entry<K, CacheEntry<V>> entry = iterator.next();
                if (entry.getValue().isExpired()) {
                    iterator.remove();
                    cleaned++;
                    evictions.incrementAndGet();
                }
            }
            
            if (cleaned > 0) {
                System.out.println("Cleanup: removed " + cleaned + " expired entries");
            }
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public void shutdown() {
        cleaner.shutdown();
    }
    
    public Map<String, Long> getStatistics() {
        lock.readLock().lock();
        try {
            Map<String, Long> stats = new HashMap<>();
            stats.put("size", (long) cache.size());
            stats.put("hits", hits.get());
            stats.put("misses", misses.get());
            stats.put("evictions", evictions.get());
            
            long total = hits.get() + misses.get();
            stats.put("hitRate", total == 0 ? 0 : (hits.get() * 100 / total));
            
            return stats;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        TTLCacheV2<String, String> cache = new TTLCacheV2<>(3, 2000);
        
        // Test size eviction
        System.out.println("\n=== Test Size Eviction ===");
        cache.put("key1", "value1");
        cache.put("key2", "value2");
        cache.put("key3", "value3");
        cache.put("key4", "value4");  // Should evict key1
        
        System.out.println("key1: " + cache.get("key1"));  // null (evicted)
        System.out.println("key2: " + cache.get("key2"));  // value2
        
        // Test TTL expiry
        System.out.println("\n=== Test TTL Expiry ===");
        cache.put("temp", "temporary", 1000);  // 1 second TTL
        System.out.println("temp (immediate): " + cache.get("temp"));  // temporary
        Thread.sleep(1500);
        System.out.println("temp (after 1.5s): " + cache.get("temp"));  // null (expired)
        
        // Test LRU
        System.out.println("\n=== Test LRU ===");
        cache.put("a", "valueA");
        cache.put("b", "valueB");
        cache.put("c", "valueC");
        cache.get("a");  // Access 'a', making it recently used
        cache.put("d", "valueD");  // Should evict 'b' (least recently used)
        
        System.out.println("a: " + cache.get("a"));  // valueA (accessed recently)
        System.out.println("b: " + cache.get("b"));  // null (evicted)
        System.out.println("c: " + cache.get("c"));  // valueC
        
        Thread.sleep(3000);
        
        System.out.println("\n=== Statistics ===");
        cache.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
        
        cache.shutdown();
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Key gotchas to watch for:

### Pitfall 1: Lock Upgrade Deadlock

```java
// ❌ WRONG - Can deadlock!
lock.readLock().lock();
try {
    if (entry.isExpired()) {
        lock.readLock().unlock();
        lock.writeLock().lock();  // Deadlock if multiple readers try this!
        try {
            cache.remove(key);
        } finally {
            lock.writeLock().unlock();
        }
    }
} finally {
    lock.readLock().unlock();  // Might be already unlocked!
}

// ✓ CORRECT - Careful lock management
lock.readLock().lock();
try {
    if (entry.isExpired()) {
        // Release read lock BEFORE acquiring write lock
        lock.readLock().unlock();
        lock.writeLock().lock();
        try {
            // Double-check after acquiring write lock
            entry = cache.get(key);
            if (entry != null && entry.isExpired()) {
                cache.remove(key);
            }
        } finally {
            // Downgrade: acquire read lock before releasing write lock
            lock.readLock().lock();
            lock.writeLock().unlock();
        }
    }
} finally {
    lock.readLock().unlock();
}
```

### Pitfall 2: Time-of-Check Time-of-Use (TOCTOU)

```java
// ❌ WRONG - Race condition
if (!entry.isExpired()) {  // Check
    return entry.value;     // Use (might be expired now!)
}

// ✓ CORRECT - Atomic check-and-use
lock.writeLock().lock();
try {
    if (entry.isExpired()) {
        cache.remove(key);
        return null;
    }
    return entry.value;
} finally {
    lock.writeLock().unlock();
}
```

### Pitfall 3: Cleanup Thread Holding Write Lock Too Long

```java
// ❌ WRONG - Blocks all readers during cleanup
lock.writeLock().lock();
try {
    for (Map.Entry<K, CacheEntry<V>> entry : cache.entrySet()) {
        if (entry.getValue().isExpired()) {
            cache.remove(entry.getKey());  // Modifies during iteration!
        }
    }
} finally {
    lock.writeLock().unlock();
}

// ✓ CORRECT - Use iterator for safe removal
lock.writeLock().lock();
try {
    Iterator<Map.Entry<K, CacheEntry<V>>> iterator = cache.entrySet().iterator();
    while (iterator.hasNext()) {
        if (iterator.next().getValue().isExpired()) {
            iterator.remove();  // Safe removal
        }
    }
} finally {
    lock.writeLock().unlock();
}
```

### Pitfall 4: Memory Leak with Weak References

```java
// Problem: Entries never cleaned up if not accessed
cache.put("key", hugeObject, Long.MAX_VALUE);  // Never expires
// If key never accessed again, hugeObject stays in memory forever!

// Solution: Maximum TTL or periodic full cleanup
private static final long MAX_TTL = TimeUnit.DAYS.toMillis(7);

public void put(K key, V value, long ttlMs) {
    long actualTtl = Math.min(ttlMs, MAX_TTL);
    // ...
}
```

### Pitfall 5: LinkedHashMap Removes During Iteration

```java
// ❌ WRONG - ConcurrentModificationException possible
for (Map.Entry<K, CacheEntry<V>> entry : cache.entrySet()) {
    if (entry.getValue().isExpired()) {
        cache.remove(entry.getKey());  // Modifies map during iteration!
    }
}

// ✓ CORRECT - Use iterator
Iterator<Map.Entry<K, CacheEntry<V>>> iterator = cache.entrySet().iterator();
while (iterator.hasNext()) {
    if (iterator.next().getValue().isExpired()) {
        iterator.remove();  // Safe
    }
}
```

### Pitfall 6: StampedLock for Optimistic Reads (Advanced)

```java
// Better performance for read-heavy workloads
private final StampedLock stampedLock = new StampedLock();

public V get(K key) {
    long stamp = stampedLock.tryOptimisticRead();  // Optimistic, no lock!
    CacheEntry<V> entry = cache.get(key);
    
    if (!stampedLock.validate(stamp)) {  // Check if write occurred
        // Validation failed, fall back to read lock
        stamp = stampedLock.readLock();
        try {
            entry = cache.get(key);
        } finally {
            stampedLock.unlockRead(stamp);
        }
    }
    
    // Process entry...
}
```
"

---

## Lock-Free Optimization

**Me**: "Can we make this lock-free?

### Challenge: LRU Ordering Requires Coordination

LRU needs to track access order, which requires some form of synchronization.

### Option 1: Approximate LRU with ConcurrentHashMap

Already shown in Approach 1 - uses `ConcurrentLinkedQueue` for approximate LRU.

**Trade-offs**:
- ✓ No lock contention
- ✗ Not perfect LRU (duplicates in queue)
- ✗ More memory overhead

### Option 2: Segment-Based (Caffeine Cache Approach)

```java
public class SegmentedCache<K, V> {
    
    private static final int SEGMENTS = 16;
    private final TTLCacheV2<K, V>[] segments;
    
    @SuppressWarnings("unchecked")
    public SegmentedCache(int totalSize, long ttl) {
        segments = new TTLCacheV2[SEGMENTS];
        int sizePerSegment = totalSize / SEGMENTS;
        
        for (int i = 0; i < SEGMENTS; i++) {
            segments[i] = new TTLCacheV2<>(sizePerSegment, ttl);
        }
    }
    
    private int segmentFor(K key) {
        return Math.abs(key.hashCode()) % SEGMENTS;
    }
    
    public void put(K key, V value) {
        segments[segmentFor(key)].put(key, value);
    }
    
    public V get(K key) {
        return segments[segmentFor(key)].get(key);
    }
}
```

**Benefits**:
- Reduces lock contention (16× less)
- Each segment has its own lock
- Different keys likely in different segments

### Option 3: W-TinyLFU (Caffeine's Approach)

Too complex for interview, but concept:
- Use bloom filter for frequency tracking
- Separate windows for recent vs frequent items
- Lock-free frequency counting with `LongAdder`

### Recommendation

**For interview**: 
1. Start with ReadWriteLock + LinkedHashMap (Approach 2)
2. Mention ConcurrentHashMap for lock-free reads
3. Discuss segmentation if asked about optimization
4. Reference Caffeine or Guava for production use

**Why locks are OK here**:
- Read lock doesn't block other readers
- Write operations are infrequent (compared to reads)
- LRU ordering requires some coordination anyway
"

---

## Interview Follow-Up Questions

**Q1: How would you handle cache warming?**

```java
public void warmUp(Map<K, V> initialData) {
    lock.writeLock().lock();
    try {
        for (Map.Entry<K, V> entry : initialData.entrySet()) {
            cache.put(entry.getKey(), 
                new CacheEntry<>(entry.getValue(), defaultTtlMs));
        }
        System.out.println("Cache warmed with " + initialData.size() + " entries");
    } finally {
        lock.writeLock().unlock();
    }
}
```

**Q2: How to implement cache-aside pattern?**

```java
public V getOrLoad(K key, Function<K, V> loader) {
    V value = get(key);
    if (value != null) {
        return value;  // Cache hit
    }
    
    // Cache miss - load from source
    value = loader.apply(key);
    if (value != null) {
        put(key, value);
    }
    return value;
}

// Problem: Multiple threads might load simultaneously!
// Better: Use computeIfAbsent pattern
public V getOrLoad(K key, Function<K, V> loader) {
    lock.writeLock().lock();
    try {
        CacheEntry<V> entry = cache.get(key);
        if (entry != null && !entry.isExpired()) {
            return entry.value;
        }
        
        // Load while holding lock (only one thread loads)
        V value = loader.apply(key);
        if (value != null) {
            cache.put(key, new CacheEntry<>(value, defaultTtlMs));
        }
        return value;
    } finally {
        lock.writeLock().unlock();
    }
}
```

**Q3: How to test expiry?**

```java
@Test
public void testExpiry() throws InterruptedException {
    TTLCacheV2<String, String> cache = new TTLCacheV2<>(10, 1000);
    
    cache.put("key", "value", 500);  // 500ms TTL
    
    assertEquals("value", cache.get("key"));  // Immediate: present
    
    Thread.sleep(600);  // Wait for expiry
    
    assertNull(cache.get("key"));  // After expiry: null
}

@Test
public void testLRU() {
    TTLCacheV2<String, String> cache = new TTLCacheV2<>(2, 10000);
    
    cache.put("key1", "value1");
    cache.put("key2", "value2");
    cache.get("key1");  // Access key1
    cache.put("key3", "value3");  // Should evict key2 (LRU)
    
    assertNotNull(cache.get("key1"));
    assertNull(cache.get("key2"));  // Evicted
    assertNotNull(cache.get("key3"));
}
```

# Rate Limiter

## Problem Introduction

**Interviewer**: "Design a rate limiter that allows N requests per time window. It should be thread-safe and handle concurrent requests."

**Me**: "Great! Let me clarify the requirements:

**Scenarios**:
```
Rate: 10 requests per second

Scenario 1: Fixed Window
Second 1: 10 requests → all allowed
Second 2: 10 requests → all allowed
Second 1 (11th request) → DENIED

Scenario 2: Sliding Window
Time 0.5s: 10 requests
Time 1.5s: Request #11 → Check last 1 second → 10 requests in window → DENIED

Scenario 3: Token Bucket
Start with 10 tokens
Request consumes 1 token
Tokens refill at 10/second
```

**Questions**:
1. **Algorithm**: Fixed window, sliding window, or token bucket?
2. **Scope**: Per-user, per-IP, or global?
3. **Distributed**: Single machine or distributed?
4. **Blocking**: Block when limit reached or reject immediately?
5. **Burst handling**: Allow bursts or strict rate?

**Assumptions**:
- Token bucket (allows bursts, smooth refill)
- Per-key rate limiting
- Single machine
- Reject immediately (non-blocking)
- Configurable rate

Sound good?"

**Interviewer**: "Yes, start with token bucket approach."

---

## High-Level Design

**Me**: "Here's the token bucket algorithm:

### Concept:
```
Bucket capacity: 10 tokens
Refill rate: 10 tokens/second (1 token per 100ms)

Request arrives:
1. Refill tokens based on time elapsed
2. If tokens >= 1: Allow request, consume 1 token
3. If tokens < 1: Deny request

Timeline:
T=0:    10 tokens, request → 9 tokens (allowed)
T=100:  10 tokens (refilled), request → 9 tokens (allowed)
T=0:    10 rapid requests → 0 tokens
T=50:   Request → 0 tokens (denied - not enough time to refill)
T=100:  1 token refilled, request → 0 tokens (allowed)
```

### Key Components:
```
RateLimiter
├── AtomicInteger tokens (current available)
├── AtomicLong lastRefillTime
├── maxTokens (capacity)
├── refillRate (tokens per second)
└── Per-key limiters (Map)
```

### Lock-Free Design:
Use `AtomicInteger` for tokens - perfect use case for CAS!"

---

## Implementation: Single Rate Limiter

```java
import java.util.concurrent.atomic.*;

public class TokenBucketRateLimiter {
    
    private final int maxTokens;
    private final double refillRate;  // tokens per second
    private final AtomicInteger tokens;
    private final AtomicLong lastRefillTime;
    
    // Statistics
    private final AtomicLong allowed = new AtomicLong(0);
    private final AtomicLong denied = new AtomicLong(0);
    
    public TokenBucketRateLimiter(int maxTokens, double refillRate) {
        if (maxTokens <= 0 || refillRate <= 0) {
            throw new IllegalArgumentException("maxTokens and refillRate must be positive");
        }
        
        this.maxTokens = maxTokens;
        this.refillRate = refillRate;
        this.tokens = new AtomicInteger(maxTokens);
        this.lastRefillTime = new AtomicLong(System.currentTimeMillis());
    }
    
    public boolean tryAcquire() {
        return tryAcquire(1);
    }
    
    public boolean tryAcquire(int permits) {
        if (permits <= 0) {
            throw new IllegalArgumentException("Permits must be positive");
        }
        
        // Refill tokens based on elapsed time
        refillTokens();
        
        // Try to consume tokens using CAS
        while (true) {
            int currentTokens = tokens.get();
            
            if (currentTokens < permits) {
                // Not enough tokens
                denied.incrementAndGet();
                System.out.println(Thread.currentThread().getName() + 
                    " - DENIED (tokens: " + currentTokens + ")");
                return false;
            }
            
            int newTokens = currentTokens - permits;
            
            // CAS: only succeed if tokens hasn't changed
            if (tokens.compareAndSet(currentTokens, newTokens)) {
                allowed.incrementAndGet();
                System.out.println(Thread.currentThread().getName() + 
                    " - ALLOWED (tokens: " + currentTokens + " -> " + newTokens + ")");
                return true;
            }
            
            // CAS failed, retry
        }
    }
    
    private void refillTokens() {
        long now = System.currentTimeMillis();
        long lastRefill = lastRefillTime.get();
        long elapsedMs = now - lastRefill;
        
        if (elapsedMs > 0) {
            // Calculate tokens to add
            double tokensToAdd = (elapsedMs / 1000.0) * refillRate;
            int tokensToAddInt = (int) tokensToAdd;
            
            if (tokensToAddInt > 0) {
                // Try to update last refill time
                if (lastRefillTime.compareAndSet(lastRefill, now)) {
                    // Successfully updated time, now add tokens
                    while (true) {
                        int currentTokens = tokens.get();
                        int newTokens = Math.min(maxTokens, currentTokens + tokensToAddInt);
                        
                        if (tokens.compareAndSet(currentTokens, newTokens)) {
                            if (newTokens > currentTokens) {
                                System.out.println("Refilled: " + tokensToAddInt + 
                                    " tokens (" + currentTokens + " -> " + newTokens + ")");
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
    
    public int getAvailableTokens() {
        refillTokens();
        return tokens.get();
    }
    
    public Map<String, Long> getStatistics() {
        Map<String, Long> stats = new HashMap<>();
        stats.put("allowed", allowed.get());
        stats.put("denied", denied.get());
        stats.put("availableTokens", (long) getAvailableTokens());
        
        long total = allowed.get() + denied.get();
        stats.put("allowRate", total == 0 ? 0 : (allowed.get() * 100 / total));
        
        return stats;
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        TokenBucketRateLimiter limiter = new TokenBucketRateLimiter(5, 2.0);  // 5 tokens, 2/sec
        
        System.out.println("=== Burst Test ===");
        // Burst: should allow 5, then deny
        for (int i = 0; i < 8; i++) {
            limiter.tryAcquire();
        }
        
        System.out.println("\n=== Refill Test ===");
        Thread.sleep(1000);  // Wait 1 second for refill (2 tokens)
        
        for (int i = 0; i < 4; i++) {
            limiter.tryAcquire();
        }
        
        System.out.println("\n=== Statistics ===");
        limiter.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
    }
}
```

---

## Implementation: Per-Key Rate Limiter

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class MultiKeyRateLimiter<K> {
    
    private final ConcurrentHashMap<K, TokenBucketRateLimiter> limiters;
    private final int maxTokens;
    private final double refillRate;
    
    // Cleanup old limiters
    private final ScheduledExecutorService cleaner;
    private final long limiterTtlMs;
    private final ConcurrentHashMap<K, Long> lastAccessTime;
    
    public MultiKeyRateLimiter(int maxTokens, double refillRate, long limiterTtlMs) {
        this.maxTokens = maxTokens;
        this.refillRate = refillRate;
        this.limiterTtlMs = limiterTtlMs;
        
        this.limiters = new ConcurrentHashMap<>();
        this.lastAccessTime = new ConcurrentHashMap<>();
        
        // Periodic cleanup of unused limiters
        this.cleaner = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "LimiterCleaner");
            t.setDaemon(true);
            return t;
        });
        
        cleaner.scheduleAtFixedRate(
            this::cleanupUnusedLimiters,
            60000,  // Initial delay: 1 minute
            60000,  // Period: 1 minute
            TimeUnit.MILLISECONDS
        );
    }
    
    public boolean tryAcquire(K key) {
        // Get or create limiter for this key
        TokenBucketRateLimiter limiter = limiters.computeIfAbsent(
            key,
            k -> new TokenBucketRateLimiter(maxTokens, refillRate)
        );
        
        // Update last access time
        lastAccessTime.put(key, System.currentTimeMillis());
        
        return limiter.tryAcquire();
    }
    
    public boolean tryAcquire(K key, int permits) {
        TokenBucketRateLimiter limiter = limiters.computeIfAbsent(
            key,
            k -> new TokenBucketRateLimiter(maxTokens, refillRate)
        );
        
        lastAccessTime.put(key, System.currentTimeMillis());
        
        return limiter.tryAcquire(permits);
    }
    
    private void cleanupUnusedLimiters() {
        long now = System.currentTimeMillis();
        int cleaned = 0;
        
        Iterator<Map.Entry<K, Long>> iterator = lastAccessTime.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<K, Long> entry = iterator.next();
            
            if (now - entry.getValue() > limiterTtlMs) {
                K key = entry.getKey();
                limiters.remove(key);
                iterator.remove();
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            System.out.println("Cleaned up " + cleaned + " unused limiters");
        }
    }
    
    public int getActiveLimiters() {
        return limiters.size();
    }
    
    public Map<String, Long> getStatistics(K key) {
        TokenBucketRateLimiter limiter = limiters.get(key);
        return limiter != null ? limiter.getStatistics() : Collections.emptyMap();
    }
    
    public void shutdown() {
        cleaner.shutdown();
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        MultiKeyRateLimiter<String> limiter = new MultiKeyRateLimiter<>(
            3,      // 3 tokens
            1.0,    // 1 token/second refill
            60000   // 1 minute TTL
        );
        
        System.out.println("=== Per-User Rate Limiting ===");
        
        // User 1: burst
        System.out.println("\nUser1 burst:");
        for (int i = 0; i < 5; i++) {
            limiter.tryAcquire("user1");
        }
        
        // User 2: different limit
        System.out.println("\nUser2 burst:");
        for (int i = 0; i < 5; i++) {
            limiter.tryAcquire("user2");
        }
        
        Thread.sleep(2000);  // Wait for refill
        
        System.out.println("\nAfter 2 seconds:");
        limiter.tryAcquire("user1");
        limiter.tryAcquire("user2");
        
        System.out.println("\n=== Statistics ===");
        System.out.println("Active limiters: " + limiter.getActiveLimiters());
        System.out.println("User1: " + limiter.getStatistics("user1"));
        System.out.println("User2: " + limiter.getStatistics("user2"));
        
        limiter.shutdown();
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Key gotchas:

### Pitfall 1: Race Condition in Refill

```java
// ❌ WRONG - Race condition
private void refillTokens() {
    long now = System.currentTimeMillis();
    long elapsed = now - lastRefillTime.get();
    
    if (elapsed > 0) {
        int tokensToAdd = (int) ((elapsed / 1000.0) * refillRate);
        
        // Two threads can both see elapsed > 0
        // Both add tokens!
        tokens.addAndGet(tokensToAdd);
        lastRefillTime.set(now);
    }
}

// ✓ CORRECT - CAS on lastRefillTime
private void refillTokens() {
    long now = System.currentTimeMillis();
    long lastRefill = lastRefillTime.get();
    long elapsed = now - lastRefill;
    
    if (elapsed > 0) {
        int tokensToAdd = (int) ((elapsed / 1000.0) * refillRate);
        
        // Only one thread succeeds in updating time
        if (lastRefillTime.compareAndSet(lastRefill, now)) {
            // Only this thread adds tokens
            while (true) {
                int current = tokens.get();
                int newTokens = Math.min(maxTokens, current + tokensToAdd);
                if (tokens.compareAndSet(current, newTokens)) {
                    break;
                }
            }
        }
    }
}
```

### Pitfall 2: Integer Overflow

```java
// ❌ WRONG - Can overflow with large elapsed time
long elapsedMs = now - lastRefillTime;  // Could be days!
int tokensToAdd = (int) ((elapsedMs / 1000.0) * refillRate);
// tokensToAdd could overflow int!

// ✓ CORRECT - Cap at maxTokens
int tokensToAdd = (int) Math.min(
    maxTokens,
    (elapsedMs / 1000.0) * refillRate
);
```

### Pitfall 3: Thundering Herd on Refill

```java
// Problem: All threads call refillTokens() on every request
// Wasteful if recently refilled

// Solution: Check if refill is needed before CAS
private void refillTokens() {
    long now = System.currentTimeMillis();
    long lastRefill = lastRefillTime.get();
    long elapsed = now - lastRefill;
    
    // Early exit if recently refilled
    if (elapsed < 100) {  // Less than 100ms
        return;  // Don't bother
    }
    
    // Proceed with refill...
}
```

### Pitfall 4: Precision Loss with Double

```java
// ❌ WRONG - Precision issues
double tokensToAdd = (elapsedMs / 1000.0) * refillRate;
int tokensInt = (int) tokensToAdd;  // Loses fractional tokens!

// Better: Track fractional tokens
private final AtomicReference<Double> fractionalTokens = new AtomicReference<>(0.0);

double tokensToAdd = (elapsedMs / 1000.0) * refillRate;
double withFractional = tokensToAdd + fractionalTokens.get();
int tokensInt = (int) withFractional;
fractionalTokens.set(withFractional - tokensInt);
```

### Pitfall 5: Memory Leak with Per-Key Limiters

```java
// ❌ WRONG - Limiters accumulate forever
ConcurrentHashMap<K, RateLimiter> limiters = new ConcurrentHashMap<>();

public boolean tryAcquire(K key) {
    RateLimiter limiter = limiters.computeIfAbsent(key, k -> new RateLimiter(...));
    return limiter.tryAcquire();
}
// Keys never removed, even for inactive users!

// ✓ CORRECT - Periodic cleanup (shown in MultiKeyRateLimiter)
private void cleanupUnusedLimiters() {
    // Remove limiters not accessed in last N minutes
}
```

### Pitfall 6: Distributed Rate Limiting

```java
// Problem: Multiple servers each have their own limiters
// User can exceed rate by hitting different servers!

// Solution: Use shared storage (Redis)
public class DistributedRateLimiter {
    private final Jedis redis;
    
    public boolean tryAcquire(String key) {
        String redisKey = "ratelimit:" + key;
        
        // Lua script for atomic get-and-decrement
        String script = 
            "local tokens = redis.call('get', KEYS[1]) " +
            "if not tokens then " +
            "  redis.call('set', KEYS[1], ARGV[1]) " +
            "  redis.call('expire', KEYS[1], ARGV[2]) " +
            "  return 1 " +
            "end " +
            "if tonumber(tokens) > 0 then " +
            "  redis.call('decr', KEYS[1]) " +
            "  return 1 " +
            "else " +
            "  return 0 " +
            "end";
        
        Object result = redis.eval(script, 
            Collections.singletonList(redisKey),
            Arrays.asList(String.valueOf(maxTokens), "60"));
        
        return "1".equals(result.toString());
    }
}
```
"

---

## Alternative Algorithms

**Me**: "Other rate limiting algorithms:

### Fixed Window Counter

```java
public class FixedWindowRateLimiter {
    private final int maxRequests;
    private final long windowMs;
    private final AtomicInteger counter;
    private final AtomicLong windowStart;
    
    public boolean tryAcquire() {
        long now = System.currentTimeMillis();
        long currentWindowStart = windowStart.get();
        
        // Check if we're in a new window
        if (now - currentWindowStart >= windowMs) {
            if (windowStart.compareAndSet(currentWindowStart, now)) {
                counter.set(0);
            }
        }
        
        // Try to increment counter
        while (true) {
            int current = counter.get();
            if (current >= maxRequests) {
                return false;  // Rate limited
            }
            if (counter.compareAndSet(current, current + 1)) {
                return true;
            }
        }
    }
}
```

**Problem**: Burst at window boundary!
```
Window 1 (0-1s): 100 requests at t=0.9s (allowed)
Window 2 (1-2s): 100 requests at t=1.0s (allowed)
Result: 200 requests in 0.1 seconds!
```

### Sliding Window Log

```java
public class SlidingWindowLogRateLimiter {
    private final int maxRequests;
    private final long windowMs;
    private final ConcurrentLinkedQueue<Long> requestLog;
    
    public boolean tryAcquire() {
        long now = System.currentTimeMillis();
        long windowStart = now - windowMs;
        
        // Remove old requests
        while (!requestLog.isEmpty() && requestLog.peek() < windowStart) {
            requestLog.poll();
        }
        
        if (requestLog.size() < maxRequests) {
            requestLog.offer(now);
            return true;
        }
        
        return false;
    }
}
```

**Problem**: Memory overhead (stores all requests)

### Sliding Window Counter (Hybrid)

```java
public class SlidingWindowCounterRateLimiter {
    private final int maxRequests;
    private final long windowMs;
    private final AtomicInteger currentCount;
    private final AtomicInteger previousCount;
    private final AtomicLong currentWindowStart;
    
    public boolean tryAcquire() {
        long now = System.currentTimeMillis();
        long currentStart = currentWindowStart.get();
        
        // Slide window if needed
        if (now - currentStart >= windowMs) {
            if (currentWindowStart.compareAndSet(currentStart, now)) {
                previousCount.set(currentCount.get());
                currentCount.set(0);
            }
        }
        
        // Calculate weighted count
        long elapsed = now - currentWindowStart.get();
        double weight = 1.0 - (elapsed / (double) windowMs);
        double estimatedCount = previousCount.get() * weight + currentCount.get();
        
        if (estimatedCount < maxRequests) {
            currentCount.incrementAndGet();
            return true;
        }
        
        return false;
    }
}
```

**Comparison**:

| Algorithm | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **Token Bucket** | Smooth, allows bursts | Complex refill logic | General purpose, API throttling |
| **Fixed Window** | Simple, memory efficient | Burst at boundaries | Low precision OK |
| **Sliding Log** | Precise | High memory | Need exact limits |
| **Sliding Counter** | Good precision, low memory | Approximation | Balance precision/memory |
"

---

## Interview Follow-Up Questions

**Q1: How to handle distributed rate limiting across multiple servers?**

```java
// Option 1: Redis with Lua script (atomic)
public class RedisRateLimiter {
    private final Jedis redis;
    
    public boolean tryAcquire(String key) {
        String script = 
            "local current = redis.call('incr', KEYS[1]) " +
            "if current == 1 then " +
            "  redis.call('expire', KEYS[1], ARGV[1]) " +
            "end " +
            "if current > tonumber(ARGV[2]) then " +
            "  return 0 " +
            "else " +
            "  return 1 " +
            "end";
        
        Object result = redis.eval(script,
            Collections.singletonList("ratelimit:" + key),
            Arrays.asList("60", "100"));  // 60s window, 100 requests
        
        return "1".equals(result.toString());
    }
}

// Option 2: Consensus-based (more complex)
// Use distributed counter with eventual consistency
```

**Q2: How to implement priority-based rate limiting?**

```java
public class PriorityRateLimiter {
    private final TokenBucketRateLimiter highPriority;
    private final TokenBucketRateLimiter lowPriority;
    
    public boolean tryAcquire(int priority) {
        if (priority == HIGH) {
            // Try high priority first, fallback to low
            return highPriority.tryAcquire() || lowPriority.tryAcquire();
        } else {
            // Low priority only uses its bucket
            return lowPriority.tryAcquire();
        }
    }
}
```

**Q3: How to test rate limiter?**

```java
@Test
public void testRateLimit() throws InterruptedException {
    RateLimiter limiter = new TokenBucketRateLimiter(10, 10.0);
    
    // Burst: should allow 10
    int allowed = 0;
    for (int i = 0; i < 15; i++) {
        if (limiter.tryAcquire()) {
            allowed++;
        }
    }
    assertEquals(10, allowed);
    
    // Wait for refill
    Thread.sleep(1000);
    
    // Should allow ~10 more
    allowed = 0;
    for (int i = 0; i < 15; i++) {
        if (limiter.tryAcquire()) {
            allowed++;
        }
    }
    assertTrue(allowed >= 9 && allowed <= 11);  // Allow some variance
}
```

---

# Multi-Stage Pipeline with Backpressure

## Problem Introduction

**Interviewer**: "Design a multi-stage processing pipeline where data flows through multiple stages. Handle backpressure when downstream stages are slow."

**Me**: "Great problem! Let me clarify:

**Scenario**:
```
Stage 1 (Fast)    Stage 2 (Medium)   Stage 3 (Slow)
Download     →    Parse          →    Store
100 items/s       50 items/s          10 items/s

Problem: Stage 1 overwhelms Stage 2, Stage 2 overwhelms Stage 3
Solution: Backpressure - slow down upstream when downstream is full
```

**Questions**:
1. **Pipeline structure**: Linear or DAG?
2. **Backpressure strategy**: Block, drop, or buffer?
3. **Queue bounds**: Fixed or unbounded?
4. **Failure handling**: Retry, dead letter, or fail fast?
5. **Ordering**: Must preserve order?

**Assumptions**:
- Linear pipeline (Stage 1 → 2 → 3)
- Block upstream (backpressure)
- Bounded queues between stages
- Simple error logging
- Order preserved

Sound good?"

**Interviewer**: "Yes, focus on backpressure mechanism."

---

## High-Level Design

**Me**: "Here's the architecture:

### Pipeline Structure:
```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Stage 1 │───▶│ Queue 1 │───▶│ Stage 2 │───▶│ Queue 2 │───▶│ Stage 3 │
│  Fast   │    │(bounded)│    │ Medium  │    │(bounded)│    │  Slow   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                    ▲                              ▲
                    │                              │
                Blocks when full              Blocks when full
```

### Backpressure Flow:
```
1. Stage 3 is slow → Queue 2 fills up
2. Stage 2 tries to put → Queue 2 blocks Stage 2
3. Queue 1 fills up (Stage 2 not consuming)
4. Stage 1 tries to put → Queue 1 blocks Stage 1
5. Stage 1 slows down → Backpressure propagated!
```

### Key Design:
- Use `BlockingQueue` (built-in blocking)
- Each stage is a thread pool
- Bounded queues create backpressure naturally"

---

## Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;

public class Pipeline<T> {
    
    private final List<Stage<?, ?>> stages;
    private volatile boolean shutdown = false;
    
    // Statistics
    private final AtomicLong itemsProcessed = new AtomicLong(0);
    private final AtomicLong itemsFailed = new AtomicLong(0);
    
    private Pipeline(List<Stage<?, ?>> stages) {
        this.stages = stages;
    }
    
    public static <T> PipelineBuilder<T> builder() {
        return new PipelineBuilder<>();
    }
    
    public void start() {
        for (Stage<?, ?> stage : stages) {
            stage.start();
        }
        System.out.println("Pipeline started with " + stages.size() + " stages");
    }
    
    public void shutdown() throws InterruptedException {
        shutdown = true;
        
        for (Stage<?, ?> stage : stages) {
            stage.shutdown();
        }
        
        System.out.println("Pipeline shut down");
    }
    
    public void submit(T item) throws InterruptedException {
        if (shutdown) {
            throw new IllegalStateException("Pipeline is shut down");
        }
        
        @SuppressWarnings("unchecked")
        Stage<T, ?> firstStage = (Stage<T, ?>) stages.get(0);
        firstStage.submit(item);
    }
    
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("itemsProcessed", itemsProcessed.get());
        stats.put("itemsFailed", itemsFailed.get());
        
        List<Map<String, Object>> stageStats = new ArrayList<>();
        for (Stage<?, ?> stage : stages) {
            stageStats.add(stage.getStatistics());
        }
        stats.put("stages", stageStats);
        
        return stats;
    }
    
    // Individual stage
    static class Stage<I, O> {
        private final String name;
        private final Function<I, O> processor;
        private final BlockingQueue<I> inputQueue;
        private final BlockingQueue<O> outputQueue;
        private final ExecutorService workers;
        private final int numWorkers;
        
        // Statistics
        private final AtomicLong processed = new AtomicLong(0);
        private final AtomicLong failed = new AtomicLong(0);
        private volatile boolean shutdown = false;
        
        public Stage(String name, 
                     int queueSize, 
                     int numWorkers,
                     Function<I, O> processor,
                     BlockingQueue<O> outputQueue) {
            this.name = name;
            this.processor = processor;
            this.inputQueue = new ArrayBlockingQueue<>(queueSize);
            this.outputQueue = outputQueue;
            this.numWorkers = numWorkers;
            this.workers = Executors.newFixedThreadPool(numWorkers);
        }
        
        public void start() {
            for (int i = 0; i < numWorkers; i++) {
                final int workerId = i;
                workers.submit(() -> processLoop(workerId));
            }
            System.out.println(name + " started with " + numWorkers + " workers");
        }
        
        private void processLoop(int workerId) {
            while (!shutdown) {
                try {
                    // Take from input queue (blocks if empty)
                    I item = inputQueue.poll(100, TimeUnit.MILLISECONDS);
                    if (item == null) {
                        continue;  // Timeout, check shutdown flag
                    }
                    
                    System.out.println(name + "-Worker-" + workerId + 
                        " processing item (queue size: " + inputQueue.size() + ")");
                    
                    // Process item
                    O result = processor.apply(item);
                    processed.incrementAndGet();
                    
                    // Put to output queue (blocks if full - BACKPRESSURE!)
                    if (outputQueue != null) {
                        outputQueue.put(result);
                        System.out.println(name + "-Worker-" + workerId + 
                            " sent to next stage (output queue size: " + outputQueue.size() + ")");
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    System.err.println(name + " error: " + e.getMessage());
                    failed.incrementAndGet();
                }
            }
        }
        
        public void submit(I item) throws InterruptedException {
            // Blocks if queue is full - BACKPRESSURE!
            inputQueue.put(item);
        }
        
        public void shutdown() throws InterruptedException {
            shutdown = true;
            workers.shutdown();
            workers.awaitTermination(10, TimeUnit.SECONDS);
            System.out.println(name + " shut down");
        }
        
        public Map<String, Object> getStatistics() {
            Map<String, Object> stats = new HashMap<>();
            stats.put("name", name);
            stats.put("processed", processed.get());
            stats.put("failed", failed.get());
            stats.put("queueSize", inputQueue.size());
            return stats;
        }
    }
    
    // Builder for fluent API
    public static class PipelineBuilder<T> {
        private final List<StageConfig<?, ?>> stageConfigs = new ArrayList<>();
        private Class<?> currentType;
        
        public PipelineBuilder() {
            this.currentType = Object.class;
        }
        
        public <O> PipelineBuilder<O> addStage(String name,
                                                int queueSize,
                                                int numWorkers,
                                                Function<T, O> processor) {
            @SuppressWarnings("unchecked")
            StageConfig<T, O> config = new StageConfig<>(
                name, queueSize, numWorkers, (Function<Object, Object>) processor
            );
            stageConfigs.add(config);
            
            @SuppressWarnings("unchecked")
            PipelineBuilder<O> next = (PipelineBuilder<O>) this;
            return next;
        }
        
        public Pipeline<T> build() {
            List<Stage<?, ?>> stages = new ArrayList<>();
            BlockingQueue<Object> outputQueue = null;
            
            // Build stages in reverse (to create output queues)
            for (int i = stageConfigs.size() - 1; i >= 0; i--) {
                StageConfig<?, ?> config = stageConfigs.get(i);
                
                @SuppressWarnings("unchecked")
                Stage<Object, Object> stage = new Stage<>(
                    config.name,
                    config.queueSize,
                    config.numWorkers,
                    config.processor,
                    outputQueue
                );
                
                stages.add(0, stage);  // Add to front
                
                // This stage's input queue becomes previous stage's output
                outputQueue = stage.inputQueue;
            }
            
            return new Pipeline<>(stages);
        }
        
        private static class StageConfig<I, O> {
            final String name;
            final int queueSize;
            final int numWorkers;
            final Function<Object, Object> processor;
            
            StageConfig(String name, int queueSize, int numWorkers, Function<Object, Object> processor) {
                this.name = name;
                this.queueSize = queueSize;
                this.numWorkers = numWorkers;
                this.processor = processor;
            }
        }
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        Pipeline<String> pipeline = Pipeline.<String>builder()
            .addStage("Stage-1-Fast", 10, 2, item -> {
                // Fast processing
                try { Thread.sleep(10); } catch (InterruptedException e) { }
                return item.toUpperCase();
            })
            .addStage("Stage-2-Medium", 5, 2, item -> {
                // Medium processing
                try { Thread.sleep(50); } catch (InterruptedException e) { }
                return item + "-processed";
            })
            .addStage("Stage-3-Slow", 3, 1, item -> {
                // Slow processing
                try { Thread.sleep(200); } catch (InterruptedException e) { }
                System.out.println("  >>> FINAL OUTPUT: " + item);
                return item + "-done";
            })
            .build();
        
        pipeline.start();
        
        // Submit items
        System.out.println("\n=== Submitting Items ===");
        for (int i = 0; i < 20; i++) {
            final int id = i;
            new Thread(() -> {
                try {
                    pipeline.submit("item-" + id);
                    System.out.println("Submitted: item-" + id);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }).start();
            
            Thread.sleep(20);  // Slow down submission to see backpressure
        }
        
        Thread.sleep(10000);  // Let pipeline process
        
        System.out.println("\n=== Statistics ===");
        Map<String, Object> stats = pipeline.getStatistics();
        System.out.println("Items processed: " + stats.get("itemsProcessed"));
        
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> stageStats = (List<Map<String, Object>>) stats.get("stages");
        for (Map<String, Object> stageStat : stageStats) {
            System.out.println(stageStat);
        }
        
        pipeline.shutdown();
    }
}
```

---

## Observing Backpressure

**Me**: "Let me trace through what happens with backpressure:

```
Timeline with Fast Producer, Slow Consumer:

T=0ms:   Submit item-0
         Stage-1 queue: [item-0] (size: 1/10)
         
T=10ms:  Stage-1 processes item-0 → "ITEM-0"
         Stage-2 queue: ["ITEM-0"] (size: 1/5)
         
T=20ms:  Submit item-1
         Stage-1 queue: [item-1] (size: 1/10)
         
T=30ms:  Stage-1 processes item-1 → "ITEM-1"
         Stage-2 queue: ["ITEM-0", "ITEM-1"] (size: 2/5)
         
... Items keep arriving every 20ms ...

T=100ms: Stage-2 queue FULL (size: 5/5)
         Stage-1 tries to put → BLOCKS!
         Stage-1 queue starts filling up
         
T=120ms: Submit item-6
         Stage-1 queue: [item-2, item-3, item-4, item-5, item-6] (size: 5/10)
         
T=200ms: Stage-1 queue FULL (size: 10/10)
         Main thread submits item-7 → BLOCKS!
         Producer is now blocked → BACKPRESSURE WORKING!
         
T=210ms: Stage-3 finishes processing "ITEM-0"
         Stage-2 queue has space → Stage-1 unblocks
         Stage-1 queue has space → Main thread unblocks
         
Result: Slow consumer controls the rate!
```

**Key observation**: Blocking propagates upstream automatically with bounded queues."

---

## Pitfalls and Edge Cases

**Me**: "Common issues:

### Pitfall 1: Unbounded Queues (No Backpressure!)

```java
// ❌ WRONG - No backpressure
BlockingQueue<T> queue = new LinkedBlockingQueue<>();  // Unbounded!

// Fast producer keeps adding, queue grows forever
// OutOfMemoryError!

// ✓ CORRECT - Bounded queue
BlockingQueue<T> queue = new ArrayBlockingQueue<>(100);
// Blocks when full, creating backpressure
```

### Pitfall 2: Dropping Items Instead of Blocking

```java
// Option 1: Block (backpressure)
queue.put(item);  // Blocks if full

// Option 2: Drop (no backpressure)
if (!queue.offer(item)) {
    System.err.println("Queue full, dropping item");
    // Data loss!
}

// Option 3: Timeout
if (!queue.offer(item, 1, TimeUnit.SECONDS)) {
    // Send to dead letter queue
    deadLetterQueue.add(item);
}
```

### Pitfall 3: Poison Pill for Shutdown

```java
// Problem: How to signal workers to stop?

// Solution: Poison pill
static final Object POISON_PILL = new Object();

public void shutdown() {
    // Send poison pills to all workers
    for (int i = 0; i < numWorkers; i++) {
        inputQueue.offer(POISON_PILL);
    }
}

private void processLoop() {
    while (true) {
        Object item = inputQueue.take();
        if (item == POISON_PILL) {
            break;  // Stop processing
        }
        // Process item...
    }
}
```

### Pitfall 4: Exception Handling Stops Worker

```java
// ❌ WRONG - Exception kills worker thread
private void processLoop() {
    while (!shutdown) {
        I item = inputQueue.take();
        O result = processor.apply(item);  // Throws exception → thread dies!
        outputQueue.put(result);
    }
}

// ✓ CORRECT - Catch exceptions
private void processLoop() {
    while (!shutdown) {
        try {
            I item = inputQueue.take();
            O result = processor.apply(item);
            outputQueue.put(result);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            break;
        } catch (Exception e) {
            System.err.println("Processing error: " + e);
            failed.incrementAndGet();
            // Worker continues!
        }
    }
}
```

### Pitfall 5: Deadlock with Circular Pipeline

```java
// ❌ WRONG - Can deadlock!
// Stage-1 → Queue-1 → Stage-2 → Queue-2 → Stage-1 (circular!)

Stage-1: Waiting to put to Queue-2 (full)
Stage-2: Waiting to put to Queue-1 (full)
DEADLOCK!

// Solution: Don't create circular pipelines with bounded queues
```

### Pitfall 6: Resource Leak on Shutdown

```java
// ❌ WRONG - Threads not properly stopped
public void shutdown() {
    shutdown = true;  // Set flag
    // Threads might still be blocked on queue.take()!
}

// ✓ CORRECT - Interrupt threads
public void shutdown() throws InterruptedException {
    shutdown = true;
    workers.shutdownNow();  // Interrupts all threads
    workers.awaitTermination(10, TimeUnit.SECONDS);
}
```
"

---

## Advanced: Non-Blocking Backpressure

**Me**: "For comparison, here's reactive-style backpressure:

### Reactive Streams (Conceptual)

```java
public interface Publisher<T> {
    void subscribe(Subscriber<T> subscriber);
}

public interface Subscriber<T> {
    void onSubscribe(Subscription subscription);
    void onNext(T item);
    void onError(Throwable error);
    void onComplete();
}

public interface Subscription {
    void request(long n);  // Request n items (backpressure!)
    void cancel();
}

// Usage
subscriber.onSubscribe(subscription);
subscription.request(10);  // "I can handle 10 items"

// Publisher only sends 10 items
// When subscriber processes some:
subscription.request(5);  // "I can handle 5 more"
```

**Comparison**:

| Approach | Blocking Pipeline | Reactive Streams |
|----------|------------------|------------------|
| **Mechanism** | Bounded queues block | Request(n) signals capacity |
| **Thread Model** | Thread per stage | Event-driven, fewer threads |
| **Complexity** | Simple | Complex |
| **Memory** | Queues buffer items | Minimal buffering |
| **Use Case** | Traditional ETL | High-throughput streaming |

**For interview**: Stick with blocking pipeline (simpler, easier to reason about)."

---

## Interview Follow-Up Questions

**Q1: How to handle fan-out (one stage to multiple)?**

```java
public class FanOutStage<I, O> {
    private final Function<I, O> processor;
    private final List<BlockingQueue<O>> outputQueues;
    
    public void process(I item) throws InterruptedException {
        O result = processor.apply(item);
        
        // Send to all output queues
        for (BlockingQueue<O> queue : outputQueues) {
            queue.put(result);  // Blocks if any queue is full
        }
    }
}
```

**Q2: How to handle fan-in (multiple stages to one)?**

```java
public class FanInStage<I, O> {
    private final Function<I, O> processor;
    private final List<BlockingQueue<I>> inputQueues;
    private final BlockingQueue<O> outputQueue;
    
    public void start() {
        // One worker per input queue
        for (BlockingQueue<I> inputQueue : inputQueues) {
            new Thread(() -> {
                while (true) {
                    I item = inputQueue.take();
                    O result = processor.apply(item);
                    outputQueue.put(result);
                }
            }).start();
        }
    }
}
```

**Q3: How to monitor backpressure?**

```java
public class MonitoredQueue<T> {
    private final BlockingQueue<T> queue;
    private final AtomicLong totalWaitTime = new AtomicLong(0);
    private final AtomicInteger putAttempts = new AtomicInteger(0);
    
    public void put(T item) throws InterruptedException {
        long start = System.nanoTime();
        queue.put(item);
        long waitTime = System.nanoTime() - start;
        
        totalWaitTime.addAndGet(waitTime);
        putAttempts.incrementAndGet();
    }
    
    public double getAverageWaitTimeMs() {
        int attempts = putAttempts.get();
        return attempts == 0 ? 0 : 
            (totalWaitTime.get() / 1_000_000.0) / attempts;
    }
}

// If average wait time is high → backpressure is happening
```

# Remaining Concurrency Design Problems - Part 3

---

# Write-Behind Cache

## Problem Introduction

**Interviewer**: "Design a write-behind cache where writes go to cache immediately but are asynchronously written to the database in batches."

**Me**: "Interesting pattern! Let me clarify:

**Concept**:
```
Write-Through Cache (Slow):
Application → Cache → Wait for DB write → Return
Latency: ~100ms (DB write time)

Write-Behind Cache (Fast):
Application → Cache → Return immediately
Background: Cache → Batch writes to DB
Latency: ~1ms (memory write)

Benefits:
- Fast writes (don't wait for DB)
- Batch writes (fewer DB calls)
- Coalesce writes (same key updated multiple times)

Risks:
- Data loss if crash before flush
- Eventual consistency
```

**Questions**:
1. **Flush strategy**: Time-based, size-based, or both?
2. **Failure handling**: Retry, dead letter, or alert?
3. **Ordering**: Preserve write order per key?
4. **Coalescing**: Merge multiple writes to same key?
5. **Consistency**: Read-your-writes guarantee?

**Assumptions**:
- Time + size based flushing
- Retry with exponential backoff
- Order preserved per key
- Coalesce writes (keep latest)
- Read-your-writes (read from cache first)

Sound good?"

**Interviewer**: "Yes, focus on the async write mechanism."

---

## High-Level Design

**Me**: "Here's the architecture:

### Components:
```
┌─────────────────────────────────────────┐
│          Write-Behind Cache             │
├─────────────────────────────────────────┤
│  Cache (ConcurrentHashMap)              │
│  ├── Key → CacheEntry                   │
│  │   ├── Value                          │
│  │   ├── isDirty flag                   │
│  │   └── lastModified                   │
├─────────────────────────────────────────┤
│  Dirty Keys Queue (for flush)           │
│  ├── Keys pending DB write              │
├─────────────────────────────────────────┤
│  Background Flusher Thread               │
│  ├── Batches dirty keys                 │
│  ├── Writes to DB                       │
│  └── Retries on failure                 │
└─────────────────────────────────────────┘
```

### Write Flow:
```
1. put(key, value)
   ├─> Update cache (fast!)
   ├─> Mark as dirty
   ├─> Add to dirty queue
   └─> Return immediately

2. Background Flusher
   ├─> Collect batch of dirty keys
   ├─> Write batch to DB
   └─> Mark as clean on success
```

### Key Design Decisions:
- Use `ConcurrentHashMap` for cache
- Use `ConcurrentLinkedQueue` for dirty keys
- Coalesce: Only keep latest value per key"

---

## Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class WriteBehindCache<K, V> {
    
    // Cache entry with metadata
    static class CacheEntry<V> {
        final V value;
        final long timestamp;
        volatile boolean isDirty;
        
        CacheEntry(V value) {
            this.value = value;
            this.timestamp = System.currentTimeMillis();
            this.isDirty = true;
        }
    }
    
    // Database interface
    public interface Database<K, V> {
        void writeBatch(Map<K, V> batch) throws Exception;
        V read(K key) throws Exception;
    }
    
    private final ConcurrentHashMap<K, CacheEntry<V>> cache;
    private final Set<K> dirtyKeys;  // Keys pending flush
    private final Database<K, V> database;
    
    // Configuration
    private final int maxBatchSize;
    private final long flushIntervalMs;
    
    // Background flusher
    private final ScheduledExecutorService flusher;
    private volatile boolean shutdown = false;
    
    // Statistics
    private final AtomicLong cacheHits = new AtomicLong(0);
    private final AtomicLong cacheMisses = new AtomicLong(0);
    private final AtomicLong writes = new AtomicLong(0);
    private final AtomicLong flushes = new AtomicLong(0);
    private final AtomicLong flushFailures = new AtomicLong(0);
    
    public WriteBehindCache(Database<K, V> database, 
                            int maxBatchSize, 
                            long flushIntervalMs) {
        this.database = database;
        this.maxBatchSize = maxBatchSize;
        this.flushIntervalMs = flushIntervalMs;
        
        this.cache = new ConcurrentHashMap<>();
        this.dirtyKeys = ConcurrentHashMap.newKeySet();
        
        // Background flusher
        this.flusher = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "CacheFlusher");
            t.setDaemon(true);
            return t;
        });
        
        flusher.scheduleAtFixedRate(
            this::flushDirtyKeys,
            flushIntervalMs,
            flushIntervalMs,
            TimeUnit.MILLISECONDS
        );
    }
    
    /**
     * Write to cache (returns immediately)
     */
    public void put(K key, V value) {
        if (key == null || value == null) {
            throw new IllegalArgumentException("Key and value cannot be null");
        }
        
        CacheEntry<V> entry = new CacheEntry<>(value);
        cache.put(key, entry);
        dirtyKeys.add(key);
        writes.incrementAndGet();
        
        System.out.println("PUT: " + key + " (dirty keys: " + dirtyKeys.size() + ")");
        
        // Check if batch size reached
        if (dirtyKeys.size() >= maxBatchSize) {
            // Trigger immediate flush
            flusher.execute(this::flushDirtyKeys);
        }
    }
    
    /**
     * Read from cache, fallback to DB
     */
    public V get(K key) {
        CacheEntry<V> entry = cache.get(key);
        
        if (entry != null) {
            cacheHits.incrementAndGet();
            System.out.println("GET: " + key + " - CACHE HIT" + 
                (entry.isDirty ? " (dirty)" : " (clean)"));
            return entry.value;
        }
        
        // Cache miss - load from DB
        cacheMisses.incrementAndGet();
        System.out.println("GET: " + key + " - CACHE MISS, loading from DB");
        
        try {
            V value = database.read(key);
            if (value != null) {
                // Cache it
                CacheEntry<V> newEntry = new CacheEntry<>(value);
                newEntry.isDirty = false;  // Clean (from DB)
                cache.put(key, newEntry);
            }
            return value;
        } catch (Exception e) {
            System.err.println("Error reading from DB: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Force flush all dirty keys
     */
    public void flush() {
        flushDirtyKeys();
    }
    
    /**
     * Background flush operation
     */
    private void flushDirtyKeys() {
        if (dirtyKeys.isEmpty()) {
            return;
        }
        
        // Collect batch of dirty keys
        Set<K> keysToFlush = new HashSet<>();
        Iterator<K> iterator = dirtyKeys.iterator();
        
        while (iterator.hasNext() && keysToFlush.size() < maxBatchSize) {
            K key = iterator.next();
            keysToFlush.add(key);
        }
        
        if (keysToFlush.isEmpty()) {
            return;
        }
        
        System.out.println("\n=== FLUSHING " + keysToFlush.size() + " keys ===");
        
        // Build batch
        Map<K, V> batch = new HashMap<>();
        for (K key : keysToFlush) {
            CacheEntry<V> entry = cache.get(key);
            if (entry != null && entry.isDirty) {
                batch.put(key, entry.value);
            }
        }
        
        // Write batch to DB
        try {
            database.writeBatch(batch);
            flushes.incrementAndGet();
            
            // Mark as clean
            for (K key : batch.keySet()) {
                CacheEntry<V> entry = cache.get(key);
                if (entry != null) {
                    entry.isDirty = false;
                }
                dirtyKeys.remove(key);
            }
            
            System.out.println("Flush successful: " + batch.size() + " keys written");
            
        } catch (Exception e) {
            System.err.println("Flush failed: " + e.getMessage());
            flushFailures.incrementAndGet();
            
            // Retry logic would go here
            retryFlush(batch, 1);
        }
    }
    
    /**
     * Retry failed flush with exponential backoff
     */
    private void retryFlush(Map<K, V> batch, int attempt) {
        if (attempt > 3) {
            System.err.println("Max retries exceeded, giving up on batch");
            // Send to dead letter queue in production
            return;
        }
        
        long delayMs = 1000 * (1L << attempt);  // Exponential backoff
        System.out.println("Scheduling retry #" + attempt + " in " + delayMs + "ms");
        
        flusher.schedule(() -> {
            try {
                database.writeBatch(batch);
                flushes.incrementAndGet();
                
                // Mark as clean
                for (K key : batch.keySet()) {
                    CacheEntry<V> entry = cache.get(key);
                    if (entry != null) {
                        entry.isDirty = false;
                    }
                    dirtyKeys.remove(key);
                }
                
                System.out.println("Retry #" + attempt + " successful");
                
            } catch (Exception e) {
                System.err.println("Retry #" + attempt + " failed: " + e.getMessage());
                retryFlush(batch, attempt + 1);
            }
        }, delayMs, TimeUnit.MILLISECONDS);
    }
    
    public void shutdown() throws InterruptedException {
        shutdown = true;
        
        // Final flush
        System.out.println("\nShutdown: flushing remaining dirty keys");
        flushDirtyKeys();
        
        flusher.shutdown();
        flusher.awaitTermination(10, TimeUnit.SECONDS);
        
        System.out.println("Write-behind cache shut down");
    }
    
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("cacheSize", cache.size());
        stats.put("dirtyKeys", dirtyKeys.size());
        stats.put("cacheHits", cacheHits.get());
        stats.put("cacheMisses", cacheMisses.get());
        stats.put("writes", writes.get());
        stats.put("flushes", flushes.get());
        stats.put("flushFailures", flushFailures.get());
        
        long totalReads = cacheHits.get() + cacheMisses.get();
        stats.put("hitRate", totalReads == 0 ? 0 : (cacheHits.get() * 100 / totalReads));
        
        return stats;
    }
    
    // Test with mock database
    public static void main(String[] args) throws InterruptedException {
        // Mock database
        Database<String, String> db = new Database<String, String>() {
            private final ConcurrentHashMap<String, String> storage = new ConcurrentHashMap<>();
            
            @Override
            public void writeBatch(Map<String, String> batch) throws Exception {
                System.out.println("  [DB] Writing batch of " + batch.size() + " items");
                Thread.sleep(100);  // Simulate slow DB write
                storage.putAll(batch);
            }
            
            @Override
            public String read(String key) throws Exception {
                System.out.println("  [DB] Reading key: " + key);
                Thread.sleep(50);  // Simulate slow DB read
                return storage.get(key);
            }
        };
        
        WriteBehindCache<String, String> cache = new WriteBehindCache<>(
            db,
            5,      // Batch size
            2000    // Flush interval: 2 seconds
        );
        
        System.out.println("=== Scenario 1: Fast Writes ===");
        // Multiple fast writes
        for (int i = 0; i < 3; i++) {
            cache.put("key" + i, "value" + i);
            Thread.sleep(50);
        }
        
        // Read immediately (should hit cache)
        System.out.println("\nReading from cache:");
        System.out.println("key0 = " + cache.get("key0"));
        System.out.println("key1 = " + cache.get("key1"));
        
        System.out.println("\n=== Scenario 2: Size-Based Flush ===");
        // Trigger size-based flush
        for (int i = 10; i < 16; i++) {
            cache.put("key" + i, "value" + i);
            Thread.sleep(50);
        }
        
        Thread.sleep(500);  // Wait for flush to complete
        
        System.out.println("\n=== Scenario 3: Coalescing Writes ===");
        // Multiple writes to same key
        cache.put("coalesce", "v1");
        cache.put("coalesce", "v2");
        cache.put("coalesce", "v3");  // Only v3 should be written to DB
        
        Thread.sleep(3000);  // Wait for time-based flush
        
        System.out.println("\n=== Statistics ===");
        cache.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
        
        cache.shutdown();
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Key gotchas to watch for:

### Pitfall 1: Lost Writes on Crash

```java
// Problem: Cache has dirty data, system crashes
cache.put("key", "value");  // In cache, not in DB yet
// CRASH! Data lost!

// Solution 1: Write-Ahead Log (WAL)
public void put(K key, V value) {
    // Write to WAL first (durable)
    wal.append(new WriteOperation(key, value));
    
    // Then update cache
    cache.put(key, new CacheEntry<>(value));
    dirtyKeys.add(key);
}

// On recovery:
public void recover() {
    for (WriteOperation op : wal.replay()) {
        database.write(op.key, op.value);
    }
}

// Solution 2: Hybrid Write-Through for Critical Data
public void put(K key, V value, boolean critical) {
    cache.put(key, new CacheEntry<>(value));
    
    if (critical) {
        // Write-through for critical data
        database.write(key, value);
    } else {
        // Write-behind for non-critical
        dirtyKeys.add(key);
    }
}
```

### Pitfall 2: Stale Reads After External DB Update

```java
// Problem:
cache.put("key", "value1");     // Cache: value1, DB: (pending)
externalSystem.update("key", "value2");  // DB: value2
cache.get("key");  // Returns value1 (stale!)

// Solution: Cache Invalidation
public void invalidate(K key) {
    cache.remove(key);
    dirtyKeys.remove(key);
}

// Or: TTL on cache entries
if (System.currentTimeMillis() - entry.timestamp > TTL) {
    cache.remove(key);  // Expired, reload from DB
}
```

### Pitfall 3: Memory Leak with Unbounded Dirty Keys

```java
// ❌ WRONG - Dirty keys can accumulate if flush fails
if (database.writeBatch(batch) fails) {
    // Keys remain in dirtyKeys forever!
    // Memory leak!
}

// ✓ CORRECT - Bounded retry, then drop
private void retryFlush(Map<K, V> batch, int attempt) {
    if (attempt > MAX_RETRIES) {
        // Give up, send to dead letter queue
        deadLetterQueue.addAll(batch.entrySet());
        
        // Remove from dirty keys
        for (K key : batch.keySet()) {
            dirtyKeys.remove(key);
        }
        return;
    }
    // Retry...
}
```

### Pitfall 4: Concurrent Modification During Flush

```java
// Problem:
Thread 1 (Flusher): Reading entry.value
Thread 2 (Writer):  cache.put(key, newValue) - updates same entry

// Solution: Snapshot the value
private void flushDirtyKeys() {
    Map<K, V> batch = new HashMap<>();
    
    for (K key : keysToFlush) {
        CacheEntry<V> entry = cache.get(key);
        if (entry != null && entry.isDirty) {
            // Snapshot value at flush time
            batch.put(key, entry.value);
        }
    }
    
    // Write snapshot (immune to concurrent updates)
    database.writeBatch(batch);
}
```

### Pitfall 5: Read-After-Write Consistency

```java
// Problem: Different threads, different views
Thread 1: cache.put("key", "value");  // Cache updated
Thread 2: db.read("key");  // Returns old value (not flushed yet)

// Solution: Always read from cache first
public V get(K key) {
    // Check cache first (includes dirty data)
    CacheEntry<V> entry = cache.get(key);
    if (entry != null) {
        return entry.value;  // Read-your-writes!
    }
    
    // Cache miss, load from DB
    return database.read(key);
}
```

### Pitfall 6: Ordering Issues with Multiple Keys

```java
// Problem: Order not preserved across keys
cache.put("account", "100");
cache.put("balance", "50");

// Flush happens in arbitrary order
// DB might see balance=50 before account=100!

// Solution: Transaction groups
public void putBatch(Map<K, V> batch) {
    String txId = UUID.randomUUID().toString();
    
    for (Map.Entry<K, V> entry : batch.entrySet()) {
        CacheEntry<V> ce = new CacheEntry<>(entry.getValue());
        ce.transactionId = txId;
        cache.put(entry.getKey(), ce);
        dirtyKeys.add(entry.getKey());
    }
}

// Flush keeps transaction together
private void flushDirtyKeys() {
    // Group by transaction ID
    Map<String, Map<K, V>> txBatches = groupByTransaction();
    
    for (Map<K, V> txBatch : txBatches.values()) {
        database.writeTransaction(txBatch);  // Atomic
    }
}
```
"

---

## Advanced: Write-Behind with Write Coalescing

**Me**: "Optimized version that coalesces multiple writes:

```java
public class CoalescingWriteBehindCache<K, V> {
    
    static class CacheEntry<V> {
        volatile V value;  // Mutable for coalescing
        final AtomicLong version;  // Track updates
        volatile boolean isDirty;
        
        CacheEntry(V value) {
            this.value = value;
            this.version = new AtomicLong(1);
            this.isDirty = true;
        }
        
        void update(V newValue) {
            this.value = newValue;
            this.version.incrementAndGet();
            this.isDirty = true;
        }
    }
    
    public void put(K key, V value) {
        CacheEntry<V> entry = cache.get(key);
        
        if (entry != null) {
            // Key exists, coalesce write
            entry.update(value);
            System.out.println("PUT: " + key + " - COALESCED (version: " + 
                entry.version.get() + ")");
        } else {
            // New key
            entry = new CacheEntry<>(value);
            cache.put(key, entry);
            dirtyKeys.add(key);
            System.out.println("PUT: " + key + " - NEW");
        }
        
        writes.incrementAndGet();
    }
    
    // When flushing, only latest value is written
    // Multiple writes to same key = single DB write!
}
```

**Benefits**:
- Multiple updates to same key = single DB write
- Reduces DB load significantly
- Example: 1000 updates to same key = 1 DB write

**Trade-off**:
- Intermediate values never written to DB
- Only latest value persists
- Fine for most use cases (e.g., session data, counters)
"

---

## Interview Follow-Up Questions

**Q1: How to implement write-behind for delete operations?**

```java
public void delete(K key) {
    cache.remove(key);
    
    // Add to delete queue
    keysToDelete.add(key);
}

private void flushDeletes() {
    Set<K> batch = new HashSet<>();
    Iterator<K> iter = keysToDelete.iterator();
    
    while (iter.hasNext() && batch.size() < maxBatchSize) {
        batch.add(iter.next());
    }
    
    database.deleteBatch(batch);
    keysToDelete.removeAll(batch);
}
```

**Q2: How to handle cache eviction of dirty data?**

```java
class EvictionListener implements RemovalListener<K, CacheEntry<V>> {
    @Override
    public void onRemoval(K key, CacheEntry<V> entry) {
        if (entry.isDirty) {
            // Dirty data being evicted!
            // Option 1: Flush immediately
            database.write(key, entry.value);
            
            // Option 2: Add to priority flush queue
            urgentFlushQueue.add(key);
            
            // Option 3: Prevent eviction of dirty data
            throw new IllegalStateException("Cannot evict dirty data");
        }
    }
}
```

**Q3: How to test write-behind cache?**

```java
@Test
public void testWriteCoalescing() throws InterruptedException {
    MockDatabase db = new MockDatabase();
    WriteBehindCache<String, String> cache = 
        new WriteBehindCache<>(db, 10, 1000);
    
    // Multiple writes to same key
    cache.put("key", "v1");
    cache.put("key", "v2");
    cache.put("key", "v3");
    
    // Force flush
    cache.flush();
    
    // Verify only one DB write
    assertEquals(1, db.getWriteCount("key"));
    assertEquals("v3", db.read("key"));  // Latest value
}

@Test
public void testReadYourWrites() {
    cache.put("key", "value");
    
    // Should read from cache (dirty)
    assertEquals("value", cache.get("key"));
    
    // Even before flush to DB
    assertFalse(db.contains("key"));
}
```

---

# Real-Time Metrics Aggregator

## Problem Introduction

**Interviewer**: "Design a system that collects metrics from multiple threads and provides real-time aggregated statistics (count, sum, average, percentiles)."

**Me**: "Great problem! Let me clarify:

**Use Case**:
```
Web Server Metrics:
- 1000s of requests/second
- Each thread reports: latency, status code, size
- Need real-time dashboard showing:
  - Request count
  - Average latency
  - P50, P95, P99 latency
  - Error rate
  - Throughput

Requirements:
- Low overhead (don't slow down requests)
- Thread-safe updates
- Real-time queries
```

**Questions**:
1. **Update frequency**: Millions per second?
2. **Metrics types**: Counters, gauges, histograms?
3. **Accuracy**: Exact or approximate OK?
4. **Time windows**: Rolling windows or cumulative?
5. **Percentiles**: Exact or approximate?

**Assumptions**:
- High frequency (100k+ updates/sec)
- Counters, histograms, timers
- Approximate OK for percentiles
- Rolling 1-minute windows
- Use efficient data structures

Sound good?"

**Interviewer**: "Yes, focus on efficient concurrent updates."

---

## High-Level Design

**Me**: "Here's the approach:

### Metric Types:
```
1. Counter: Simple increment (requests, errors)
   → Use LongAdder (lock-free, high-contention optimized)

2. Gauge: Current value (active connections)
   → Use AtomicLong

3. Histogram: Distribution (latency, size)
   → Use HdrHistogram or approximate sketch

4. Timer: Measure duration
   → Histogram + Counter
```

### Key Design:
```
MetricsAggregator
├── Counters (LongAdder)
├── Gauges (AtomicLong)
├── Histograms (ConcurrentHashMap<String, Histogram>)
└── Time Windows (Sliding window)
```

### Lock-Free Optimization:
Use `LongAdder` instead of `AtomicLong` for counters - much faster under contention!"

---

## Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class MetricsAggregator {
    
    // Counter: monotonically increasing
    static class Counter {
        private final LongAdder count = new LongAdder();
        
        void increment() {
            count.increment();
        }
        
        void add(long delta) {
            count.add(delta);
        }
        
        long get() {
            return count.sum();
        }
        
        void reset() {
            count.reset();
        }
    }
    
    // Gauge: point-in-time value
    static class Gauge {
        private final AtomicLong value = new AtomicLong(0);
        
        void set(long newValue) {
            value.set(newValue);
        }
        
        void increment() {
            value.incrementAndGet();
        }
        
        void decrement() {
            value.decrementAndGet();
        }
        
        long get() {
            return value.get();
        }
    }
    
    // Histogram: distribution of values
    static class Histogram {
        private final ConcurrentHashMap<Long, LongAdder> buckets;
        private final LongAdder count;
        private final LongAdder sum;
        private final AtomicLong min;
        private final AtomicLong max;
        
        Histogram() {
            this.buckets = new ConcurrentHashMap<>();
            this.count = new LongAdder();
            this.sum = new LongAdder();
            this.min = new AtomicLong(Long.MAX_VALUE);
            this.max = new AtomicLong(Long.MIN_VALUE);
        }
        
        void record(long value) {
            // Update count and sum
            count.increment();
            sum.add(value);
            
            // Update min/max
            updateMin(value);
            updateMax(value);
            
            // Update histogram buckets (logarithmic)
            long bucket = bucketFor(value);
            buckets.computeIfAbsent(bucket, k -> new LongAdder()).increment();
        }
        
        private void updateMin(long value) {
            long current;
            do {
                current = min.get();
                if (value >= current) return;
            } while (!min.compareAndSet(current, value));
        }
        
        private void updateMax(long value) {
            long current;
            do {
                current = max.get();
                if (value <= current) return;
            } while (!max.compareAndSet(current, value));
        }
        
        private long bucketFor(long value) {
            // Logarithmic buckets: 0-1, 1-2, 2-4, 4-8, 8-16, ...
            if (value <= 0) return 0;
            return 1L << (63 - Long.numberOfLeadingZeros(value));
        }
        
        HistogramSnapshot snapshot() {
            return new HistogramSnapshot(
                count.sum(),
                sum.sum(),
                min.get() == Long.MAX_VALUE ? 0 : min.get(),
                max.get() == Long.MIN_VALUE ? 0 : max.get(),
                new HashMap<>(buckets)  // Copy for thread safety
            );
        }
    }
    
    static class HistogramSnapshot {
        final long count;
        final long sum;
        final long min;
        final long max;
        final Map<Long, LongAdder> buckets;
        
        HistogramSnapshot(long count, long sum, long min, long max, 
                         Map<Long, LongAdder> buckets) {
            this.count = count;
            this.sum = sum;
            this.min = min;
            this.max = max;
            this.buckets = buckets;
        }
        
        double mean() {
            return count == 0 ? 0 : (double) sum / count;
        }
        
        long percentile(double p) {
            if (count == 0) return 0;
            
            long target = (long) (count * p / 100.0);
            long cumulative = 0;
            
            // Sort buckets
            List<Long> sortedBuckets = new ArrayList<>(buckets.keySet());
            Collections.sort(sortedBuckets);
            
            for (Long bucket : sortedBuckets) {
                cumulative += buckets.get(bucket).sum();
                if (cumulative >= target) {
                    return bucket;
                }
            }
            
            return max;
        }
        
        @Override
        public String toString() {
            return String.format(
                "count=%d, min=%d, max=%d, mean=%.2f, p50=%d, p95=%d, p99=%d",
                count, min, max, mean(), 
                percentile(50), percentile(95), percentile(99)
            );
        }
    }
    
    // Main aggregator
    private final ConcurrentHashMap<String, Counter> counters;
    private final ConcurrentHashMap<String, Gauge> gauges;
    private final ConcurrentHashMap<String, Histogram> histograms;
    
    public MetricsAggregator() {
        this.counters = new ConcurrentHashMap<>();
        this.gauges = new ConcurrentHashMap<>();
        this.histograms = new ConcurrentHashMap<>();
    }
    
    // Counter operations
    public void incrementCounter(String name) {
        counters.computeIfAbsent(name, k -> new Counter()).increment();
    }
    
    public void addToCounter(String name, long delta) {
        counters.computeIfAbsent(name, k -> new Counter()).add(delta);
    }
    
    public long getCounter(String name) {
        Counter counter = counters.get(name);
        return counter != null ? counter.get() : 0;
    }
    
    // Gauge operations
    public void setGauge(String name, long value) {
        gauges.computeIfAbsent(name, k -> new Gauge()).set(value);
    }
    
    public void incrementGauge(String name) {
        gauges.computeIfAbsent(name, k -> new Gauge()).increment();
    }
    
    public void decrementGauge(String name) {
        gauges.computeIfAbsent(name, k -> new Gauge()).decrement();
    }
    
    public long getGauge(String name) {
        Gauge gauge = gauges.get(name);
        return gauge != null ? gauge.get() : 0;
    }
    
    // Histogram operations
    public void recordValue(String name, long value) {
        histograms.computeIfAbsent(name, k -> new Histogram()).record(value);
    }
    
    public HistogramSnapshot getHistogram(String name) {
        Histogram histogram = histograms.get(name);
        return histogram != null ? histogram.snapshot() : 
            new HistogramSnapshot(0, 0, 0, 0, Collections.emptyMap());
    }
    
    // Timer helper
    public TimerContext startTimer(String name) {
        return new TimerContext(this, name);
    }
    
    static class TimerContext implements AutoCloseable {
        private final MetricsAggregator aggregator;
        private final String name;
        private final long startTime;
        
        TimerContext(MetricsAggregator aggregator, String name) {
            this.aggregator = aggregator;
            this.name = name;
            this.startTime = System.nanoTime();
        }
        
        @Override
        public void close() {
            long durationNs = System.nanoTime() - startTime;
            long durationMs = durationNs / 1_000_000;
            aggregator.recordValue(name, durationMs);
        }
    }
    
    // Get all metrics
    public Map<String, Object> getAllMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        Map<String, Long> counterValues = new HashMap<>();
        counters.forEach((name, counter) -> counterValues.put(name, counter.get()));
        metrics.put("counters", counterValues);
        
        Map<String, Long> gaugeValues = new HashMap<>();
        gauges.forEach((name, gauge) -> gaugeValues.put(name, gauge.get()));
        metrics.put("gauges", gaugeValues);
        
        Map<String, String> histogramValues = new HashMap<>();
        histograms.forEach((name, histogram) -> 
            histogramValues.put(name, histogram.snapshot().toString()));
        metrics.put("histograms", histogramValues);
        
        return metrics;
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        MetricsAggregator metrics = new MetricsAggregator();
        
        System.out.println("=== Simulating Web Server Metrics ===\n");
        
        // Simulate multiple threads handling requests
        ExecutorService executor = Executors.newFixedThreadPool(10);
        CountDownLatch latch = new CountDownLatch(10);
        
        for (int i = 0; i < 10; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < 1000; j++) {
                        // Increment request counter
                        metrics.incrementCounter("requests");
                        
                        // Update active connections gauge
                        metrics.incrementGauge("activeConnections");
                        
                        // Measure request latency
                        try (TimerContext timer = metrics.startTimer("requestLatency")) {
                            // Simulate request processing
                            int latency = 10 + (int) (Math.random() * 100);
                            Thread.sleep(latency);
                        }
                        
                        // Record response size
                        long size = 1000 + (long) (Math.random() * 10000);
                        metrics.recordValue("responseSize", size);
                        
                        // Random errors
                        if (Math.random() < 0.05) {
                            metrics.incrementCounter("errors");
                        }
                        
                        metrics.decrementGauge("activeConnections");
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        // Print metrics periodically
        for (int i = 0; i < 5; i++) {
            Thread.sleep(2000);
            System.out.println("\n=== Metrics Snapshot (t=" + (i * 2) + "s) ===");
            System.out.println("Requests: " + metrics.getCounter("requests"));
            System.out.println("Errors: " + metrics.getCounter("errors"));
            System.out.println("Active Connections: " + metrics.getGauge("activeConnections"));
            System.out.println("Request Latency: " + metrics.getHistogram("requestLatency"));
            System.out.println("Response Size: " + metrics.getHistogram("responseSize"));
        }
        
        latch.await();
        executor.shutdown();
        
        System.out.println("\n=== Final Metrics ===");
        metrics.getAllMetrics().forEach((key, value) -> {
            System.out.println(key + ": " + value);
        });
    }
}
```

---

## Why LongAdder vs AtomicLong?

**Me**: "Let me explain the performance difference:

### AtomicLong (CAS on single variable)

```java
class AtomicLong {
    private volatile long value;
    
    long incrementAndGet() {
        while (true) {
            long current = value;
            long next = current + 1;
            if (compareAndSet(current, next)) {
                return next;
            }
            // Retry on contention
        }
    }
}

// Under high contention (16 threads):
Thread-1: CAS → success
Thread-2: CAS → fail → retry
Thread-3: CAS → fail → retry
Thread-4: CAS → fail → retry
...
Many retries!
```

### LongAdder (Striped counters)

```java
class LongAdder {
    // Multiple cells, threads update different cells
    private Cell[] cells;
    private volatile long base;
    
    void increment() {
        int probe = Thread.currentThread().threadLocalRandomProbe();
        Cell cell = cells[probe % cells.length];
        cell.add(1);  // Less contention per cell!
    }
    
    long sum() {
        long total = base;
        for (Cell cell : cells) {
            total += cell.value;
        }
        return total;
    }
}

// Under high contention (16 threads, 4 cells):
Thread-1: Update cell-0 ✓
Thread-2: Update cell-1 ✓ (different cell!)
Thread-3: Update cell-2 ✓
Thread-4: Update cell-3 ✓
Thread-5: Update cell-0 (small contention)
...
Much less contention!
```

### Performance Comparison

```java
// Benchmark
public class LongAdderVsAtomicLong {
    public static void main(String[] args) throws InterruptedException {
        int threads = 32;
        int iterations = 1_000_000;
        
        // AtomicLong
        AtomicLong atomicCounter = new AtomicLong();
        long start = System.nanoTime();
        runTest(threads, iterations, atomicCounter::incrementAndGet);
        long atomicTime = System.nanoTime() - start;
        
        // LongAdder
        LongAdder adderCounter = new LongAdder();
        start = System.nanoTime();
        runTest(threads, iterations, adderCounter::increment);
        long adderTime = System.nanoTime() - start;
        
        System.out.println("AtomicLong: " + atomicTime / 1_000_000 + " ms");
        System.out.println("LongAdder:  " + adderTime / 1_000_000 + " ms");
        System.out.println("Speedup:    " + (double)atomicTime / adderTime + "x");
    }
}

// Typical results:
// AtomicLong: 2500 ms
// LongAdder:  450 ms
// Speedup: 5.5x
```

**When to use each**:
- **AtomicLong**: Low contention, need exact value frequently
- **LongAdder**: High contention (8+ threads), infrequent reads
"

---

## Advanced: Sliding Time Windows

**Me**: "For time-based metrics:

```java
public class SlidingWindowCounter {
    private static final int WINDOW_SIZE_MS = 60_000;  // 1 minute
    private static final int BUCKET_SIZE_MS = 1_000;   // 1 second
    private static final int NUM_BUCKETS = WINDOW_SIZE_MS / BUCKET_SIZE_MS;
    
    static class TimeBucket {
        final long timestamp;
        final LongAdder count;
        
        TimeBucket(long timestamp) {
            this.timestamp = timestamp;
            this.count = new LongAdder();
        }
    }
    
    private final TimeBucket[] buckets = new TimeBucket[NUM_BUCKETS];
    private final AtomicInteger currentIndex = new AtomicInteger(0);
    
    public SlidingWindowCounter() {
        long now = System.currentTimeMillis();
        for (int i = 0; i < NUM_BUCKETS; i++) {
            buckets[i] = new TimeBucket(now - (NUM_BUCKETS - i) * BUCKET_SIZE_MS);
        }
    }
    
    public void increment() {
        long now = System.currentTimeMillis();
        int index = (int) ((now / BUCKET_SIZE_MS) % NUM_BUCKETS);
        
        TimeBucket bucket = buckets[index];
        
        // Check if bucket is stale
        if (now - bucket.timestamp >= WINDOW_SIZE_MS) {
            // Reset bucket
            synchronized (bucket) {
                if (now - bucket.timestamp >= WINDOW_SIZE_MS) {
                    bucket.count.reset();
                    buckets[index] = new TimeBucket(now);
                }
            }
        }
        
        buckets[index].count.increment();
    }
    
    public long getCount() {
        long now = System.currentTimeMillis();
        long total = 0;
        
        for (TimeBucket bucket : buckets) {
            if (now - bucket.timestamp < WINDOW_SIZE_MS) {
                total += bucket.count.sum();
            }
        }
        
        return total;
    }
    
    public double getRate() {
        return getCount() / (WINDOW_SIZE_MS / 1000.0);  // per second
    }
}
```

**Benefits**:
- Get rate over last N seconds
- Automatically expires old data
- Lock-free increments (LongAdder per bucket)
"
# Remaining Concurrency Design Problems - Part 4

---

# Producer-Consumer with Multiple Queues

## Problem Introduction

**Interviewer**: "Design a producer-consumer system with multiple queues. Tasks should be routed to different queues based on priority or type, and multiple consumers process from these queues."

**Me**: "Interesting! Let me clarify:

**Scenario**:
```
Producers → [High Priority Queue]   → Consumer Pool 1
         → [Medium Priority Queue] → Consumer Pool 2
         → [Low Priority Queue]    → Consumer Pool 3

Or:

Producers → Route by type → [Queue-A] → Consumers
                          → [Queue-B] → Consumers
                          → [Queue-C] → Consumers
```

**Questions**:
1. **Routing strategy**: Priority-based, type-based, or round-robin?
2. **Consumer allocation**: Dedicated consumers per queue or shared pool?
3. **Starvation**: How to prevent low-priority queue starvation?
4. **Load balancing**: Static or dynamic consumer allocation?
5. **Ordering**: Preserve order within queue or globally?

**Assumptions**:
- Priority-based routing (high, medium, low)
- Shared consumer pool with weighted consumption
- Prevent starvation with fair scheduling
- Dynamic load balancing
- Order within priority level

Sound good?"

**Interviewer**: "Yes, focus on preventing starvation of low-priority tasks."

---

## High-Level Design

**Me**: "Here's the architecture:

### Structure:
```
┌──────────────────────────────────────────────┐
│         Multi-Queue System                   │
├──────────────────────────────────────────────┤
│  High Priority Queue [===]   (weight: 70%)  │
│  Med Priority Queue  [====]  (weight: 20%)  │
│  Low Priority Queue  [=====] (weight: 10%)  │
├──────────────────────────────────────────────┤
│         Consumer Pool (shared)               │
│  [Consumer-1] [Consumer-2] ... [Consumer-N] │
└──────────────────────────────────────────────┘
```

### Fair Consumption Strategy:
```
Out of 10 tasks consumed:
- 7 from high priority (70%)
- 2 from medium priority (20%)
- 1 from low priority (10%)

Even if high priority queue has 1000 tasks,
low priority still gets processed!
```

### Key Design:
- Weighted round-robin selection
- BlockingQueue for each priority
- Shared consumer pool"

---

## Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class MultiQueueProcessor<T> {
    
    enum Priority {
        HIGH(70),    // 70% of consumption
        MEDIUM(20),  // 20% of consumption
        LOW(10);     // 10% of consumption
        
        final int weight;
        Priority(int weight) { this.weight = weight; }
    }
    
    static class Task<T> {
        final T data;
        final Priority priority;
        
        Task(T data, Priority priority) {
            this.data = data;
            this.priority = priority;
        }
    }
    
    private final Map<Priority, BlockingQueue<Task<T>>> queues;
    private final ExecutorService consumers;
    private final int numConsumers;
    private volatile boolean shutdown = false;
    
    // Task processor
    private final java.util.function.Consumer<T> processor;
    
    // Statistics per priority
    private final Map<Priority, AtomicLong> processedCount;
    private final Map<Priority, AtomicLong> queuedCount;
    
    public MultiQueueProcessor(int numConsumers, 
                               int queueSize,
                               java.util.function.Consumer<T> processor) {
        this.numConsumers = numConsumers;
        this.processor = processor;
        
        // Initialize queues for each priority
        this.queues = new EnumMap<>(Priority.class);
        this.processedCount = new EnumMap<>(Priority.class);
        this.queuedCount = new EnumMap<>(Priority.class);
        
        for (Priority p : Priority.values()) {
            queues.put(p, new ArrayBlockingQueue<>(queueSize));
            processedCount.put(p, new AtomicLong(0));
            queuedCount.put(p, new AtomicLong(0));
        }
        
        // Start consumers
        this.consumers = Executors.newFixedThreadPool(numConsumers);
        for (int i = 0; i < numConsumers; i++) {
            consumers.submit(new Consumer(i));
        }
        
        System.out.println("Multi-queue processor started with " + 
            numConsumers + " consumers");
    }
    
    /**
     * Submit task with priority
     */
    public void submit(T data, Priority priority) throws InterruptedException {
        if (shutdown) {
            throw new IllegalStateException("Processor is shut down");
        }
        
        Task<T> task = new Task<>(data, priority);
        queues.get(priority).put(task);
        queuedCount.get(priority).incrementAndGet();
        
        System.out.println(Thread.currentThread().getName() + 
            " submitted " + priority + " priority task (queue size: " + 
            queues.get(priority).size() + ")");
    }
    
    /**
     * Consumer that uses weighted selection
     */
    class Consumer implements Runnable {
        private final int id;
        private final Random random = new Random();
        
        Consumer(int id) {
            this.id = id;
        }
        
        @Override
        public void run() {
            System.out.println("Consumer-" + id + " started");
            
            while (!shutdown) {
                try {
                    // Select queue based on weighted probability
                    Priority priority = selectQueueWeighted();
                    
                    // Try to get task from selected queue
                    Task<T> task = queues.get(priority).poll(100, TimeUnit.MILLISECONDS);
                    
                    if (task != null) {
                        System.out.println("Consumer-" + id + 
                            " processing " + task.priority + " task");
                        
                        processor.accept(task.data);
                        processedCount.get(task.priority).incrementAndGet();
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    System.err.println("Consumer-" + id + " error: " + e.getMessage());
                }
            }
            
            System.out.println("Consumer-" + id + " stopped");
        }
        
        /**
         * Weighted random selection
         */
        private Priority selectQueueWeighted() {
            int rand = random.nextInt(100);
            
            // 0-69: HIGH (70%)
            if (rand < 70) {
                return Priority.HIGH;
            }
            // 70-89: MEDIUM (20%)
            else if (rand < 90) {
                return Priority.MEDIUM;
            }
            // 90-99: LOW (10%)
            else {
                return Priority.LOW;
            }
        }
    }
    
    /**
     * Alternative: Round-robin with skip empty
     */
    class FairConsumer implements Runnable {
        private final int id;
        private int currentPriorityIndex = 0;
        private final Priority[] priorities = Priority.values();
        
        @Override
        public void run() {
            while (!shutdown) {
                try {
                    // Try each queue in round-robin
                    boolean foundTask = false;
                    
                    for (int i = 0; i < priorities.length; i++) {
                        Priority p = priorities[currentPriorityIndex];
                        currentPriorityIndex = (currentPriorityIndex + 1) % priorities.length;
                        
                        Task<T> task = queues.get(p).poll(10, TimeUnit.MILLISECONDS);
                        
                        if (task != null) {
                            processor.accept(task.data);
                            processedCount.get(p).incrementAndGet();
                            foundTask = true;
                            break;
                        }
                    }
                    
                    if (!foundTask) {
                        Thread.sleep(10);  // All queues empty
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }
    
    public void shutdown() throws InterruptedException {
        shutdown = true;
        consumers.shutdown();
        consumers.awaitTermination(10, TimeUnit.SECONDS);
        
        System.out.println("\n=== Shutdown Complete ===");
        printStatistics();
    }
    
    public void printStatistics() {
        System.out.println("\nStatistics:");
        for (Priority p : Priority.values()) {
            long queued = queuedCount.get(p).get();
            long processed = processedCount.get(p).get();
            int remaining = queues.get(p).size();
            
            System.out.printf("%s: queued=%d, processed=%d, remaining=%d\n",
                p, queued, processed, remaining);
        }
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        MultiQueueProcessor<String> processor = new MultiQueueProcessor<>(
            3,      // 3 consumers
            100,    // Queue size
            data -> {
                try {
                    Thread.sleep(50);  // Simulate processing
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        );
        
        // Submit tasks with different priorities
        ExecutorService producers = Executors.newFixedThreadPool(5);
        
        for (int i = 0; i < 5; i++) {
            final int producerId = i;
            producers.submit(() -> {
                try {
                    for (int j = 0; j < 20; j++) {
                        Priority p;
                        if (j % 10 == 0) {
                            p = Priority.LOW;
                        } else if (j % 3 == 0) {
                            p = Priority.MEDIUM;
                        } else {
                            p = Priority.HIGH;
                        }
                        
                        processor.submit("task-" + producerId + "-" + j, p);
                        Thread.sleep(10);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        
        producers.shutdown();
        producers.awaitTermination(30, TimeUnit.SECONDS);
        
        Thread.sleep(5000);  // Let consumers finish
        
        processor.shutdown();
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Key gotchas:

### Pitfall 1: Starvation of Low Priority

```java
// ❌ WRONG - Always check high priority first
Task task = highQueue.poll();
if (task == null) {
    task = mediumQueue.poll();
}
if (task == null) {
    task = lowQueue.poll();
}

// If high queue always has tasks, low queue never processed!

// ✓ CORRECT - Weighted selection or round-robin
Priority selected = selectWithWeight();
Task task = queues.get(selected).poll();
```

### Pitfall 2: Inefficient Weighted Selection

```java
// ❌ WRONG - Check each queue every time
for (int i = 0; i < 7; i++) {
    task = highQueue.poll();
    if (task != null) process(task);
}
for (int i = 0; i < 2; i++) {
    task = mediumQueue.poll();
    if (task != null) process(task);
}
// Complex, inefficient

// ✓ CORRECT - Random weighted selection
int rand = random.nextInt(100);
if (rand < 70) return HIGH;
else if (rand < 90) return MEDIUM;
else return LOW;
```

### Pitfall 3: Unbounded Queue Growth

```java
// Problem: High priority queue fills up, producers blocked
// Meanwhile low priority queue is empty

// Solution: Cross-queue load balancing
public void submit(T data, Priority priority) throws InterruptedException {
    BlockingQueue<Task<T>> queue = queues.get(priority);
    
    if (!queue.offer(data)) {
        // Queue full, try lower priority
        Priority lowerPriority = priority.lower();
        if (lowerPriority != null) {
            queues.get(lowerPriority).put(data);
            System.out.println("Downgraded task from " + priority + 
                " to " + lowerPriority);
        } else {
            queue.put(data);  // Block on lowest queue
        }
    }
}
```

### Pitfall 4: Work Stealing

```java
// Better approach: Work stealing
class WorkStealingConsumer implements Runnable {
    @Override
    public void run() {
        while (!shutdown) {
            Task task = null;
            
            // Try own queue first
            task = myQueue.poll();
            
            // If empty, steal from other queues
            if (task == null) {
                for (BlockingQueue<Task> otherQueue : allQueues) {
                    task = otherQueue.poll();
                    if (task != null) {
                        System.out.println("Stole task from other queue");
                        break;
                    }
                }
            }
            
            if (task != null) {
                process(task);
            } else {
                Thread.sleep(10);
            }
        }
    }
}
```

### Pitfall 5: Priority Inversion

```java
// Problem:
High priority task needs result from low priority task
Low priority task stuck behind many other low priority tasks
High priority task waits!

// Solution: Priority inheritance
when high priority task waits on low priority task:
    boost low priority task to high priority
    process it immediately
    restore original priority
```

### Pitfall 6: Thundering Herd on Queue Selection

```java
// ❌ WRONG - All consumers check same queue
synchronized (queues) {
    Priority p = selectQueue();
    return queues.get(p).poll();
}
// Serializes queue selection!

// ✓ CORRECT - Lock-free selection
Priority p = selectQueueWeighted();  // No lock
return queues.get(p).poll(100, TimeUnit.MILLISECONDS);
```
"

---

## Advanced: Dynamic Priority Adjustment

**Me**: "Production systems often adjust priorities dynamically:

```java
public class AdaptiveMultiQueue<T> {
    
    static class QueueStats {
        final AtomicLong size = new AtomicLong(0);
        final AtomicLong processed = new AtomicLong(0);
        final AtomicLong waitTime = new AtomicLong(0);
        
        double getAverageWaitTime() {
            long p = processed.get();
            return p == 0 ? 0 : waitTime.get() / (double) p;
        }
    }
    
    private final Map<Priority, QueueStats> stats;
    
    /**
     * Adaptive weight calculation based on queue depth
     */
    private Priority selectQueueAdaptive() {
        // If low priority queue is getting too deep, boost its weight
        long lowSize = queues.get(Priority.LOW).size();
        long highSize = queues.get(Priority.HIGH).size();
        
        if (lowSize > 100 && highSize < 10) {
            // Temporarily boost low priority
            return Priority.LOW;
        }
        
        // Otherwise use normal weights
        return selectQueueWeighted();
    }
    
    /**
     * Age-based priority boost
     */
    static class AgingTask<T> extends Task<T> {
        final long submittedAt;
        
        AgingTask(T data, Priority priority) {
            super(data, priority);
            this.submittedAt = System.currentTimeMillis();
        }
        
        Priority getEffectivePriority() {
            long age = System.currentTimeMillis() - submittedAt;
            
            // After 5 seconds, boost priority
            if (age > 5000 && priority == Priority.LOW) {
                return Priority.MEDIUM;
            }
            if (age > 10000 && priority == Priority.MEDIUM) {
                return Priority.HIGH;
            }
            
            return priority;
        }
    }
}
```
"

---

## Interview Follow-Up Questions

**Q1: How would you implement work stealing between queues?**

```java
class WorkStealingConsumer implements Runnable {
    private final List<BlockingQueue<Task>> allQueues;
    private int preferredQueue;
    
    @Override
    public void run() {
        while (!shutdown) {
            Task task = null;
            
            // Try preferred queue
            task = allQueues.get(preferredQueue).poll();
            
            // Try stealing from others
            if (task == null) {
                for (int i = 0; i < allQueues.size(); i++) {
                    if (i == preferredQueue) continue;
                    
                    task = allQueues.get(i).poll();
                    if (task != null) {
                        System.out.println("Stole from queue " + i);
                        break;
                    }
                }
            }
            
            if (task != null) {
                process(task);
            }
        }
    }
}
```

**Q2: How to monitor and alert on starvation?**

```java
class StarvationMonitor {
    private final ScheduledExecutorService monitor;
    
    public StarvationMonitor() {
        this.monitor = Executors.newSingleThreadScheduledExecutor();
        
        monitor.scheduleAtFixedRate(() -> {
            for (Priority p : Priority.values()) {
                QueueStats stats = queueStats.get(p);
                
                double avgWaitTime = stats.getAverageWaitTime();
                int queueSize = queues.get(p).size();
                
                // Alert if low priority queue is starving
                if (p == Priority.LOW && avgWaitTime > 5000) {
                    System.err.println("ALERT: Low priority queue starving! " +
                        "Avg wait: " + avgWaitTime + "ms, size: " + queueSize);
                }
            }
        }, 10, 10, TimeUnit.SECONDS);
    }
}
```

---

# Concurrent Data Deduplicator

## Problem Introduction

**Interviewer**: "Design a deduplicator that processes a stream of data and ensures each unique item is processed only once, even with concurrent threads submitting items."

**Me**: "Interesting! Let me clarify:

**Scenario**:
```
Stream of events:
E1, E2, E1 (duplicate), E3, E2 (duplicate), E4

Output:
E1 → processed
E2 → processed
E1 → skipped (duplicate)
E3 → processed
E2 → skipped (duplicate)
E4 → processed
```

**Questions**:
1. **Deduplication scope**: All-time or within time window?
2. **Identity**: By hash, by field, or custom equals?
3. **Scale**: Millions or billions of unique items?
4. **Memory**: Unbounded or need eviction?
5. **Processing**: Sync or async after dedup?

**Assumptions**:
- Time-window based (last 1 hour)
- By object equality
- Millions of items
- LRU eviction if memory limit
- Sync processing (return immediately)

Sound good?"

**Interviewer**: "Yes, focus on efficient concurrent deduplication."

---

## High-Level Design

**Me**: "Here's the architecture:

### Simple Approach:
```
Deduplicator
├── ConcurrentHashMap<Item, Timestamp> (seen items)
├── Process if new
└── Periodic cleanup of old entries
```

### For Scale (Billions):
```
Deduplicator
├── Bloom Filter (fast negative check)
│   └── "Probably seen" or "definitely not seen"
├── ConcurrentHashMap (precise check)
└── Time-based eviction
```

### Key Optimization:
Use `ConcurrentHashMap.newKeySet()` - perfect for deduplication!"

---

## Implementation: Basic Deduplicator

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.*;

public class Deduplicator<T> {
    
    // Seen items with timestamp
    static class SeenEntry {
        final long timestamp;
        
        SeenEntry() {
            this.timestamp = System.currentTimeMillis();
        }
        
        boolean isExpired(long ttlMs) {
            return System.currentTimeMillis() - timestamp > ttlMs;
        }
    }
    
    private final ConcurrentHashMap<T, SeenEntry> seenItems;
    private final long deduplicationWindowMs;
    private final Consumer<T> processor;
    
    // Cleanup
    private final ScheduledExecutorService cleaner;
    
    // Statistics
    private final AtomicLong processed = new AtomicLong(0);
    private final AtomicLong duplicates = new AtomicLong(0);
    private final AtomicLong expired = new AtomicLong(0);
    
    public Deduplicator(long deduplicationWindowMs, Consumer<T> processor) {
        this.seenItems = new ConcurrentHashMap<>();
        this.deduplicationWindowMs = deduplicationWindowMs;
        this.processor = processor;
        
        // Periodic cleanup
        this.cleaner = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "DedupCleaner");
            t.setDaemon(true);
            return t;
        });
        
        long cleanupInterval = Math.max(deduplicationWindowMs / 10, 1000);
        cleaner.scheduleAtFixedRate(
            this::cleanupExpired,
            cleanupInterval,
            cleanupInterval,
            TimeUnit.MILLISECONDS
        );
    }
    
    /**
     * Process item if not seen before
     */
    public boolean processIfNew(T item) {
        if (item == null) {
            throw new IllegalArgumentException("Item cannot be null");
        }
        
        // Try to add to seen set
        SeenEntry existing = seenItems.putIfAbsent(item, new SeenEntry());
        
        if (existing != null) {
            // Already seen
            if (existing.isExpired(deduplicationWindowMs)) {
                // Expired, allow reprocessing
                seenItems.put(item, new SeenEntry());  // Update timestamp
                System.out.println(Thread.currentThread().getName() + 
                    " - Item reprocessed (expired): " + item);
                processor.accept(item);
                processed.incrementAndGet();
                expired.incrementAndGet();
                return true;
            } else {
                // Duplicate within window
                System.out.println(Thread.currentThread().getName() + 
                    " - DUPLICATE: " + item);
                duplicates.incrementAndGet();
                return false;
            }
        }
        
        // New item, process it
        System.out.println(Thread.currentThread().getName() + 
            " - Processing NEW item: " + item);
        processor.accept(item);
        processed.incrementAndGet();
        return true;
    }
    
    /**
     * Check if item is duplicate without processing
     */
    public boolean isDuplicate(T item) {
        SeenEntry entry = seenItems.get(item);
        return entry != null && !entry.isExpired(deduplicationWindowMs);
    }
    
    private void cleanupExpired() {
        long now = System.currentTimeMillis();
        int cleaned = 0;
        
        Iterator<Map.Entry<T, SeenEntry>> iterator = seenItems.entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<T, SeenEntry> entry = iterator.next();
            
            if (now - entry.getValue().timestamp > deduplicationWindowMs) {
                iterator.remove();
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            System.out.println("Cleanup: removed " + cleaned + " expired entries");
        }
    }
    
    public void shutdown() {
        cleaner.shutdown();
    }
    
    public Map<String, Long> getStatistics() {
        Map<String, Long> stats = new HashMap<>();
        stats.put("seenItems", (long) seenItems.size());
        stats.put("processed", processed.get());
        stats.put("duplicates", duplicates.get());
        stats.put("expired", expired.get());
        
        long total = processed.get() + duplicates.get();
        stats.put("deduplicationRate", total == 0 ? 0 : (duplicates.get() * 100 / total));
        
        return stats;
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        Deduplicator<String> dedup = new Deduplicator<>(
            5000,  // 5 second window
            item -> System.out.println("  >>> PROCESSED: " + item)
        );
        
        System.out.println("=== Test 1: Basic Deduplication ===");
        dedup.processIfNew("item1");
        dedup.processIfNew("item2");
        dedup.processIfNew("item1");  // Duplicate
        dedup.processIfNew("item3");
        dedup.processIfNew("item2");  // Duplicate
        
        System.out.println("\n=== Test 2: Expiry ===");
        dedup.processIfNew("temp");
        Thread.sleep(6000);  // Wait for expiry
        dedup.processIfNew("temp");  // Should process again (expired)
        
        System.out.println("\n=== Test 3: Concurrent Submissions ===");
        ExecutorService executor = Executors.newFixedThreadPool(5);
        CountDownLatch latch = new CountDownLatch(5);
        
        for (int i = 0; i < 5; i++) {
            executor.submit(() -> {
                try {
                    for (int j = 0; j < 10; j++) {
                        dedup.processIfNew("shared-item-" + (j % 5));
                        Thread.sleep(10);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await();
        executor.shutdown();
        
        System.out.println("\n=== Statistics ===");
        dedup.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
        
        dedup.shutdown();
    }
}
```

---

## Advanced: Bloom Filter for Scale

**Me**: "For billions of items, use Bloom filter:

```java
import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;

public class ScalableDeduplicator<T> {
    
    // Bloom filter for fast negative check
    private final BloomFilter<T> bloomFilter;
    
    // Precise set for positive confirmation
    private final ConcurrentHashMap<T, SeenEntry> seenItems;
    
    // Configuration
    private final int maxSize;
    private final double falsePositiveRate;
    
    public ScalableDeduplicator(int expectedItems, double falsePositiveRate) {
        this.maxSize = expectedItems;
        this.falsePositiveRate = falsePositiveRate;
        
        // Create bloom filter
        this.bloomFilter = BloomFilter.create(
            Funnels.byteArrayFunnel(),
            expectedItems,
            falsePositiveRate
        );
        
        this.seenItems = new ConcurrentHashMap<>();
    }
    
    public boolean processIfNew(T item) {
        // Fast check: Bloom filter (lock-free read)
        if (!bloomFilter.mightContain(item)) {
            // Definitely not seen before
            bloomFilter.put(item);
            seenItems.put(item, new SeenEntry());
            processor.accept(item);
            return true;
        }
        
        // Might have seen it, check precise set
        SeenEntry existing = seenItems.putIfAbsent(item, new SeenEntry());
        
        if (existing != null) {
            // Confirmed duplicate
            return false;
        }
        
        // False positive from bloom filter, but actually new
        processor.accept(item);
        return true;
    }
}
```

**Bloom Filter Benefits**:
```
1 billion items:

Without Bloom Filter:
- Memory: 1B × (object overhead + timestamp) ≈ 20GB
- Check time: HashMap lookup ≈ 100ns

With Bloom Filter (1% false positive):
- Memory: ~1.2GB (bloom filter) + 20GB (hash map) = 21.2GB
- Check time: 
  - New item: Bloom check ≈ 50ns (skip HashMap!)
  - Duplicate: Bloom + HashMap ≈ 150ns

If 90% are duplicates:
- 90% fast path (bloom only): 50ns
- 10% slow path (bloom + hash): 150ns
- Average: 60ns vs 100ns
- 1.6× faster + early rejection
```

**Trade-off**:
- ✓ Fast rejection of new items
- ✓ Lower memory if most items are new
- ✗ False positives (need HashMap anyway)
- ✗ Additional complexity
"

---

# Multi-Level Cache

## Problem Introduction

**Interviewer**: "Design a multi-level cache system with L1 (thread-local), L2 (shared in-memory), and L3 (distributed/remote) caches."

**Me**: "Classic cache hierarchy! Let me clarify:

**Structure**:
```
Request → L1 (thread-local) → hit? Return
       → L2 (shared memory) → hit? Promote to L1, return
       → L3 (Redis/remote) → hit? Promote to L2, return
       → Database → Store in all levels, return

Example latencies:
L1: 10ns (local variable)
L2: 100ns (ConcurrentHashMap)
L3: 1ms (network)
DB: 10ms (disk + network)
```

**Questions**:
1. **Promotion**: Always promote or conditional?
2. **Invalidation**: How to sync across levels?
3. **Size limits**: Per-level limits?
4. **Consistency**: Strong or eventual?
5. **Write strategy**: Write-through or write-behind?

**Assumptions**:
- Always promote on hit
- Invalidate all levels on write
- Different sizes per level (L1: 10, L2: 100, L3: 1000)
- Eventual consistency
- Write-through for simplicity

Sound good?"

**Interviewer**: "Yes, focus on the promotion and invalidation mechanisms."

---

## High-Level Design

**Me**: "Here's the architecture:

### Cache Hierarchy:
```
┌─────────────────────────────────────────┐
│  L1: ThreadLocal<Map<K, V>>             │
│  Size: 10-100 per thread                │
│  Latency: ~10ns                         │
└─────────────────────────────────────────┘
            ↓ (miss)
┌─────────────────────────────────────────┐
│  L2: ConcurrentHashMap<K, V>            │
│  Size: 1000-10000 (shared)              │
│  Latency: ~100ns                        │
└─────────────────────────────────────────┘
            ↓ (miss)
┌─────────────────────────────────────────┐
│  L3: Remote Cache (Redis)               │
│  Size: 100000+ (distributed)            │
│  Latency: ~1ms                          │
└─────────────────────────────────────────┘
            ↓ (miss)
┌─────────────────────────────────────────┐
│  Database                               │
│  Latency: ~10-100ms                     │
└─────────────────────────────────────────┘
```

### Key Operations:
1. **Get**: Check L1 → L2 → L3 → DB, promote on hit
2. **Put**: Write to DB → Invalidate all levels
3. **Invalidate**: Clear from all levels"

---

## Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class MultiLevelCache<K, V> {
    
    // L1: Thread-local cache (fastest)
    private final ThreadLocal<Map<K, CacheEntry<V>>> l1Cache;
    private final int l1MaxSize;
    
    // L2: Shared in-memory cache
    private final ConcurrentHashMap<K, CacheEntry<V>> l2Cache;
    private final int l2MaxSize;
    
    // L3: Remote cache interface
    private final RemoteCache<K, V> l3Cache;
    
    // Database interface
    private final Database<K, V> database;
    
    // TTL
    private final long defaultTtlMs;
    
    // Statistics per level
    private final AtomicLong l1Hits = new AtomicLong(0);
    private final AtomicLong l2Hits = new AtomicLong(0);
    private final AtomicLong l3Hits = new AtomicLong(0);
    private final AtomicLong dbHits = new AtomicLong(0);
    
    static class CacheEntry<V> {
        final V value;
        final long expiryTime;
        
        CacheEntry(V value, long ttlMs) {
            this.value = value;
            this.expiryTime = System.currentTimeMillis() + ttlMs;
        }
        
        boolean isExpired() {
            return System.currentTimeMillis() > expiryTime;
        }
    }
    
    public interface RemoteCache<K, V> {
        V get(K key);
        void put(K key, V value, long ttlMs);
        void remove(K key);
    }
    
    public interface Database<K, V> {
        V read(K key);
        void write(K key, V value);
        void delete(K key);
    }
    
    public MultiLevelCache(int l1MaxSize, 
                          int l2MaxSize,
                          long defaultTtlMs,
                          RemoteCache<K, V> l3Cache,
                          Database<K, V> database) {
        this.l1MaxSize = l1MaxSize;
        this.l2MaxSize = l2MaxSize;
        this.defaultTtlMs = defaultTtlMs;
        this.l3Cache = l3Cache;
        this.database = database;
        
        // Initialize L1 cache (per thread)
        this.l1Cache = ThreadLocal.withInitial(() -> 
            new LinkedHashMap<K, CacheEntry<V>>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<K, CacheEntry<V>> eldest) {
                    return size() > l1MaxSize;
                }
            }
        );
        
        // Initialize L2 cache
        this.l2Cache = new ConcurrentHashMap<>();
        
        System.out.println("Multi-level cache initialized: L1=" + l1MaxSize + 
            ", L2=" + l2MaxSize);
    }
    
    /**
     * Get value from cache hierarchy
     */
    public V get(K key) {
        // L1: Thread-local (no synchronization!)
        Map<K, CacheEntry<V>> l1 = l1Cache.get();
        CacheEntry<V> entry = l1.get(key);
        
        if (entry != null && !entry.isExpired()) {
            l1Hits.incrementAndGet();
            System.out.println(Thread.currentThread().getName() + 
                " - L1 HIT: " + key);
            return entry.value;
        }
        
        // L1 miss, try L2
        entry = l2Cache.get(key);
        if (entry != null && !entry.isExpired()) {
            l2Hits.incrementAndGet();
            System.out.println(Thread.currentThread().getName() + 
                " - L2 HIT: " + key + " (promoting to L1)");
            
            // Promote to L1
            l1.put(key, entry);
            return entry.value;
        }
        
        // L2 miss, try L3
        V value = l3Cache.get(key);
        if (value != null) {
            l3Hits.incrementAndGet();
            System.out.println(Thread.currentThread().getName() + 
                " - L3 HIT: " + key + " (promoting to L2 and L1)");
            
            // Promote to L2 and L1
            CacheEntry<V> newEntry = new CacheEntry<>(value, defaultTtlMs);
            promoteToL2(key, newEntry);
            l1.put(key, newEntry);
            return value;
        }
        
        // All cache misses, load from database
        dbHits.incrementAndGet();
        System.out.println(Thread.currentThread().getName() + 
            " - DB HIT: " + key + " (loading and caching)");
        
        value = database.read(key);
        if (value != null) {
            // Store in all levels
            CacheEntry<V> newEntry = new CacheEntry<>(value, defaultTtlMs);
            l3Cache.put(key, value, defaultTtlMs);
            promoteToL2(key, newEntry);
            l1.put(key, newEntry);
        }
        
        return value;
    }
    
    /**
     * Put value (write-through)
     */
    public void put(K key, V value) {
        // Write to database first
        database.write(key, value);
        
        // Invalidate all cache levels
        invalidate(key);
        
        System.out.println("PUT: " + key + " (invalidated all levels)");
    }
    
    /**
     * Invalidate across all levels
     */
    public void invalidate(K key) {
        // L1: Remove from current thread's cache
        l1Cache.get().remove(key);
        
        // L2: Remove from shared cache
        l2Cache.remove(key);
        
        // L3: Remove from remote cache
        l3Cache.remove(key);
        
        System.out.println("INVALIDATE: " + key);
    }
    
    /**
     * Promote to L2 with size limit
     */
    private void promoteToL2(K key, CacheEntry<V> entry) {
        // Check size limit
        if (l2Cache.size() >= l2MaxSize) {
            evictFromL2();
        }
        
        l2Cache.put(key, entry);
    }
    
    /**
     * Evict from L2 (approximate LRU)
     */
    private void evictFromL2() {
        // Find oldest entry (approximate)
        K oldestKey = null;
        long oldestTime = Long.MAX_VALUE;
        
        int sampled = 0;
        for (Map.Entry<K, CacheEntry<V>> entry : l2Cache.entrySet()) {
            if (entry.getValue().expiryTime < oldestTime) {
                oldestTime = entry.getValue().expiryTime;
                oldestKey = entry.getKey();
            }
            
            if (++sampled > 10) break;  // Sample only 10 entries
        }
        
        if (oldestKey != null) {
            l2Cache.remove(oldestKey);
        }
    }
    
    public Map<String, Long> getStatistics() {
        Map<String, Long> stats = new HashMap<>();
        stats.put("l1Hits", l1Hits.get());
        stats.put("l2Hits", l2Hits.get());
        stats.put("l3Hits", l3Hits.get());
        stats.put("dbHits", dbHits.get());
        stats.put("l2Size", (long) l2Cache.size());
        
        long total = l1Hits.get() + l2Hits.get() + l3Hits.get() + dbHits.get();
        stats.put("totalRequests", total);
        
        if (total > 0) {
            stats.put("l1HitRate", l1Hits.get() * 100 / total);
            stats.put("l2HitRate", (l1Hits.get() + l2Hits.get()) * 100 / total);
            stats.put("overallHitRate", (total - dbHits.get()) * 100 / total);
        }
        
        return stats;
    }
    
    // Mock implementations for testing
    static class MockRemoteCache<K, V> implements RemoteCache<K, V> {
        private final ConcurrentHashMap<K, V> storage = new ConcurrentHashMap<>();
        
        @Override
        public V get(K key) {
            try {
                Thread.sleep(1);  // Simulate network latency
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return storage.get(key);
        }
        
        @Override
        public void put(K key, V value, long ttlMs) {
            storage.put(key, value);
        }
        
        @Override
        public void remove(K key) {
            storage.remove(key);
        }
    }
    
    static class MockDatabase<K, V> implements Database<K, V> {
        private final ConcurrentHashMap<K, V> storage = new ConcurrentHashMap<>();
        
        @Override
        public V read(K key) {
            try {
                Thread.sleep(10);  // Simulate slow DB
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return storage.get(key);
        }
        
        @Override
        public void write(K key, V value) {
            storage.put(key, value);
        }
        
        @Override
        public void delete(K key) {
            storage.remove(key);
        }
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        MockRemoteCache<String, String> l3 = new MockRemoteCache<>();
        MockDatabase<String, String> db = new MockDatabase<>();
        
        // Pre-populate database
        db.write("key1", "value1");
        db.write("key2", "value2");
        db.write("key3", "value3");
        
        MultiLevelCache<String, String> cache = new MultiLevelCache<>(
            5,      // L1 size
            20,     // L2 size
            60000,  // 1 minute TTL
            l3,
            db
        );
        
        System.out.println("=== Test Cache Hierarchy ===\n");
        
        // First access: DB hit
        System.out.println("First access:");
        String v1 = cache.get("key1");
        
        // Second access: L1 hit (same thread)
        System.out.println("\nSecond access (same thread):");
        String v2 = cache.get("key1");
        
        // Third access: Different thread (L2 hit)
        System.out.println("\nThird access (different thread):");
        new Thread(() -> {
            String v3 = cache.get("key1");
            System.out.println("Got: " + v3);
        }).start();
        
        Thread.sleep(100);
        
        // Write invalidates all levels
        System.out.println("\nWrite operation:");
        cache.put("key1", "newValue1");
        
        // Next read should be DB hit again
        System.out.println("\nAfter invalidation:");
        String v4 = cache.get("key1");
        
        Thread.sleep(100);
        
        System.out.println("\n=== Statistics ===");
        cache.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Key gotchas:

### Pitfall 1: Stale Reads After Invalidation

```java
// Problem:
Thread-1: cache.put("key", "v2");  // Invalidates all
Thread-2: (already read "v1" from L1) → uses stale value

// Solution: Version numbers
static class VersionedEntry<V> {
    final V value;
    final long version;
    
    VersionedEntry(V value, long version) {
        this.value = value;
        this.version = version;
    }
}

private final AtomicLong globalVersion = new AtomicLong(0);

public V get(K key) {
    long currentVersion = globalVersion.get();
    
    // Check L1
    VersionedEntry<V> entry = l1.get(key);
    if (entry != null && entry.version == currentVersion) {
        return entry.value;  // Version matches, not stale
    }
    
    // Load from lower levels...
}

public void put(K key, V value) {
    globalVersion.incrementAndGet();  // Bump version
    // All cached entries are now stale
}
```

### Pitfall 2: Promotion Storm

```java
// Problem: Hot key accessed by 100 threads
// All threads miss L1, hit L2
// All threads try to promote to L1
// Wasteful!

// Solution: Lazy promotion
public V get(K key) {
    // Check L1
    V value = l1.get(key);
    if (value != null) return value;
    
    // Check L2
    value = l2.get(key);
    if (value != null) {
        // Only promote occasionally (10% of time)
        if (ThreadLocalRandom.current().nextInt(10) == 0) {
            l1.put(key, value);
        }
        return value;
    }
    // ...
}
```

### Pitfall 3: L1 Invalidation Across Threads

```java
// Problem: Write on Thread-1 doesn't invalidate L1 on Thread-2!
Thread-1: cache.put("key", "v2");
          Invalidates own L1
Thread-2: L1 still has "v1" (stale!)

// Solution 1: Don't use L1 for mutable data
// Solution 2: Versioning (shown above)
// Solution 3: L1 only for immutable/read-only data
```

### Pitfall 4: Memory Overhead

```java
// Problem: 100 threads × 100 L1 entries = 10,000 cached items
// Plus L2: 1000 items
// Plus L3: 10,000 items
// Same key cached 102 times!

// Solution: Smart sizing
// L1: 10-50 items (hot data only)
// L2: 1000-10000 items
// L3: 100000+ items
```

### Pitfall 5: Thundering Herd on Cache Miss

```java
// Problem: Key not in any cache, 100 threads request it
// All 100 threads query database simultaneously!

// Solution: Request coalescing
private final ConcurrentHashMap<K, CompletableFuture<V>> loadingKeys = 
    new ConcurrentHashMap<>();

public V get(K key) {
    // Check caches...
    
    // Cache miss, load from DB
    CompletableFuture<V> future = loadingKeys.computeIfAbsent(key, k -> 
        CompletableFuture.supplyAsync(() -> {
            V value = database.read(k);
            // Cache it...
            return value;
        }).whenComplete((v, ex) -> loadingKeys.remove(k))
    );
    
    return future.join();  // Wait for result (only one thread loads!)
}
```

### Pitfall 6: Write-Through Performance

```java
// Problem: Writes are slow (must write to DB)
cache.put("key", "value");  // Waits for DB write (10ms)

// Solution: Write-behind for L1/L2, write-through for L3
public void put(K key, V value) {
    // Update L1 immediately
    l1Cache.get().put(key, new CacheEntry<>(value, defaultTtlMs));
    
    // Update L2 immediately
    l2Cache.put(key, new CacheEntry<>(value, defaultTtlMs));
    
    // Async write to L3 and DB
    CompletableFuture.runAsync(() -> {
        l3Cache.put(key, value, defaultTtlMs);
        database.write(key, value);
    });
    
    System.out.println("PUT: " + key + " (async write to L3/DB)");
}
```
"

---

## Interview Follow-Up Questions

**Q1: How to implement cache warming?**

```java
public void warmUp(List<K> hotKeys) {
    System.out.println("Warming up cache with " + hotKeys.size() + " keys");
    
    // Load in parallel
    ExecutorService loader = Executors.newFixedThreadPool(10);
    CountDownLatch latch = new CountDownLatch(hotKeys.size());
    
    for (K key : hotKeys) {
        loader.submit(() -> {
            try {
                V value = database.read(key);
                if (value != null) {
                    CacheEntry<V> entry = new CacheEntry<>(value, defaultTtlMs);
                    l2Cache.put(key, entry);
                    l3Cache.put(key, value, defaultTtlMs);
                }
            } finally {
                latch.countDown();
            }
        });
    }
    
    latch.await();
    loader.shutdown();
    
    System.out.println("Cache warming complete");
}
```

**Q2: How to handle cache coherence in distributed system?**

```java
// Problem: Multiple app servers, each has L1/L2
// Server 1 writes key=v2
// Server 2's L1/L2 still have key=v1 (stale!)

// Solution 1: Pub/Sub for invalidation
public class DistributedCache<K, V> {
    private final RedisPubSub invalidationChannel;
    
    public void put(K key, V value) {
        database.write(key, value);
        
        // Broadcast invalidation
        invalidationChannel.publish("invalidate:" + key);
    }
    
    // Listen for invalidations
    private void setupInvalidationListener() {
        invalidationChannel.subscribe(message -> {
            if (message.startsWith("invalidate:")) {
                String key = message.substring(11);
                l1Cache.get().remove(key);
                l2Cache.remove(key);
            }
        });
    }
}

// Solution 2: TTL-based (eventual consistency)
// Set short TTL (1-5 seconds)
// Accept brief staleness
```

**Q3: How to test multi-level cache?**

```java
@Test
public void testPromotion() {
    // Put in L3 only
    l3Cache.put("key", "value");
    
    // First get: L3 hit, promotes to L2
    assertEquals("value", cache.get("key"));
    assertEquals(1, l3Hits.get());
    assertEquals(0, l2Hits.get());
    
    // Second get: L2 hit
    assertEquals("value", cache.get("key"));
    assertEquals(1, l2Hits.get());
}

@Test
public void testInvalidation() {
    cache.put("key", "v1");
    assertEquals("v1", cache.get("key"));  // Cache it
    
    cache.put("key", "v2");  // Invalidates
    
    // Should reload from DB
    assertEquals("v2", cache.get("key"));
    assertEquals(2, dbHits.get());  // Both loads from DB
}
```

---

# Thread Pool with Dynamic Sizing

## Problem Introduction

**Interviewer**: "Design a thread pool that dynamically adjusts the number of worker threads based on workload. Scale up when tasks are queued, scale down when idle."

**Me**: "Great problem! Let me clarify:

**Concept**:
```
Low load:  [W1] [W2]           (2 threads, no queue)
Medium:    [W1] [W2] [W3] [W4] (4 threads, small queue)
High load: [W1] [W2] ... [W10] (10 threads, queue growing)

Auto-scaling:
- Queue depth > threshold → Add workers
- Workers idle > timeout → Remove workers
- Min/max bounds
```

**Questions**:
1. **Scaling triggers**: Queue depth, CPU usage, or latency?
2. **Scale up/down rate**: How fast to add/remove threads?
3. **Bounds**: Min and max thread count?
4. **Idle timeout**: How long before removing idle thread?
5. **Task queue**: Bounded or unbounded?

**Assumptions**:
- Queue depth based scaling
- Scale up fast, scale down slow
- Min: 2, Max: 20
- Idle timeout: 60 seconds
- Bounded queue with backpressure

Sound good?"

**Interviewer**: "Yes, focus on the dynamic scaling mechanism."

---

## High-Level Design

**Me**: "Here's the architecture:

### Components:
```
DynamicThreadPool
├── Task Queue (BlockingQueue)
├── Worker Threads (List)
├── Monitor Thread (checks metrics)
└── Scale up/down logic
```

### Scaling Logic:
```
Scale Up Triggers:
- Queue size > threshold
- All workers busy
- Task rejection rate high

Scale Down Triggers:
- Workers idle > timeout
- Queue empty
- CPU utilization low
```

### Key Challenge:
How to safely add/remove threads while they're processing?"

---

## Implementation

```java
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;

public class DynamicThreadPool {
    
    // Configuration
    private final int minThreads;
    private final int maxThreads;
    private final long idleTimeoutMs;
    private final int queueCapacity;
    
    // Task queue
    private final BlockingQueue<Runnable> taskQueue;
    
    // Worker management
    private final List<Worker> workers;
    private final Lock workerLock;
    private final AtomicInteger activeWorkers;
    
    // State
    private volatile boolean shutdown = false;
    
    // Monitoring
    private final ScheduledExecutorService monitor;
    
    // Statistics
    private final AtomicLong tasksSubmitted = new AtomicLong(0);
    private final AtomicLong tasksCompleted = new AtomicLong(0);
    private final AtomicLong tasksRejected = new AtomicLong(0);
    private final AtomicInteger peakThreads = new AtomicInteger(0);
    
    public DynamicThreadPool(int minThreads, int maxThreads, 
                            long idleTimeoutMs, int queueCapacity) {
        this.minThreads = minThreads;
        this.maxThreads = maxThreads;
        this.idleTimeoutMs = idleTimeoutMs;
        this.queueCapacity = queueCapacity;
        
        this.taskQueue = new ArrayBlockingQueue<>(queueCapacity);
        this.workers = new CopyOnWriteArrayList<>();
        this.workerLock = new ReentrantLock();
        this.activeWorkers = new AtomicInteger(0);
        
        // Start minimum threads
        for (int i = 0; i < minThreads; i++) {
            addWorker();
        }
        
        // Start monitor thread
        this.monitor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "PoolMonitor");
            t.setDaemon(true);
            return t;
        });
        
        monitor.scheduleAtFixedRate(
            this::adjustPoolSize,
            1000,
            1000,
            TimeUnit.MILLISECONDS
        );
        
        System.out.println("Dynamic thread pool started: min=" + minThreads + 
            ", max=" + maxThreads);
    }
    
    /**
     * Submit task
     */
    public void submit(Runnable task) {
        if (shutdown) {
            throw new RejectedExecutionException("Pool is shut down");
        }
        
        tasksSubmitted.incrementAndGet();
        
        if (!taskQueue.offer(task)) {
            // Queue full, try to scale up
            if (scaleUp()) {
                // Successfully added worker, try again
                if (!taskQueue.offer(task)) {
                    tasksRejected.incrementAndGet();
                    throw new RejectedExecutionException("Queue full");
                }
            } else {
                tasksRejected.incrementAndGet();
                throw new RejectedExecutionException("Queue full, max threads reached");
            }
        }
        
        System.out.println("Task submitted (queue size: " + taskQueue.size() + 
            ", workers: " + workers.size() + ")");
    }
    
    /**
     * Worker thread
     */
    class Worker implements Runnable {
        private final int id;
        private final AtomicLong lastActiveTime;
        private volatile boolean stopped = false;
        
        Worker(int id) {
            this.id = id;
            this.lastActiveTime = new AtomicLong(System.currentTimeMillis());
        }
        
        @Override
        public void run() {
            System.out.println("Worker-" + id + " started");
            
            while (!stopped && !shutdown) {
                try {
                    // Poll with timeout
                    Runnable task = taskQueue.poll(1, TimeUnit.SECONDS);
                    
                    if (task != null) {
                        // Got task, process it
                        activeWorkers.incrementAndGet();
                        lastActiveTime.set(System.currentTimeMillis());
                        
                        try {
                            System.out.println("Worker-" + id + " processing task");
                            task.run();
                            tasksCompleted.incrementAndGet();
                        } catch (Exception e) {
                            System.err.println("Worker-" + id + " task error: " + e);
                        } finally {
                            activeWorkers.decrementAndGet();
                        }
                    } else {
                        // Timeout, check if should stop
                        if (shouldStop()) {
                            stopped = true;
                        }
                    }
                    
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            
            System.out.println("Worker-" + id + " stopped");
        }
        
        boolean shouldStop() {
            // Don't stop if below minimum
            if (workers.size() <= minThreads) {
                return false;
            }
            
            // Check idle time
            long idleTime = System.currentTimeMillis() - lastActiveTime.get();
            return idleTime > idleTimeoutMs;
        }
        
        void stopWorker() {
            stopped = true;
        }
    }
    
    /**
     * Add worker thread
     */
    private boolean addWorker() {
        workerLock.lock();
        try {
            if (workers.size() >= maxThreads) {
                return false;
            }
            
            Worker worker = new Worker(workers.size());
            workers.add(worker);
            new Thread(worker, "Worker-" + worker.id).start();
            
            int size = workers.size();
            if (size > peakThreads.get()) {
                peakThreads.set(size);
            }
            
            System.out.println("Added worker (total: " + size + ")");
            return true;
            
        } finally {
            workerLock.unlock();
        }
    }
    
    /**
     * Remove idle workers
     */
    private void removeIdleWorkers() {
        workerLock.lock();
        try {
            if (workers.size() <= minThreads) {
                return;
            }
            
            Iterator<Worker> iterator = workers.iterator();
            while (iterator.hasNext()) {
                Worker worker = iterator.next();
                
                if (worker.shouldStop()) {
                    worker.stopWorker();
                    iterator.remove();
                    System.out.println("Removed idle worker (total: " + workers.size() + ")");
                    
                    if (workers.size() <= minThreads) {
                        break;
                    }
                }
            }
        } finally {
            workerLock.unlock();
        }
    }
    
    /**
     * Scale up if needed
     */
    private boolean scaleUp() {
        int queueSize = taskQueue.size();
        int workerCount = workers.size();
        
        // Scale up if queue is growing and not at max
        if (queueSize > queueCapacity / 2 && workerCount < maxThreads) {
            return addWorker();
        }
        
        return false;
    }
    
    /**
     * Periodic adjustment
     */
    private void adjustPoolSize() {
        int queueSize = taskQueue.size();
        int workerCount = workers.size();
        int active = activeWorkers.get();
        
        System.out.println("\n[Monitor] Queue: " + queueSize + 
            ", Workers: " + workerCount + ", Active: " + active);
        
        // Scale up conditions
        if (queueSize > 10 && workerCount < maxThreads) {
            System.out.println("[Monitor] Queue building up, scaling up");
            addWorker();
        }
        else if (queueSize > 50 && workerCount < maxThreads) {
            System.out.println("[Monitor] Queue critical, adding multiple workers");
            addWorker();
            addWorker();
        }
        // Scale down conditions
        else if (queueSize == 0 && active == 0 && workerCount > minThreads) {
            System.out.println("[Monitor] All idle, checking for removal");
            removeIdleWorkers();
        }
    }
    
    public void shutdown() throws InterruptedException {
        shutdown = true;
        monitor.shutdown();
        
        // Stop all workers
        for (Worker worker : workers) {
            worker.stopWorker();
        }
        
        System.out.println("Thread pool shut down");
    }
    
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("currentThreads", workers.size());
        stats.put("activeThreads", activeWorkers.get());
        stats.put("peakThreads", peakThreads.get());
        stats.put("queueSize", taskQueue.size());
        stats.put("tasksSubmitted", tasksSubmitted.get());
        stats.put("tasksCompleted", tasksCompleted.get());
        stats.put("tasksRejected", tasksRejected.get());
        return stats;
    }
    
    // Test
    public static void main(String[] args) throws InterruptedException {
        DynamicThreadPool pool = new DynamicThreadPool(
            2,      // Min threads
            10,     // Max threads
            5000,   // 5 second idle timeout
            100     // Queue capacity
        );
        
        System.out.println("\n=== Phase 1: Low Load ===");
        for (int i = 0; i < 5; i++) {
            pool.submit(() -> {
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            Thread.sleep(100);
        }
        
        Thread.sleep(2000);
        
        System.out.println("\n=== Phase 2: High Load ===");
        for (int i = 0; i < 50; i++) {
            pool.submit(() -> {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
            Thread.sleep(10);
        }
        
        Thread.sleep(5000);
        
        System.out.println("\n=== Phase 3: Cool Down ===");
        Thread.sleep(10000);  // Let workers become idle
        
        System.out.println("\n=== Final Statistics ===");
        pool.getStatistics().forEach((key, value) -> 
            System.out.println(key + ": " + value));
        
        pool.shutdown();
    }
}
```

---

## Pitfalls and Edge Cases

**Me**: "Key gotchas:

### Pitfall 1: Scaling Too Fast

```java
// ❌ WRONG - Add thread for every queued task
if (queueSize > 0) {
    addWorker();  // Thrashing!
}

// ✓ CORRECT - Gradual scaling with thresholds
if (queueSize > 10 && workerCount < maxThreads) {
    addWorker();
}
```

### Pitfall 2: Removing Active Workers

```java
// ❌ WRONG - Remove any worker
workers.remove(0);  // Might be processing!

// ✓ CORRECT - Only remove idle workers
for (Worker w : workers) {
    if (w.isIdle() && w.idleTime() > threshold) {
        w.stopGracefully();
        workers.remove(w);
    }
}
```

### Pitfall 3: Race Condition in Worker Count

```java
// ❌ WRONG - Check-then-act race
if (workers.size() < maxThreads) {
    // Another thread might add worker here!
    addWorker();  // Might exceed max!
}

// ✓ CORRECT - Atomic check-and-add
workerLock.lock();
try {
    if (workers.size() < maxThreads) {
        addWorker();
    }
} finally {
    workerLock.unlock();
}
```

### Pitfall 4: Thread Leak on Exception

```java
// ❌ WRONG - Thread dies on exception
public void run() {
    while (!stopped) {
        Runnable task = queue.take();
        task.run();  // Exception kills thread!
    }
}

// ✓ CORRECT - Catch exceptions
public void run() {
    while (!stopped) {
        try {
            Runnable task = queue.take();
            task.run();
        } catch (Exception e) {
            System.err.println("Task error: " + e);
            // Thread continues!
        }
    }
}
```

### Pitfall 5: Shutdown Race

```java
// Problem: Shutdown while adding workers
Thread-1: adjustPoolSize() → addWorker()
Thread-2: shutdown() → stop all workers
Thread-1: New worker added after shutdown!

// Solution: Check shutdown flag
private boolean addWorker() {
    if (shutdown) {
        return false;
    }
    
    workerLock.lock();
    try {
        if (shutdown || workers.size() >= maxThreads) {
            return false;
        }
        // Add worker...
    } finally {
        workerLock.unlock();
    }
}
```

### Pitfall 6: Oscillation (Thrashing)

```java
// Problem:
Load increases → Add workers
Workers start → Process queue → Queue empty
Monitor sees empty queue → Remove workers
Load increases again → Add workers
Repeat! (thrashing)

// Solution: Hysteresis (different thresholds for up/down)
private void adjustPoolSize() {
    int queueSize = taskQueue.size();
    int workerCount = workers.size();
    
    // Scale up: queue > 20
    if (queueSize > 20 && workerCount < maxThreads) {
        addWorker();
    }
    // Scale down: queue < 5 (lower threshold!)
    else if (queueSize < 5 && workerCount > minThreads) {
        removeIdleWorkers();
    }
}
```
"

---

## Lock-Free Optimization

**Me**: "Can we optimize with lock-free?

### Current Bottleneck: Worker List Management
