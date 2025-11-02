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

---
