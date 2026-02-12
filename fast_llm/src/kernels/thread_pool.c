/*
 * Thread Pool Implementation
 * Windows threads with work queue
 */

#include "thread_pool.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#define MAX_WORKERS 64
#define TASK_QUEUE_SIZE 1024

/* Task */
typedef struct {
    task_func_t func;
    void* arg;
} task_t;

/* Thread pool state */
struct thread_pool {
    int num_workers;
    
#ifdef _WIN32
    HANDLE threads[MAX_WORKERS];
    CRITICAL_SECTION queue_lock;
    CONDITION_VARIABLE queue_not_empty;
    CONDITION_VARIABLE queue_empty;
#else
    pthread_t threads[MAX_WORKERS];
    pthread_mutex_t queue_lock;
    pthread_cond_t queue_not_empty;
    pthread_cond_t queue_empty;
#endif
    
    /* Task queue */
    task_t queue[TASK_QUEUE_SIZE];
    volatile int queue_head;
    volatile int queue_tail;
    volatile int queue_count;
    
    /* Synchronization */
    volatile bool shutdown;
    volatile int active_tasks;
};

/* Worker thread */
#ifdef _WIN32
static DWORD WINAPI worker_thread(LPVOID arg) {
#else
static void* worker_thread(void* arg) {
#endif
    thread_pool_t* pool = (thread_pool_t*)arg;
    int worker_id = 0;
    
    /* Find worker ID */
    for (int i = 0; i < pool->num_workers; i++) {
#ifdef _WIN32
        if (pool->threads[i] == GetCurrentThread()) {
            worker_id = i;
            break;
        }
#else
        if (pthread_equal(pool->threads[i], pthread_self())) {
            worker_id = i;
            break;
        }
#endif
    }
    
    while (true) {
        task_t task = {NULL, NULL};
        bool has_task = false;
        
        /* Lock and get task */
#ifdef _WIN32
        EnterCriticalSection(&pool->queue_lock);
#else
        pthread_mutex_lock(&pool->queue_lock);
#endif
        
        /* Wait for task or shutdown */
        while (pool->queue_count == 0 && !pool->shutdown) {
#ifdef _WIN32
            SleepConditionVariableCS(&pool->queue_not_empty, &pool->queue_lock, INFINITE);
#else
            pthread_cond_wait(&pool->queue_not_empty, &pool->queue_lock);
#endif
        }
        
        if (pool->shutdown && pool->queue_count == 0) {
#ifdef _WIN32
            LeaveCriticalSection(&pool->queue_lock);
#else
            pthread_mutex_unlock(&pool->queue_lock);
#endif
            break;
        }
        
        /* Get task */
        if (pool->queue_count > 0) {
            task = pool->queue[pool->queue_head];
            pool->queue_head = (pool->queue_head + 1) % TASK_QUEUE_SIZE;
            pool->queue_count--;
            pool->active_tasks++;
            has_task = true;
            
            /* Signal that queue has space */
#ifdef _WIN32
            WakeConditionVariable(&pool->queue_empty);
#else
            pthread_cond_signal(&pool->queue_empty);
#endif
        }
        
#ifdef _WIN32
        LeaveCriticalSection(&pool->queue_lock);
#else
        pthread_mutex_unlock(&pool->queue_lock);
#endif
        
        /* Execute task */
        if (has_task && task.func) {
            task.func(task.arg, worker_id);
            
            /* Mark task complete */
#ifdef _WIN32
            EnterCriticalSection(&pool->queue_lock);
#else
            pthread_mutex_lock(&pool->queue_lock);
#endif
            pool->active_tasks--;
            if (pool->active_tasks == 0 && pool->queue_count == 0) {
#ifdef _WIN32
                WakeAllConditionVariable(&pool->queue_empty);
#else
                pthread_cond_broadcast(&pool->queue_empty);
#endif
            }
#ifdef _WIN32
            LeaveCriticalSection(&pool->queue_lock);
#else
            pthread_mutex_unlock(&pool->queue_lock);
#endif
        }
    }
    
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

thread_pool_t* thread_pool_create(int num_workers) {
    if (num_workers < 1) num_workers = 1;
    if (num_workers > MAX_WORKERS) num_workers = MAX_WORKERS;
    
    thread_pool_t* pool = calloc(1, sizeof(thread_pool_t));
    pool->num_workers = num_workers;
    pool->shutdown = false;
    
    /* Initialize synchronization */
#ifdef _WIN32
    InitializeCriticalSection(&pool->queue_lock);
    InitializeConditionVariable(&pool->queue_not_empty);
    InitializeConditionVariable(&pool->queue_empty);
#else
    pthread_mutex_init(&pool->queue_lock, NULL);
    pthread_cond_init(&pool->queue_not_empty, NULL);
    pthread_cond_init(&pool->queue_empty, NULL);
#endif
    
    /* Create worker threads */
    for (int i = 0; i < num_workers; i++) {
#ifdef _WIN32
        pool->threads[i] = CreateThread(NULL, 0, worker_thread, pool, 0, NULL);
#else
        pthread_create(&pool->threads[i], NULL, worker_thread, pool);
#endif
    }
    
    return pool;
}

void thread_pool_destroy(thread_pool_t* pool) {
    if (!pool) return;
    
    /* Signal shutdown */
#ifdef _WIN32
    EnterCriticalSection(&pool->queue_lock);
#else
    pthread_mutex_lock(&pool->queue_lock);
#endif
    pool->shutdown = true;
#ifdef _WIN32
    WakeAllConditionVariable(&pool->queue_not_empty);
    LeaveCriticalSection(&pool->queue_lock);
#else
    pthread_cond_broadcast(&pool->queue_not_empty);
    pthread_mutex_unlock(&pool->queue_lock);
#endif
    
    /* Wait for threads */
    for (int i = 0; i < pool->num_workers; i++) {
#ifdef _WIN32
        WaitForSingleObject(pool->threads[i], INFINITE);
        CloseHandle(pool->threads[i]);
#else
        pthread_join(pool->threads[i], NULL);
#endif
    }
    
    /* Cleanup */
#ifdef _WIN32
    DeleteCriticalSection(&pool->queue_lock);
#else
    pthread_mutex_destroy(&pool->queue_lock);
    pthread_cond_destroy(&pool->queue_not_empty);
    pthread_cond_destroy(&pool->queue_empty);
#endif
    
    free(pool);
}

void thread_pool_submit_batch(thread_pool_t* pool,
                               task_func_t func,
                               void** args,
                               int num_tasks) {
    if (!pool || !func || num_tasks <= 0) return;
    
    for (int i = 0; i < num_tasks; i++) {
        /* Wait for queue space */
#ifdef _WIN32
        EnterCriticalSection(&pool->queue_lock);
        while (pool->queue_count >= TASK_QUEUE_SIZE) {
            SleepConditionVariableCS(&pool->queue_empty, &pool->queue_lock, INFINITE);
        }
#else
        pthread_mutex_lock(&pool->queue_lock);
        while (pool->queue_count >= TASK_QUEUE_SIZE) {
            pthread_cond_wait(&pool->queue_empty, &pool->queue_lock);
        }
#endif
        
        /* Add task */
        pool->queue[pool->queue_tail].func = func;
        pool->queue[pool->queue_tail].arg = args[i];
        pool->queue_tail = (pool->queue_tail + 1) % TASK_QUEUE_SIZE;
        pool->queue_count++;
        
        /* Signal workers */
#ifdef _WIN32
        WakeConditionVariable(&pool->queue_not_empty);
        LeaveCriticalSection(&pool->queue_lock);
#else
        pthread_cond_signal(&pool->queue_not_empty);
        pthread_mutex_unlock(&pool->queue_lock);
#endif
    }
}

void thread_pool_wait(thread_pool_t* pool) {
    if (!pool) return;
    
#ifdef _WIN32
    EnterCriticalSection(&pool->queue_lock);
    while (pool->queue_count > 0 || pool->active_tasks > 0) {
        SleepConditionVariableCS(&pool->queue_empty, &pool->queue_lock, INFINITE);
    }
    LeaveCriticalSection(&pool->queue_lock);
#else
    pthread_mutex_lock(&pool->queue_lock);
    while (pool->queue_count > 0 || pool->active_tasks > 0) {
        pthread_cond_wait(&pool->queue_empty, &pool->queue_lock);
    }
    pthread_mutex_unlock(&pool->queue_lock);
#endif
}

int thread_pool_get_num_workers(thread_pool_t* pool) {
    return pool ? pool->num_workers : 0;
}
