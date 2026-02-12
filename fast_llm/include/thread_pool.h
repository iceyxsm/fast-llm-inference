/*
 * Thread Pool for LLM Inference
 * Persistent worker threads to eliminate OpenMP overhead
 */

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Task function type */
typedef void (*task_func_t)(void* arg, int worker_id);

/* Thread pool */
typedef struct thread_pool thread_pool_t;

/* Create thread pool with specified number of workers */
thread_pool_t* thread_pool_create(int num_workers);

/* Destroy thread pool */
void thread_pool_destroy(thread_pool_t* pool);

/* Submit a batch of tasks */
void thread_pool_submit_batch(thread_pool_t* pool, 
                               task_func_t func,
                               void** args,
                               int num_tasks);

/* Wait for all tasks to complete */
void thread_pool_wait(thread_pool_t* pool);

/* Get number of workers */
int thread_pool_get_num_workers(thread_pool_t* pool);

#ifdef __cplusplus
}
#endif

#endif /* THREAD_POOL_H */
