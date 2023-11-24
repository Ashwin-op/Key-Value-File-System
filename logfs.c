/**
 * Tony Givargis
 * Copyright (C), 2023
 * University of California, Irvine
 *
 * CS 238P - Operating Systems
 * logfs.c
 */

#include <pthread.h>
#include "device.h"
#include "logfs.h"

#define WCACHE_BLOCKS 32
#define RCACHE_BLOCKS 256

/**
 * Needs:
 *   pthread_create()
 *   pthread_join()
 *   pthread_mutex_init()
 *   pthread_mutex_destroy()
 *   pthread_mutex_lock()
 *   pthread_mutex_unlock()
 *   pthread_cond_init()
 *   pthread_cond_destroy()
 *   pthread_cond_wait()
 *   pthread_cond_signal()
 */

/* research the above Needed API and design accordingly */

struct queue {
    char *data; // contiguous memory to store the data
    uint64_t head, tail; // head and tail of the queue
    uint64_t capacity; // capacity of the queue
    uint64_t utilized; // utilized space in the queue
};

struct worker {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int stop_thread;
    int force_write;
};

struct cache_block {
    char *data;
    uint64_t offset; // offset of the block in the device (multiples of block size)
    short valid;
    uint64_t idx;
};

struct logfs {
    struct device *device;
    uint64_t utilized;
    uint64_t capacity;
    struct queue *write_queue;
    struct worker *worker;
    struct cache_block read_cache[RCACHE_BLOCKS];
};

struct logfs_metadata {
    uint64_t utilized;
};

void mark_cache_invalid(struct logfs *logfs, uint64_t offset) {
    int i;
    for (i = 0; i < RCACHE_BLOCKS; i++) {
        if (logfs->read_cache[i].valid && logfs->read_cache[i].offset == offset) {
            logfs->read_cache[i].valid = 0;
        }
    }
}

void write_to_disk(struct logfs *logfs) {
    uint64_t block_size = device_block(logfs->device);

    // if tail is greater than head
    if (logfs->write_queue->tail > logfs->write_queue->head) {
        // check how many blocks needed to be read from the device
        uint64_t size_to_read = logfs->write_queue->tail - logfs->write_queue->head;
        // see if it's overlapping between two blocks in the device
        uint64_t blocks_to_read = size_to_read / block_size;
        if ((logfs->utilized + size_to_read) / block_size > logfs->utilized / block_size) {
            blocks_to_read++;
        }
        if (size_to_read % block_size) {
            blocks_to_read++;
        }
        // get the starting block offset
        uint64_t start_block_offset = (logfs->utilized / block_size * block_size) + device_block(logfs->device);
        // read the blocks from the device
        char *blocks = malloc(blocks_to_read * block_size);
        if (!blocks) {
            TRACE("out of memory");
            return;
        }
        if (device_read(logfs->device, blocks, start_block_offset, blocks_to_read * block_size)) {
            TRACE("device_read()");
            free(blocks);
            return;
        }
        // copy the data from the queue to the blocks
        memcpy(logfs->utilized % block_size + blocks, logfs->write_queue->data + logfs->write_queue->head,
               logfs->write_queue->tail - logfs->write_queue->head);
        // write the blocks to the device
        if (device_write(logfs->device, blocks, start_block_offset, blocks_to_read * block_size)) {
            TRACE("device_write()");
            free(blocks);
            return;
        }
        // update the utilized
        logfs->utilized += (logfs->write_queue->tail - logfs->write_queue->head);
        // update the utilized
        logfs->write_queue->utilized -= (logfs->write_queue->tail - logfs->write_queue->head);
        // update the head
        logfs->write_queue->head = logfs->write_queue->tail;
        free(blocks);

        mark_cache_invalid(logfs, start_block_offset);
    } else if (logfs->write_queue->tail < logfs->write_queue->head) {
        // write from head to the end of the queue
        // then write from the beginning of the queue to tail

        // check how many blocks needed to be read from the device
        uint64_t size_to_read = logfs->write_queue->capacity - logfs->write_queue->head;
        // see if it's overlapping between two blocks in the device
        uint64_t blocks_to_read = size_to_read / block_size;
        if ((logfs->utilized + size_to_read) / block_size > logfs->utilized / block_size) {
            blocks_to_read++;
        }
        if (size_to_read % block_size) {
            blocks_to_read++;
        }
        // get the starting block offset
        uint64_t start_block_offset = (logfs->utilized / block_size * block_size) + device_block(logfs->device);
        // read the blocks from the device
        char *blocks = malloc(blocks_to_read * block_size);
        if (!blocks) {
            TRACE("out of memory");
            return;
        }
        if (device_read(logfs->device, blocks, start_block_offset, blocks_to_read * block_size)) {
            TRACE("device_read()");
            free(blocks);
            return;
        }
        // copy the data from the queue to the blocks
        memcpy(logfs->utilized % block_size + blocks, logfs->write_queue->data + logfs->write_queue->head,
               logfs->write_queue->capacity - logfs->write_queue->head);
        // write the blocks to the device
        if (device_write(logfs->device, blocks, start_block_offset, blocks_to_read * block_size)) {
            TRACE("device_write()");
            free(blocks);
            return;
        }
        // update the utilized
        logfs->utilized += (logfs->write_queue->capacity - logfs->write_queue->head);
        // update the utilized
        logfs->write_queue->utilized -= (logfs->write_queue->capacity - logfs->write_queue->head);
        // update the head
        logfs->write_queue->head = 0;
        free(blocks);

        mark_cache_invalid(logfs, start_block_offset);

        // check how many blocks needed to be read from the device
        size_to_read = logfs->write_queue->tail;
        // see if it's overlapping between two blocks in the device
        blocks_to_read = size_to_read / block_size;
        if ((logfs->utilized + size_to_read) / block_size > logfs->utilized / block_size) {
            blocks_to_read++;
        }
        if (size_to_read % block_size) {
            blocks_to_read++;
        }
        // get the starting block offset
        start_block_offset = (logfs->utilized / block_size * block_size) + device_block(logfs->device);
        // read the blocks from the device
        blocks = malloc(blocks_to_read * block_size);
        if (!blocks) {
            TRACE("out of memory");
            return;
        }
        if (device_read(logfs->device, blocks, start_block_offset, blocks_to_read * block_size)) {
            TRACE("device_read()");
            free(blocks);
            return;
        }
        // copy the data from the queue to the blocks
        memcpy(logfs->utilized % block_size + blocks, logfs->write_queue->data,
               logfs->write_queue->tail);
        // write the blocks to the device
        if (device_write(logfs->device, blocks, start_block_offset, blocks_to_read * block_size)) {
            TRACE("device_write()");
            free(blocks);
            return;
        }
        // update the utilized
        logfs->utilized += (logfs->write_queue->tail);
        // update the utilized
        logfs->write_queue->utilized -= (logfs->write_queue->tail);
        // update the head
        logfs->write_queue->head = logfs->write_queue->tail;
        free(blocks);

        mark_cache_invalid(logfs, start_block_offset);
    }
}

void *worker(void *arg) {
    struct logfs *logfs = arg;

    while (1) {
        if (pthread_mutex_lock(&logfs->worker->mutex)) {
            TRACE("pthread_mutex_lock()");
            return NULL;
        }

        while (logfs->write_queue->utilized == 0 && !logfs->worker->stop_thread) {
            if (pthread_cond_wait(&logfs->worker->cond, &logfs->worker->mutex)) {
                TRACE("pthread_cond_wait()");
                return NULL;
            }
        }

        // If there is data in the queue, write it to the device
        if (logfs->write_queue->utilized >= device_block(logfs->device) || logfs->worker->force_write) {
            write_to_disk(logfs);

            logfs->worker->force_write = 0;

            if (pthread_cond_signal(&logfs->worker->cond)) {
                TRACE("pthread_cond_signal()");
                return NULL;
            }
        }

        if (pthread_mutex_unlock(&logfs->worker->mutex)) {
            TRACE("pthread_mutex_unlock()");
            return NULL;
        }

        if (logfs->worker->stop_thread) {
            return NULL;
        }
    }
}

uint64_t get_metadata(struct logfs *logfs) {
    uint64_t utilized;

    char *metadata = malloc(device_block(logfs->device));
    if (!metadata) {
        TRACE("out of memory");
        return -1;
    }

    if (device_read(logfs->device, metadata, 0, device_block(logfs->device))) {
        TRACE("device_read()");
        free(metadata);
        return -1;
    }

    utilized = ((struct logfs_metadata *) metadata)->utilized;

    free(metadata);

    return utilized;
}

void set_metadata(struct logfs *logfs, uint64_t utilized) {
    char *metadata = malloc(device_block(logfs->device));
    if (!metadata) {
        TRACE("out of memory");
        return;
    }

    ((struct logfs_metadata *) metadata)->utilized = utilized;

    if (device_write(logfs->device, metadata, 0, device_block(logfs->device))) {
        TRACE("device_write()");
        free(metadata);
        return;
    }

    free(metadata);
}

int setup_device(struct logfs *logfs, const char *pathname) {
    if (!(logfs->device = device_open(pathname))) {
        return -1;
    }

    logfs->capacity = device_size(logfs->device);
    logfs->utilized = get_metadata(logfs);

    return 0;
}

int setup_queue(struct logfs *logfs) {
    if (!(logfs->write_queue = malloc(sizeof(struct queue)))) {
        return -1;
    }
    memset(logfs->write_queue, 0, sizeof(struct queue));

    logfs->write_queue->head = 0;
    logfs->write_queue->tail = 0;
    logfs->write_queue->capacity = device_block(logfs->device) * WCACHE_BLOCKS;
    logfs->write_queue->utilized = 0;

    if (!(logfs->write_queue->data = malloc(logfs->write_queue->capacity))) {
        return -1;
    }
    memset(logfs->write_queue->data, 0, logfs->write_queue->capacity);

    return 0;
}

int setup_cache(struct logfs *logfs) {
    int i;

    for (i = 0; i < RCACHE_BLOCKS; i++) {
        if (!(logfs->read_cache[i].data = malloc(device_block(logfs->device)))) {
            return -1;
        }
        memset(logfs->read_cache[i].data, 0, device_block(logfs->device));

        logfs->read_cache[i].valid = 0;
        logfs->read_cache[i].idx = i;
    }

    return 0;
}

int setup_worker(struct logfs *logfs) {
    if (!(logfs->worker = malloc(sizeof(struct worker)))) {
        return -1;
    }
    memset(logfs->worker, 0, sizeof(struct worker));

    if (pthread_mutex_init(&logfs->worker->mutex, NULL) ||
        pthread_cond_init(&logfs->worker->cond, NULL) ||
        pthread_create(&logfs->worker->thread, NULL, worker, logfs)) {
        return -1;
    }

    return 0;
}

struct logfs *logfs_open(const char *pathname) {
    struct logfs *logfs;

    assert(safe_strlen(pathname));

    if (!(logfs = malloc(sizeof(struct logfs)))) {
        TRACE("out of memory");
        return NULL;
    }
    memset(logfs, 0, sizeof(struct logfs));

    if (setup_device(logfs, pathname) || setup_queue(logfs) || setup_cache(logfs) || setup_worker(logfs)) {
        logfs_close(logfs);
        TRACE(0);
        return NULL;
    }

    return logfs;
}

void logfs_close(struct logfs *logfs) {
    int i;

    assert(logfs);

    set_metadata(logfs, logfs->utilized);

    if (logfs) {
        if (logfs->worker) {
            if (pthread_mutex_lock(&logfs->worker->mutex)) {
                TRACE("pthread_mutex_lock()");
            }
            logfs->worker->stop_thread = 1;
            if (pthread_cond_signal(&logfs->worker->cond)) {
                TRACE("pthread_cond_signal()");
            }
            if (pthread_mutex_unlock(&logfs->worker->mutex)) {
                TRACE("pthread_mutex_unlock()");
            }
            if (pthread_join(logfs->worker->thread, NULL)) {
                TRACE("pthread_join()");
            }
            if (pthread_mutex_destroy(&logfs->worker->mutex)) {
                TRACE("pthread_mutex_destroy()");
            }
            if (pthread_cond_destroy(&logfs->worker->cond)) {
                TRACE("pthread_cond_destroy()");
            }
        }
        if (logfs->write_queue) {
            FREE(logfs->write_queue->data);
            FREE(logfs->write_queue);
        }
        for (i = 0; i < RCACHE_BLOCKS; ++i) {
            if (logfs->read_cache[i].data) {
                FREE(logfs->read_cache[i].data);
            }
        }
        if (logfs->worker) {
            FREE(logfs->worker);
        }
        if (logfs->write_queue) {
            FREE(logfs->write_queue);
        }
        if (logfs->device) {
            device_close(logfs->device);
        }
        memset(logfs, 0, sizeof(struct logfs));
    }
    FREE(logfs);
}

int logfs_read(struct logfs *logfs, void *buf, uint64_t off, size_t len) {
    uint64_t current_offset, min_idx, max_idx;
    char *buffer_ptr;
    int i;

    if (!buf || !len) {
        return 0;
    }

    off += device_block(logfs->device);

    while (1) {
        if (pthread_mutex_lock(&logfs->worker->mutex)) {
            TRACE("pthread_mutex_lock()");
            return -1;
        }

        while (logfs->write_queue->utilized > 0) {
            if (logfs->write_queue->utilized < device_block(logfs->device)) {
                logfs->worker->force_write = 1;
            }
            if (pthread_cond_wait(&logfs->worker->cond, &logfs->worker->mutex)) {
                TRACE("pthread_cond_wait()");
                return -1;
            }
        }

        if (pthread_mutex_unlock(&logfs->worker->mutex)) {
            TRACE("pthread_mutex_unlock()");
            return -1;
        }

        break;
    }

    // Iterate through blocks and read from cache or device
    current_offset = off;
    buffer_ptr = (char *) buf;

    while (current_offset < off + len) {
        uint64_t block_idx = current_offset / device_block(logfs->device);

        // Check if the block is in the cache
        int found_in_cache = 0;
        for (i = 0; i < RCACHE_BLOCKS; i++) {
            if (logfs->read_cache[i].valid && logfs->read_cache[i].offset == block_idx * device_block(logfs->device)) {
                // Copy data from cache to buffer
                memcpy(buffer_ptr,
                       logfs->read_cache[i].data + (current_offset - block_idx * device_block(logfs->device)),
                       MIN(off + len - current_offset,
                           device_block(logfs->device) - (current_offset % device_block(logfs->device))));
                found_in_cache = 1;
                break;
            }
        }

        // If not found in cache, read from device and update cache
        if (!found_in_cache) {
            char *block = malloc(device_block(logfs->device));
            if (!block) {
                TRACE("out of memory");
                return -1;
            }

            if (device_read(logfs->device, block, block_idx * device_block(logfs->device),
                            device_block(logfs->device))) {
                TRACE("device_read()");
                free(block);
                return -1;
            }

            // get the min and max idx
            min_idx = 0;
            max_idx = 0;
            for (i = 0; i < RCACHE_BLOCKS; i++) {
                if (logfs->read_cache[i].idx < logfs->read_cache[min_idx].idx) {
                    min_idx = i;
                }
                if (logfs->read_cache[i].idx > logfs->read_cache[max_idx].idx) {
                    max_idx = i;
                }
            }

            // replace the block with the min idx
            memcpy(logfs->read_cache[min_idx].data, block, device_block(logfs->device));
            logfs->read_cache[min_idx].valid = 1;
            logfs->read_cache[min_idx].offset = block_idx * device_block(logfs->device);
            logfs->read_cache[min_idx].idx = logfs->read_cache[max_idx].idx + 1;

            // Copy data from block to buffer
            memcpy(buffer_ptr, block + (current_offset - block_idx * device_block(logfs->device)),
                   MIN(off + len - current_offset,
                       device_block(logfs->device) - (current_offset % device_block(logfs->device))));

            free(block);
        }

        // Update buffer pointer and current offset
        buffer_ptr += MIN(off + len - current_offset,
                          device_block(logfs->device) - (current_offset % device_block(logfs->device)));
        current_offset += MIN(off + len - current_offset,
                              device_block(logfs->device) - (current_offset % device_block(logfs->device)));
    }

    return 0;
}

void queue_add(struct logfs *logfs, const void *buf, uint64_t len) {
    if (logfs->write_queue->tail + len <= logfs->write_queue->capacity) {
        memcpy(logfs->write_queue->data + logfs->write_queue->tail, buf, len);
        logfs->write_queue->tail += len;
    } else {
        uint64_t space_at_end = logfs->write_queue->capacity - logfs->write_queue->tail;
        uint64_t leftover = len - space_at_end;
        memcpy(logfs->write_queue->data + logfs->write_queue->tail, buf, space_at_end);
        memcpy(logfs->write_queue->data, (char *) buf + space_at_end, leftover);
        logfs->write_queue->tail = leftover;
    }

    logfs->write_queue->utilized += len;
}

int logfs_append(struct logfs *logfs, const void *buf, uint64_t len) {
    assert(logfs);
    assert(buf || !len);

    while (1) {
        if (pthread_mutex_lock(&logfs->worker->mutex)) {
            TRACE("pthread_mutex_lock()");
            return -1;
        }

        while (logfs->write_queue->utilized + len > logfs->write_queue->capacity) {
            if (pthread_cond_wait(&logfs->worker->cond, &logfs->worker->mutex)) {
                TRACE("pthread_cond_wait()");
                return -1;
            }
        }

        if (logfs->write_queue->utilized + len <= logfs->write_queue->capacity) {
            queue_add(logfs, buf, len);

            if (pthread_cond_signal(&logfs->worker->cond)) {
                TRACE("pthread_cond_signal()");
                return -1;
            }
            if (pthread_mutex_unlock(&logfs->worker->mutex)) {
                TRACE("pthread_mutex_unlock()");
                return -1;
            }
            return 0;
        }

        if (pthread_mutex_unlock(&logfs->worker->mutex)) {
            TRACE("pthread_mutex_unlock()");
            return -1;
        }
    }

    return 0;
}
