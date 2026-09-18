// A small piece of libibverbs for mmh3: a device with memory registered on it, reliable
// connections to other machines, and reads from their memory.
//
// The registrations belong to the device rather than to one connection, so a rank that talks to
// several peers registers its regions once and offers them all one remote key.
//
// NOTE: the verbs structures are wide and their layout belongs to the installed headers, so the
// calls live here in C rather than behind a hand-written Rust layout. Rust sees the handle, the
// address it exchanges over TCP, and four calls.

#include <errno.h>
#include <infiniband/verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// What one side tells the other over TCP so that both can reach the same connection.
typedef struct {
    uint32_t queue_pair;
    uint32_t sequence;
    uint16_t local_id;
    uint8_t global_id[16];
} Mmh3RdmaAddress;

// The device and its protection domain, which the memory regions belong to.
typedef struct {
    struct ibv_context *context;
    struct ibv_pd *protection;
    struct ibv_port_attr port;
    uint8_t port_number;
    int global_id_index;
} Mmh3Rdma;

// One reliable connection to one peer, with a queue pair of its own so that reads on two of them
// never wait for each other's completions.
typedef struct {
    Mmh3Rdma *device;
    struct ibv_cq *completions;
    struct ibv_qp *queue_pair;
    uint32_t sequence;
} Mmh3RdmaLink;

#define COMPLETION_DEPTH 64

// The first port of `device` that is up, or of the first device with one when `device` is null.
static int open_device(Mmh3Rdma *rdma, const char *device) {
    int count = 0;
    struct ibv_device **devices = ibv_get_device_list(&count);
    if (devices == NULL) {
        return -ENODEV;
    }
    int status = -ENODEV;
    for (int index = 0; index < count && status != 0; index++) {
        const char *name = ibv_get_device_name(devices[index]);
        if (device != NULL && *device != '\0' && strcmp(device, name) != 0) {
            continue;
        }
        struct ibv_context *context = ibv_open_device(devices[index]);
        if (context == NULL) {
            continue;
        }
        struct ibv_device_attr attributes;
        if (ibv_query_device(context, &attributes) != 0) {
            ibv_close_device(context);
            continue;
        }
        for (uint8_t port = 1; port <= attributes.phys_port_cnt; port++) {
            struct ibv_port_attr port_attributes;
            if (ibv_query_port(context, port, &port_attributes) != 0) {
                continue;
            }
            if (port_attributes.state != IBV_PORT_ACTIVE) {
                continue;
            }
            rdma->context = context;
            rdma->port = port_attributes;
            rdma->port_number = port;
            status = 0;
            break;
        }
        if (status != 0) {
            ibv_close_device(context);
        }
    }
    ibv_free_device_list(devices);
    return status;
}

Mmh3Rdma *mmh3_rdma_open(const char *device, int global_id_index) {
    Mmh3Rdma *rdma = calloc(1, sizeof(Mmh3Rdma));
    if (rdma == NULL) {
        return NULL;
    }
    rdma->global_id_index = global_id_index;
    if (open_device(rdma, device) != 0) {
        free(rdma);
        return NULL;
    }
    rdma->protection = ibv_alloc_pd(rdma->context);
    if (rdma->protection == NULL) {
        mmh3_rdma_close(rdma);
        return NULL;
    }
    return rdma;
}

void mmh3_rdma_close(Mmh3Rdma *rdma) {
    if (rdma == NULL) {
        return;
    }
    if (rdma->protection != NULL) {
        ibv_dealloc_pd(rdma->protection);
    }
    if (rdma->context != NULL) {
        ibv_close_device(rdma->context);
    }
    free(rdma);
}

// One connection of `rdma`, ready for its peer's address.
Mmh3RdmaLink *mmh3_rdma_link_open(Mmh3Rdma *rdma) {
    if (rdma == NULL) {
        return NULL;
    }
    Mmh3RdmaLink *link = calloc(1, sizeof(Mmh3RdmaLink));
    if (link == NULL) {
        return NULL;
    }
    link->device = rdma;
    link->completions = ibv_create_cq(rdma->context, COMPLETION_DEPTH, NULL, NULL, 0);
    if (link->completions == NULL) {
        mmh3_rdma_link_close(link);
        return NULL;
    }
    struct ibv_qp_init_attr initial;
    memset(&initial, 0, sizeof(initial));
    initial.send_cq = link->completions;
    initial.recv_cq = link->completions;
    initial.qp_type = IBV_QPT_RC;
    initial.cap.max_send_wr = COMPLETION_DEPTH;
    initial.cap.max_recv_wr = COMPLETION_DEPTH;
    initial.cap.max_send_sge = 1;
    initial.cap.max_recv_sge = 1;
    link->queue_pair = ibv_create_qp(rdma->protection, &initial);
    if (link->queue_pair == NULL) {
        mmh3_rdma_link_close(link);
        return NULL;
    }
    struct ibv_qp_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_INIT;
    attributes.port_num = rdma->port_number;
    attributes.pkey_index = 0;
    attributes.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(link->queue_pair, &attributes,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        mmh3_rdma_link_close(link);
        return NULL;
    }
    link->sequence = (uint32_t)(rand() & 0xffffff);
    return link;
}

void mmh3_rdma_link_close(Mmh3RdmaLink *link) {
    if (link == NULL) {
        return;
    }
    if (link->queue_pair != NULL) {
        ibv_destroy_qp(link->queue_pair);
    }
    if (link->completions != NULL) {
        ibv_destroy_cq(link->completions);
    }
    free(link);
}

int mmh3_rdma_link_address(Mmh3RdmaLink *link, Mmh3RdmaAddress *address) {
    if (link == NULL || address == NULL) {
        return -EINVAL;
    }
    Mmh3Rdma *rdma = link->device;
    union ibv_gid global_id;
    memset(&global_id, 0, sizeof(global_id));
    if (ibv_query_gid(rdma->context, rdma->port_number, rdma->global_id_index, &global_id) != 0) {
        return -errno;
    }
    address->queue_pair = link->queue_pair->qp_num;
    address->sequence = link->sequence;
    address->local_id = rdma->port.lid;
    memcpy(address->global_id, global_id.raw, sizeof(address->global_id));
    return 0;
}

// Moves one connection to ready, which both sides do once they have exchanged addresses.
int mmh3_rdma_link_connect(Mmh3RdmaLink *link, const Mmh3RdmaAddress *peer) {
    if (link == NULL || peer == NULL) {
        return -EINVAL;
    }
    Mmh3Rdma *rdma = link->device;
    struct ibv_qp_attr attributes;
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTR;
    attributes.path_mtu = rdma->port.active_mtu;
    attributes.dest_qp_num = peer->queue_pair;
    attributes.rq_psn = peer->sequence;
    attributes.max_dest_rd_atomic = 4;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.dlid = peer->local_id;
    attributes.ah_attr.sl = 0;
    attributes.ah_attr.src_path_bits = 0;
    attributes.ah_attr.port_num = rdma->port_number;
    attributes.ah_attr.grh.hop_limit = 64;
    attributes.ah_attr.grh.sgid_index = (uint8_t)rdma->global_id_index;
    attributes.ah_attr.grh.traffic_class = 0;
    memcpy(attributes.ah_attr.grh.dgid.raw, peer->global_id, sizeof(peer->global_id));
    if (ibv_modify_qp(link->queue_pair, &attributes,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
        return -errno;
    }
    memset(&attributes, 0, sizeof(attributes));
    attributes.qp_state = IBV_QPS_RTS;
    attributes.timeout = 14;
    attributes.retry_cnt = 7;
    attributes.rnr_retry = 7;
    attributes.sq_psn = link->sequence;
    attributes.max_rd_atomic = 4;
    if (ibv_modify_qp(link->queue_pair, &attributes,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
        return -errno;
    }
    return 0;
}

// Registers memory on the device, which every connection of it may then serve. The handle is the
// memory region, which the caller keeps until it releases the buffer.
void *mmh3_rdma_register(Mmh3Rdma *rdma, void *buffer, size_t bytes, uint32_t *remote_key) {
    if (rdma == NULL || buffer == NULL || bytes == 0) {
        return NULL;
    }
    struct ibv_mr *region =
        ibv_reg_mr(rdma->protection, buffer, bytes,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (region == NULL) {
        return NULL;
    }
    if (remote_key != NULL) {
        *remote_key = region->rkey;
    }
    return region;
}

void mmh3_rdma_unregister(void *region) {
    if (region != NULL) {
        ibv_dereg_mr((struct ibv_mr *)region);
    }
}

// Reads `bytes` of the peer's registered memory into a local registered buffer and waits for it.
int mmh3_rdma_read(Mmh3RdmaLink *link, void *region, void *local, size_t bytes,
                   uint64_t remote_address, uint32_t remote_key, int milliseconds) {
    if (link == NULL || region == NULL || local == NULL || bytes == 0) {
        return -EINVAL;
    }
    struct ibv_sge segment;
    memset(&segment, 0, sizeof(segment));
    segment.addr = (uint64_t)local;
    segment.length = (uint32_t)bytes;
    segment.lkey = ((struct ibv_mr *)region)->lkey;

    struct ibv_send_wr request;
    memset(&request, 0, sizeof(request));
    request.wr_id = 1;
    request.sg_list = &segment;
    request.num_sge = 1;
    request.opcode = IBV_WR_RDMA_READ;
    request.send_flags = IBV_SEND_SIGNALED;
    request.wr.rdma.remote_addr = remote_address;
    request.wr.rdma.rkey = remote_key;

    struct ibv_send_wr *failed = NULL;
    if (ibv_post_send(link->queue_pair, &request, &failed) != 0) {
        return -errno;
    }
    // A read of hundreds of megabytes takes tens of milliseconds, so poll rather than wait on an
    // event channel, which would cost a file descriptor and a wake-up for no gain here.
    struct ibv_wc completion;
    for (int waited = 0; milliseconds <= 0 || waited < milliseconds * 1000; waited++) {
        int ready = ibv_poll_cq(link->completions, 1, &completion);
        if (ready < 0) {
            return -EIO;
        }
        if (ready > 0) {
            return completion.status == IBV_WC_SUCCESS ? 0 : -(int)completion.status;
        }
        struct timespec pause = {0, 1000};
        nanosleep(&pause, NULL);
    }
    return -ETIMEDOUT;
}

// Splits a read that passes what one work request may carry.
int mmh3_rdma_read_all(Mmh3RdmaLink *link, void *region, void *local, size_t bytes,
                       uint64_t remote_address, uint32_t remote_key, int milliseconds) {
    const size_t limit = 1u << 30;
    size_t done = 0;
    while (done < bytes) {
        size_t step = bytes - done < limit ? bytes - done : limit;
        int status = mmh3_rdma_read(link, region, (char *)local + done, step, remote_address + done,
                                    remote_key, milliseconds);
        if (status != 0) {
            return status;
        }
        done += step;
    }
    return 0;
}
