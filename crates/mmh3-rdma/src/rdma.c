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
    // The RoCE v2 entries of the port's table: an IPv4 one when the interface has an address, an
    // IPv6 link-local one when it has that. A connection takes the one whose family matches the
    // peer's, since the two ends have to speak the same.
    int global_id_v4;
    int global_id_v6;
    // Reads this device will answer and ask for at once, as it reports them.
    int read_atomic;
    int destination_read_atomic;
} Mmh3Rdma;

// One reliable connection to one peer, with a queue pair of its own so that reads on two of them
// never wait for each other's completions.
typedef struct {
    Mmh3Rdma *device;
    struct ibv_cq *completions;
    struct ibv_qp *queue_pair;
    uint32_t sequence;
    // Pieces of one read this connection keeps on the link at a time.
    int depth;
} Mmh3RdmaLink;

// How many connections one payload may be split over, and one connection's share of it.
#define MMH3_RDMA_PARTS 4

typedef struct {
    Mmh3RdmaLink *link;
    void *region;
    char *local;
    size_t bytes;
    uint64_t remote_address;
    uint32_t remote_key;
} Mmh3RdmaPart;

#define COMPLETION_DEPTH 64
// How long a device name may be, as `mmh3_rdma_device_names` writes them.
#define MMH3_RDMA_NAME_BYTES 64
// How a read is cut up and how many of the pieces are in flight. One request at a time reaches
// 8.7 GB/s over the 200 GbE between two Sparks where `ib_write_bw` reaches 13.3: a read pays a
// round trip for its segments, and the only way to fill the link is to have another request's
// segments on it meanwhile.
#define READ_CHUNK_BYTES (8u << 20)
#define READ_DEPTH 16

// Whether a GID is an IPv4 address mapped into IPv6, which is how RoCE v2 carries one.
static int is_mapped_v4(const union ibv_gid *global_id) {
    static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    return memcmp(global_id->raw, prefix, sizeof(prefix)) == 0;
}

// The RoCE v2 entries of a port's table. The kernel names the type of each in sysfs, and an entry
// is only usable when its address is configured on the interface, which is why a port with no
// address has none. `preferred` is tried first, for a caller that knows which it wants.
static void choose_global_ids(Mmh3Rdma *rdma, const char *device, int preferred) {
    struct ibv_device_attr attributes;
    int count = 16;
    if (ibv_query_device(rdma->context, &attributes) == 0 && rdma->port.gid_tbl_len > 0) {
        count = rdma->port.gid_tbl_len < 64 ? rdma->port.gid_tbl_len : 64;
    }
    for (int order = 0; order < count + 1; order++) {
        // The preferred index goes first and then every index in turn.
        const int index = order == 0 ? preferred : order - 1;
        if (index < 0 || index >= count || (order > 0 && index == preferred)) {
            continue;
        }
        char path[256];
        snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%u/gid_attrs/types/%d", device,
                 (unsigned)rdma->port_number, index);
        FILE *file = fopen(path, "r");
        if (file == NULL) {
            continue;
        }
        char type[32] = {0};
        const char *read = fgets(type, sizeof(type), file);
        fclose(file);
        if (read == NULL || strncmp(type, "RoCE v2", 7) != 0) {
            continue;
        }
        union ibv_gid global_id;
        memset(&global_id, 0, sizeof(global_id));
        if (ibv_query_gid(rdma->context, rdma->port_number, index, &global_id) != 0) {
            continue;
        }
        static const union ibv_gid zero;
        if (memcmp(&global_id, &zero, sizeof(global_id)) == 0) {
            continue;
        }
        if (is_mapped_v4(&global_id)) {
            if (rdma->global_id_v4 < 0) {
                rdma->global_id_v4 = index;
            }
        } else if (rdma->global_id_v6 < 0) {
            rdma->global_id_v6 = index;
        }
    }
}

// The first port of `device` that is up, or of the first device with one when `device` is null.
static int open_device(Mmh3Rdma *rdma, const char *device, int preferred) {
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
        rdma->read_atomic = attributes.max_qp_rd_atom;
        rdma->destination_read_atomic = attributes.max_qp_init_rd_atom;
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
            choose_global_ids(rdma, name, preferred);
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

// Writes the names of the devices with an active port into `names`, each `MMH3_RDMA_NAME_BYTES`
// long, and returns how many there are. A Spark has one NIC on two PCIe links, which is two names
// for the one cable, and using both doubles what the wire carries: 22.2 GB/s against 13.0.
int mmh3_rdma_device_names(char *names, int capacity) {
    int count = 0;
    struct ibv_device **devices = ibv_get_device_list(&count);
    if (devices == NULL) {
        return 0;
    }
    int found = 0;
    for (int index = 0; index < count && found < capacity; index++) {
        struct ibv_context *context = ibv_open_device(devices[index]);
        if (context == NULL) {
            continue;
        }
        struct ibv_device_attr attributes;
        int active = 0;
        if (ibv_query_device(context, &attributes) == 0) {
            for (uint8_t port = 1; port <= attributes.phys_port_cnt && !active; port++) {
                struct ibv_port_attr port_attributes;
                active = ibv_query_port(context, port, &port_attributes) == 0 &&
                         port_attributes.state == IBV_PORT_ACTIVE;
            }
        }
        ibv_close_device(context);
        if (!active) {
            continue;
        }
        const char *name = ibv_get_device_name(devices[index]);
        snprintf(names + (size_t)found * MMH3_RDMA_NAME_BYTES, MMH3_RDMA_NAME_BYTES, "%s", name);
        found++;
    }
    ibv_free_device_list(devices);
    return found;
}

Mmh3Rdma *mmh3_rdma_open(const char *device, int global_id_index) {
    Mmh3Rdma *rdma = calloc(1, sizeof(Mmh3Rdma));
    if (rdma == NULL) {
        return NULL;
    }
    rdma->global_id_v4 = -1;
    rdma->global_id_v6 = -1;
    if (open_device(rdma, device, global_id_index) != 0) {
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
    link->depth = rdma->read_atomic < READ_DEPTH ? rdma->read_atomic : READ_DEPTH;
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
    const int index = rdma->global_id_v4 >= 0 ? rdma->global_id_v4 : rdma->global_id_v6;
    if (index < 0) {
        return -ENODEV;
    }
    union ibv_gid global_id;
    memset(&global_id, 0, sizeof(global_id));
    if (ibv_query_gid(rdma->context, rdma->port_number, index, &global_id) != 0) {
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
    attributes.max_dest_rd_atomic =
        rdma->destination_read_atomic < READ_DEPTH ? rdma->destination_read_atomic : READ_DEPTH;
    attributes.min_rnr_timer = 12;
    attributes.ah_attr.is_global = 1;
    attributes.ah_attr.dlid = peer->local_id;
    attributes.ah_attr.sl = 0;
    attributes.ah_attr.src_path_bits = 0;
    attributes.ah_attr.port_num = rdma->port_number;
    attributes.ah_attr.grh.hop_limit = 64;
    // The two ends have to be of one family, so this side answers the peer's choice.
    union ibv_gid destination;
    memcpy(destination.raw, peer->global_id, sizeof(destination.raw));
    const int index = is_mapped_v4(&destination) ? rdma->global_id_v4 : rdma->global_id_v6;
    if (index < 0) {
        return -EAFNOSUPPORT;
    }
    attributes.ah_attr.grh.sgid_index = (uint8_t)index;
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
    attributes.max_rd_atomic = rdma->read_atomic < READ_DEPTH ? rdma->read_atomic : READ_DEPTH;
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

// Posts one piece of a transfer. The pieces of one call are never distinguished, so they share a
// work request id and the caller only counts their completions.
static int post_piece(Mmh3RdmaLink *link, struct ibv_mr *region, void *local, size_t bytes,
                      uint64_t remote_address, uint32_t remote_key, int opcode) {
    struct ibv_sge segment;
    memset(&segment, 0, sizeof(segment));
    segment.addr = (uint64_t)local;
    segment.length = (uint32_t)bytes;
    segment.lkey = region->lkey;

    struct ibv_send_wr request;
    memset(&request, 0, sizeof(request));
    request.wr_id = 1;
    request.sg_list = &segment;
    request.num_sge = 1;
    request.opcode = (enum ibv_wr_opcode)opcode;
    request.send_flags = IBV_SEND_SIGNALED;
    request.wr.rdma.remote_addr = remote_address;
    request.wr.rdma.rkey = remote_key;

    struct ibv_send_wr *failed = NULL;
    return ibv_post_send(link->queue_pair, &request, &failed) == 0 ? 0 : -errno;
}

// Moves one payload over as many connections as it was split across, keeping several pieces of
// each on its link at a time. Every part is driven from the one loop, so the parts are on their
// links together and the payload takes what all of them carry.
static int transfer(const Mmh3RdmaPart *parts, int count, int opcode, int milliseconds) {
    if (parts == NULL || count < 1 || count > MMH3_RDMA_PARTS) {
        return -EINVAL;
    }
    size_t posted[MMH3_RDMA_PARTS] = {0};
    int flying[MMH3_RDMA_PARTS] = {0};
    int done[MMH3_RDMA_PARTS] = {0};
    int waited = 0;
    int left = 0;
    for (int part = 0; part < count; part++) {
        if (parts[part].link == NULL || parts[part].region == NULL || parts[part].local == NULL) {
            return -EINVAL;
        }
        if (parts[part].bytes > 0) {
            left++;
        } else {
            done[part] = 1;
        }
    }
    while (left > 0) {
        int ready_anywhere = 0;
        for (int part = 0; part < count; part++) {
            if (done[part]) {
                continue;
            }
            const Mmh3RdmaPart *piece = &parts[part];
            Mmh3RdmaLink *link = piece->link;
            // A write needs no room in the peer's read resources, so only a read is held to the
            // connection's depth.
            const int depth =
                opcode == IBV_WR_RDMA_READ ? (link->depth < 1 ? 1 : link->depth) : READ_DEPTH;
            while (flying[part] < depth && posted[part] < piece->bytes) {
                size_t rest = piece->bytes - posted[part];
                size_t step = rest < READ_CHUNK_BYTES ? rest : READ_CHUNK_BYTES;
                int status = post_piece(link, (struct ibv_mr *)piece->region,
                                        piece->local + posted[part], step,
                                        piece->remote_address + posted[part], piece->remote_key,
                                        opcode);
                if (status != 0) {
                    return status;
                }
                posted[part] += step;
                flying[part]++;
            }
            // A transfer of hundreds of megabytes takes tens of milliseconds, so poll rather than
            // wait on an event channel, which would cost a file descriptor and a wake-up for no
            // gain here.
            struct ibv_wc completions[COMPLETION_DEPTH];
            int ready = ibv_poll_cq(link->completions, COMPLETION_DEPTH, completions);
            if (ready < 0) {
                return -EIO;
            }
            for (int index = 0; index < ready; index++) {
                if (completions[index].status != IBV_WC_SUCCESS) {
                    return -(int)completions[index].status;
                }
            }
            flying[part] -= ready;
            ready_anywhere += ready;
            if (flying[part] == 0 && posted[part] == piece->bytes) {
                done[part] = 1;
                left--;
            }
        }
        if (left == 0) {
            break;
        }
        if (ready_anywhere > 0) {
            waited = 0;
            continue;
        }
        if (milliseconds > 0 && waited >= milliseconds * 1000) {
            return -ETIMEDOUT;
        }
        struct timespec pause = {0, 1000};
        nanosleep(&pause, NULL);
        waited++;
    }
    return 0;
}

// Reads the peer's memory into a local registered buffer.
int mmh3_rdma_read_all(const Mmh3RdmaPart *parts, int count, int milliseconds) {
    return transfer(parts, count, IBV_WR_RDMA_READ, milliseconds);
}

// Writes a local registered buffer into the peer's memory. Over the 200 GbE between two Sparks one
// connection reaches 13.0 GB/s where a read reaches 9.6, and the two PCIe links of the one cable
// together reach 22.2, which is why the ranks push their rows and split them over both.
int mmh3_rdma_write_all(const Mmh3RdmaPart *parts, int count, int milliseconds) {
    return transfer(parts, count, IBV_WR_RDMA_WRITE, milliseconds);
}
