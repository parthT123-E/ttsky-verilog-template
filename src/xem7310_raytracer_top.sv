`default_nettype none

// =============================================================================
// XEM7310 / FrontPanel wrapper for the voxel raytracer.
//
// FrontPanel endpoint map
// -----------------------
// WireIns:
//   0x00: starting voxel, one byte-aligned slot per axis --
//         x at bit 0, y at bit 8, z at bit 16, each COORD_REG_BITS wide
//         (7 bits for the current 64^3 grid; host sends 0..63)
//   0x01: [9:0] max_steps, [10] sx, [11] sy, [12] sz,
//         [26:13] pixel_id
//   0x02..0x04: next_x, next_y, next_z
//   0x05..0x07: inc_x, inc_y, inc_z
//
// TriggerIn 0x40:
//   bit 0: reset (stretches to 16 okClk cycles)
//   bit 1: begin/restart scene load (accepted only while tracer is idle)
//   bit 2: submit job
//   bit 3: pop current result
//   bit 4: clear sticky wrapper errors
//   bit 5: abort scene load
//
// Block-Throttled PipeIn 0x80:
//   Exactly SCENE_BYTES bytes / SCENE_WORDS little-endian 32-bit words per
//   scene (see the grid-geometry localparams below; currently 262144 bytes /
//   65536 words for a 128^3 grid). Word N contains voxel addresses 32*N
//   through 32*N+31, LSB first. Linear voxel address is
//   {z[ADDR_COORD_BITS-1:0], y[ADDR_COORD_BITS-1:0], x[ADDR_COORD_BITS-1:0]}.
//   BTPipeIn is fed through a small streaming FIFO (FIFO_DEPTH words) that
//   only smooths burst-vs-serial timing -- it is NOT sized to the whole
//   scene, so BRAM cost stays flat as the grid grows.
//
// Block-Throttled PipeIn 0x81 (bulk ray jobs):
//   A stream of 8-word ray jobs, same field layout as the WireIns above:
//     word 0: xyz      (as WireIn 0x00)
//     word 1: meta     (as WireIn 0x01)
//     words 2-4: next_x, next_y, next_z
//     words 5-7: inc_x, inc_y, inc_z
//   Jobs are queued and dispatched to the tracer automatically, so the host
//   pays one bulk transfer per BATCH instead of 4 USB round trips per ray.
//
// PipeOut 0xA0 (bulk results), enabled by WireIn 0x08 bit 0:
//   A stream of 4-word results: {info, xy, z_steps, seq} -- the first three
//   are identical to WireOuts 0x22/0x23/0x24; seq is a free-running counter
//   so the host can detect dropped or mis-framed results. 4 words keeps each
//   record 16 bytes, satisfying the FrontPanel pipe alignment requirement.
//   The host must read only as many words as WireOut 0x26 reports.
//
// WireIns (cont):
//   0x08: [0] route results to PipeOut 0xA0 instead of the result WireOuts
//
// WireOuts:
//   0x20: status (see assignments below)
//   0x21: [31:0] voxels written (bit-level scene-load progress)
//   0x26: [15:0] result words ready in the PipeOut FIFO,
//         [31:16] job words queued in the PipeIn FIFO
//   0x22: [13:0] pixel_id, [14] hit, [15] timeout, [18:16] face_id
//         face_id 0..5 = the face the ray entered the hit voxel through
//         (0/1=+-x, 2/3=+-y, 4/5=+-z); 6 = ray hit its own starting voxel
//         (no entry face)
//   0x23: [15:0] hit_x, [31:16] hit_y
//   0x24: [15:0] hit_z, [31:16] steps
//   0x25: build/interface identifier ("RTF1")
//
// TriggerOut 0x60:
//   bit 0: a result became available (also pulses for the next queued result)
//   bit 1: scene load completed
//   bit 2: wrapper protocol error occurred
// =============================================================================
module xem7310_raytracer_top (
    input  wire [4:0]  okUH,
    output wire [2:0]  okHU,
    inout  wire [31:0] okUHU,
    inout  wire        okAA,
    output wire [7:0]  led
);

    // -------------------------------------------------------------------------
    // Voxel grid geometry. Bump ADDR_COORD_BITS to resize the scene (grid is
    // always 2**ADDR_COORD_BITS voxels per axis) -- everything below derives
    // from it. Must stay a power-of-two grid per axis: voxel_addr_map.sv
    // builds the RAM address by bit-concatenating {z,y,x} rather than
    // multiplying, so each axis needs a fixed power-of-two-wide address
    // slot. The blk_mem_gen_0 IP must be regenerated (in Vivado) at depth
    // 2**ADDR_BITS to match before this will synthesize correctly.
    // -------------------------------------------------------------------------
    localparam int ADDR_COORD_BITS = 7;                     // 128 voxels/axis
    localparam int ADDR_BITS       = ADDR_COORD_BITS * 3;    // 18
    localparam int MAX_COORD_VAL   = (1 << ADDR_COORD_BITS) - 1;  // 63
    localparam int SCENE_WORDS     = (1 << ADDR_BITS) / 32;  // BTPipeIn words/scene

    // Coordinate REGISTERS carry one bit more than the grid needs. That extra
    // bit is what makes out-of-bounds detectable: stepping off the +face
    // yields MAX_COORD_VAL+1 and stepping off the -face wraps to all-ones,
    // both of which compare > MAX_COORD_VAL in bounds_check. Sizing these
    // registers to ADDR_COORD_BITS instead would make every coordinate
    // trivially in-range, bounds_check could never fire, and rays would wrap
    // around the grid until they hit the max_steps timeout.
    localparam int COORD_REG_BITS  = ADDR_COORD_BITS + 1;    // 8

    // The voxel RAM is RAM_WORD_BITS wide, not 1 bit: the Block Memory
    // Generator caps a port at 2^20 addresses, and a 128^3 grid needs 2^21
    // voxels. 32 x 64K fits, and matching the scene pipe's word width means
    // scene loading writes whole words instead of serialising 32 single-bit
    // writes per word.
    localparam int RAM_WORD_BITS   = 32;
    localparam int RAM_WORD_AW     = $clog2(RAM_WORD_BITS);       // 5
    localparam int WORD_ADDR_BITS  = ADDR_BITS - RAM_WORD_AW;     // 16

    // Streaming buffer that smooths BTPipeIn's bursty 32-bit-word delivery
    // against the core's 1-bit/cycle consumption rate. Deliberately NOT
    // sized to the whole scene, so BRAM usage stays flat as the grid (and
    // therefore SCENE_WORDS) grows.
    //
    // BLOCK_WORDS must match the block size the host passes to
    // WriteToBlockPipeIn (PIPE_BLOCK_SIZE bytes / 4). okBTPipeIn commits a
    // WHOLE block once ep_ready is asserted -- it cannot be throttled
    // mid-block -- so the FIFO must have room for a full block at that
    // moment or words are silently dropped. Depth is two blocks: one being
    // filled by the pipe while the loader drains the other.
    localparam int BLOCK_WORDS = 256;              // host PIPE_BLOCK_SIZE = 1024 B
    localparam int FIFO_DEPTH  = 2 * BLOCK_WORDS;  // 512 words = 16 Kbit
    localparam int FIFO_AW     = $clog2(FIFO_DEPTH);

    // -------------------------------------------------------------------------
    // Bulk job / result streaming
    // -------------------------------------------------------------------------
    // Driving one ray over WireIns costs 4 USB round trips (~1.15 ms measured),
    // against ~5 us of actual tracing -- the link, not the tracer, is the
    // bottleneck. These two pipes move jobs in and results out in bulk so the
    // per-ray cost becomes bandwidth rather than latency.
    //
    // The job FIFO is read by our own sequencer, so BRAM's 1-cycle read latency
    // is fine. The result FIFO feeds okPipeOut, which samples ep_datain
    // combinationally on ep_read, so it must be distributed (LUT) RAM to give
    // first-word-fall-through behaviour.
    // Both FIFOs must hold a WHOLE host batch. The host writes every job of a
    // batch before reading any result, so in the worst case all of that
    // batch's results are queued while the job write is still in flight. If
    // the result FIFO overflowed there the tracer would stall, jobs would stop
    // draining, and the blocking job write would never complete -- a deadlock
    // rather than a wrong answer. Sized here for BATCH_RAYS = 1024, with the
    // result side given 2x margin since it is the deadlock-critical one.
    localparam int JOB_WORDS      = 8;
    localparam int JOB_FIFO_DEPTH = 8192;                  // 1024 queued jobs
    localparam int JOB_FIFO_AW    = $clog2(JOB_FIFO_DEPTH);

    localparam int RES_WORDS      = 4;   // {info, xy, z_steps, tag} = 16 bytes
    localparam int RES_FIFO_DEPTH = 8192;                 // 2048 queued results
    localparam int RES_FIFO_AW    = $clog2(RES_FIFO_DEPTH);

    // The 4th word of each result is {MAGIC, 24-bit sequence}. The magic byte
    // makes the record boundary self-identifying: info/xy/z_steps never set
    // bits [31:24] (info uses [18:0]; coordinates and step counts are 16-bit),
    // so only a genuine tag word can carry it. Without it the framing is
    // ambiguous -- consecutive pixel_ids make info words look like a running
    // counter too.
    localparam logic [7:0] RES_TAG_MAGIC = 8'hA5;

    // -------------------------------------------------------------------------
    // FrontPanel host and endpoint buses
    // -------------------------------------------------------------------------
    wire         okClk;
    wire [112:0] okHE;
    wire [64:0]  okEH;

    wire [31:0] wi_job_xyz;
    wire [31:0] wi_job_meta;
    wire [31:0] wi_next_x;
    wire [31:0] wi_next_y;
    wire [31:0] wi_next_z;
    wire [31:0] wi_inc_x;
    wire [31:0] wi_inc_y;
    wire [31:0] wi_inc_z;
    wire [31:0] ti_control;
    wire [31:0] wi_pipe_ctrl;

    wire         pipe_write;
    wire         pipe_blockstrobe;
    wire [31:0]  pipe_data;
    wire         pipe_ready;

    wire         jobpipe_write;
    wire         jobpipe_blockstrobe;
    wire [31:0]  jobpipe_data;
    wire         jobpipe_ready;

    wire         respipe_read;
    wire [31:0]  respipe_data;

    logic [31:0] wo_status;
    logic [31:0] wo_scene_progress;
    logic [31:0] wo_result_info;
    logic [31:0] wo_result_xy;
    logic [31:0] wo_result_z_steps;
    // Bump this whenever the host-visible endpoint map changes, so a stale
    // bitstream is reported as a mismatch instead of failing later as an
    // unexplained pipe timeout.
    //   RTF1 - wires/triggers + scene BTPipeIn 0x80
    //   RTF2 - adds bulk job BTPipeIn 0x81, result PipeOut 0xA0,
    //          WireIn 0x08, WireOut 0x26
    //   RTF3 - result tag word carries RES_TAG_MAGIC; result PipeOut uses a
    //          registered (non-FWFT) read so word 0 is no longer skipped
    //   RTF4 - grid grown to 128^3 (scene payload 32 KB -> 256 KB)
    //   RTF5 - job/result FIFOs deepened for BATCH_RAYS = 1024. A host using
    //          1024-ray batches against an RTF4 bitstream DEADLOCKS (its
    //          result FIFO cannot hold the batch), so the ID must gate it.
    wire  [31:0] wo_build_id = 32'h5254_4635; // ASCII "RTF5"
    logic [31:0] wo_pipe_status;
    logic [31:0] to_events;

    // Seven WireOuts, one TriggerOut, two BTPipeIns and one PipeOut drive okEH.
    wire [11*65-1:0] okEHx;

    okHost u_ok_host (
        .okUH(okUH),
        .okHU(okHU),
        .okUHU(okUHU),
        .okAA(okAA),
        .okClk(okClk),
        .okHE(okHE),
        .okEH(okEH),
        .dna(),
        .dna_valid()
    );

    okWireOR #(.N(11)) u_ok_wire_or (
        .okEH(okEH),
        .okEHx(okEHx)
    );

    okWireIn u_wi00 (.okHE(okHE), .ep_addr(8'h00), .ep_dataout(wi_job_xyz));
    okWireIn u_wi01 (.okHE(okHE), .ep_addr(8'h01), .ep_dataout(wi_job_meta));
    okWireIn u_wi02 (.okHE(okHE), .ep_addr(8'h02), .ep_dataout(wi_next_x));
    okWireIn u_wi03 (.okHE(okHE), .ep_addr(8'h03), .ep_dataout(wi_next_y));
    okWireIn u_wi04 (.okHE(okHE), .ep_addr(8'h04), .ep_dataout(wi_next_z));
    okWireIn u_wi05 (.okHE(okHE), .ep_addr(8'h05), .ep_dataout(wi_inc_x));
    okWireIn u_wi06 (.okHE(okHE), .ep_addr(8'h06), .ep_dataout(wi_inc_y));
    okWireIn u_wi07 (.okHE(okHE), .ep_addr(8'h07), .ep_dataout(wi_inc_z));
    okWireIn u_wi08 (.okHE(okHE), .ep_addr(8'h08), .ep_dataout(wi_pipe_ctrl));

    okTriggerIn u_ti40 (
        .okHE(okHE), .ep_addr(8'h40), .ep_clk(okClk),
        .ep_trigger(ti_control)
    );

    okWireOut u_wo20 (
        .okHE(okHE), .okEH(okEHx[0*65 +: 65]),
        .ep_addr(8'h20), .ep_datain(wo_status)
    );
    okWireOut u_wo21 (
        .okHE(okHE), .okEH(okEHx[1*65 +: 65]),
        .ep_addr(8'h21), .ep_datain(wo_scene_progress)
    );
    okWireOut u_wo22 (
        .okHE(okHE), .okEH(okEHx[2*65 +: 65]),
        .ep_addr(8'h22), .ep_datain(wo_result_info)
    );
    okWireOut u_wo23 (
        .okHE(okHE), .okEH(okEHx[3*65 +: 65]),
        .ep_addr(8'h23), .ep_datain(wo_result_xy)
    );
    okWireOut u_wo24 (
        .okHE(okHE), .okEH(okEHx[4*65 +: 65]),
        .ep_addr(8'h24), .ep_datain(wo_result_z_steps)
    );
    okWireOut u_wo25 (
        .okHE(okHE), .okEH(okEHx[5*65 +: 65]),
        .ep_addr(8'h25), .ep_datain(wo_build_id)
    );

    okWireOut u_wo26 (
        .okHE(okHE), .okEH(okEHx[6*65 +: 65]),
        .ep_addr(8'h26), .ep_datain(wo_pipe_status)
    );

    okTriggerOut u_to60 (
        .okHE(okHE), .okEH(okEHx[7*65 +: 65]),
        .ep_addr(8'h60), .ep_clk(okClk), .ep_trigger(to_events)
    );

    okBTPipeIn u_bti80 (
        .okHE(okHE), .okEH(okEHx[8*65 +: 65]),
        .ep_addr(8'h80), .ep_write(pipe_write),
        .ep_blockstrobe(pipe_blockstrobe), .ep_dataout(pipe_data),
        .ep_ready(pipe_ready)
    );

    okBTPipeIn u_bti81 (
        .okHE(okHE), .okEH(okEHx[9*65 +: 65]),
        .ep_addr(8'h81), .ep_write(jobpipe_write),
        .ep_blockstrobe(jobpipe_blockstrobe), .ep_dataout(jobpipe_data),
        .ep_ready(jobpipe_ready)
    );

    okPipeOut u_po_a0 (
        .okHE(okHE), .okEH(okEHx[10*65 +: 65]),
        .ep_addr(8'hA0), .ep_read(respipe_read), .ep_datain(respipe_data)
    );

    // -------------------------------------------------------------------------
    // Power-up and host-requested reset. Artix-7 register INIT values make the
    // shift register start asserted immediately after configuration.
    // -------------------------------------------------------------------------
    logic [15:0] reset_sr = 16'hffff;
    always_ff @(posedge okClk) begin
        if (ti_control[0])
            reset_sr <= 16'hffff;
        else
            reset_sr <= {reset_sr[14:0], 1'b0};
    end
    wire rst_n = ~(|reset_sr);

    // -------------------------------------------------------------------------
    // Raytracer job and result interface
    // -------------------------------------------------------------------------
    logic        job_pending;
    logic [COORD_REG_BITS-1:0] job_x;
    logic [COORD_REG_BITS-1:0] job_y;
    logic [COORD_REG_BITS-1:0] job_z;
    logic        job_sx;
    logic        job_sy;
    logic        job_sz;
    logic [31:0] job_next_x;
    logic [31:0] job_next_y;
    logic [31:0] job_next_z;
    logic [31:0] job_inc_x;
    logic [31:0] job_inc_y;
    logic [31:0] job_inc_z;
    logic [9:0]  job_max_steps;
    logic [13:0] job_pixel_id;

    wire         rt_job_ready;
    wire         rt_result_valid;
    wire [13:0]  rt_result_pixel_id;
    wire         rt_result_hit;
    wire         rt_result_timeout;
    wire [15:0]  rt_result_x;
    wire [15:0]  rt_result_y;
    wire [15:0]  rt_result_z;
    wire [2:0]   rt_result_face;
    wire [15:0]  rt_result_steps;
    wire         rt_tracer_idle;

    wire         rt_load_ready;
    wire [15:0]  rt_write_count;
    wire         rt_load_complete;
    // Driven below once the result-pipe sequencer is declared: either the
    // legacy TriggerIn pop or the bulk pipe's own drain retires a result.
    wire         rt_result_ready;

    // Legacy single-result outputs from raytracer_top are intentionally unused;
    // the streaming result port below is the authoritative interface.
    wire         unused_ray_done;
    wire         unused_ray_hit;
    wire         unused_ray_timeout;
    wire [15:0]  unused_hit_x;
    wire [15:0]  unused_hit_y;
    wire [15:0]  unused_hit_z;
    wire [2:0]   unused_hit_face;
    wire [15:0]  unused_steps;

    // -------------------------------------------------------------------------
    // Scene PipeIn buffer. Words go straight into the voxel RAM, which is
    // RAM_WORD_BITS wide -- no bit serialisation needed.
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) logic [31:0] pipe_fifo [0:FIFO_DEPTH-1];
    logic [FIFO_AW-1:0] fifo_wr_ptr;
    logic [FIFO_AW-1:0] fifo_rd_ptr;
    logic [FIFO_AW:0]   fifo_count;
    logic [31:0]        accepted_words;  // counts up to SCENE_WORDS, not FIFO_DEPTH

    logic        scene_loading;
    logic        scene_loaded;
    logic [31:0] voxels_written;
    logic [WORD_ADDR_BITS-1:0] word_addr;   // WORD address into the voxel RAM
    logic [31:0] scene_word;                // word currently being written
    logic        word_ready;                // scene_word holds a valid word

    logic pipe_error;
    logic job_overflow;
    logic result_underflow;

    wire fifo_full = (fifo_count == FIFO_DEPTH);
    wire fifo_push = pipe_write && scene_loading &&
                     (accepted_words < SCENE_WORDS) && !fifo_full;
    // Fetch the next word only once the current one has been consumed, so the
    // synchronous FIFO read has a full cycle to land in scene_word.
    wire fifo_pop  = scene_loading && !word_ready && (fifo_count != 0);

    // Only advertise readiness when a COMPLETE block still fits (see
    // BLOCK_WORDS above): once okBTPipeIn starts a block it pushes every
    // word of it regardless of ep_ready, so anything less would drop data.
    wire fifo_block_space = (fifo_count <= (FIFO_DEPTH - BLOCK_WORDS));

    assign pipe_ready = scene_loading && (accepted_words < SCENE_WORDS) &&
                        fifo_block_space;

    wire rt_load_valid = scene_loading && word_ready;
    wire [31:0] rt_load_data = scene_word;
    wire rt_load_accept = rt_load_valid && rt_load_ready;

    logic result_valid_d;
    logic result_pop_d;

    // -------------------------------------------------------------------------
    // Bulk job queue (BTPipeIn 0x81 -> job registers -> tracer)
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) logic [31:0] job_fifo [0:JOB_FIFO_DEPTH-1];
    logic [JOB_FIFO_AW-1:0] job_wr_ptr;
    logic [JOB_FIFO_AW-1:0] job_rd_ptr;
    logic [JOB_FIFO_AW:0]   job_count;
    logic [31:0]            job_fifo_dout;
    logic                   jobq_active;   // mid-way through loading a job
    logic [2:0]             jobq_idx;
    logic                   job_pipe_error;

    wire job_fifo_full = (job_count == JOB_FIFO_DEPTH);
    wire job_push      = jobpipe_write && !job_fifo_full;
    wire job_pop       = jobq_active;

    // Same whole-block reservation rule as the scene pipe: okBTPipeIn cannot
    // be throttled mid-block, so only advertise readiness when a full block fits.
    assign jobpipe_ready = (job_count <= (JOB_FIFO_DEPTH - BLOCK_WORDS));

    // Read one word ahead so the BRAM's registered output presents the right
    // word on the cycle the sequencer consumes it.
    wire [JOB_FIFO_AW-1:0] job_rd_next = job_pop ? (job_rd_ptr + 1'b1) : job_rd_ptr;

    // -------------------------------------------------------------------------
    // Bulk result queue (tracer -> PipeOut 0xA0)
    // -------------------------------------------------------------------------
    // okPipeOut samples ep_datain the cycle AFTER it asserts ep_read, so the
    // FIFO presents a REGISTERED output that updates on each read -- i.e. a
    // standard synchronous FIFO, not first-word-fall-through. Driving
    // ep_datain combinationally from res_fifo[res_rd_ptr] and advancing the
    // pointer on the same cycle skips the very first word of every stream.
    // Because the read is registered, this can live in block RAM.
    (* ram_style = "block" *) logic [31:0] res_fifo [0:RES_FIFO_DEPTH-1];
    logic [31:0]            res_dout;
    logic [RES_FIFO_AW-1:0] res_wr_ptr;
    logic [RES_FIFO_AW-1:0] res_rd_ptr;
    logic [RES_FIFO_AW:0]   res_count;
    logic [2:0]             resq_state;   // 0 = idle, 1..4 = emitting words
    logic [31:0]            res_seq;      // free-running, for drop detection
    logic                   res_pipe_error;

    wire result_pipe_en = wi_pipe_ctrl[0];
    wire res_space_ok   = (res_count <= (RES_FIFO_DEPTH - RES_WORDS));
    wire res_push       = (resq_state != 3'd0);
    wire res_pop        = respipe_read && (res_count != 0);

    assign respipe_data = res_dout;

    // The pipe path drains the tracer's result FIFO itself; the legacy
    // TriggerIn pop stays available when the pipe is disabled.
    wire res_pipe_take = (resq_state == 3'd4);

    logic [31:0] res_word;
    always_comb begin
        case (resq_state)
            3'd1:    res_word = wo_result_info;
            3'd2:    res_word = wo_result_xy;
            3'd3:    res_word = wo_result_z_steps;
            default: res_word = {RES_TAG_MAGIC, res_seq[23:0]};   // 3'd4
        endcase
    end

    assign rt_result_ready = (ti_control[3] || res_pipe_take) && rt_result_valid;

    always_ff @(posedge okClk) begin
        if (!rst_n) begin
            job_pending       <= 1'b0;
            job_x             <= '0;
            job_y             <= '0;
            job_z             <= '0;
            job_sx            <= 1'b0;
            job_sy            <= 1'b0;
            job_sz            <= 1'b0;
            job_next_x        <= '0;
            job_next_y        <= '0;
            job_next_z        <= '0;
            job_inc_x         <= '0;
            job_inc_y         <= '0;
            job_inc_z         <= '0;
            job_max_steps     <= '0;
            job_pixel_id      <= '0;

            fifo_wr_ptr       <= '0;
            fifo_rd_ptr       <= '0;
            fifo_count        <= '0;
            accepted_words    <= '0;
            scene_loading     <= 1'b0;
            scene_loaded      <= 1'b0;
            voxels_written    <= '0;
            word_addr         <= '0;
            scene_word        <= '0;
            word_ready        <= 1'b0;

            pipe_error        <= 1'b0;
            job_overflow      <= 1'b0;
            result_underflow  <= 1'b0;
            result_valid_d    <= 1'b0;
            result_pop_d      <= 1'b0;
            to_events         <= '0;

            job_wr_ptr        <= '0;
            job_rd_ptr        <= '0;
            job_count         <= '0;
            job_fifo_dout     <= '0;
            jobq_active       <= 1'b0;
            jobq_idx          <= '0;
            job_pipe_error    <= 1'b0;

            res_wr_ptr        <= '0;
            res_rd_ptr        <= '0;
            res_dout          <= '0;
            res_count         <= '0;
            resq_state        <= 3'd0;
            res_seq           <= '0;
            res_pipe_error    <= 1'b0;
        end else begin
            to_events    <= '0;
            result_pop_d <= ti_control[3] && rt_result_valid;

            // Pulse when the first result arrives or when popping exposes the
            // next entry while result_valid remains asserted.
            if (rt_result_valid && (!result_valid_d || result_pop_d))
                to_events[0] <= 1'b1;
            result_valid_d <= rt_result_valid;

            if (ti_control[4]) begin
                pipe_error       <= 1'b0;
                job_overflow     <= 1'b0;
                result_underflow <= 1'b0;
                job_pipe_error   <= 1'b0;
                res_pipe_error   <= 1'b0;
            end

            // Keep job_valid asserted until the raytracer accepts it.
            if (job_pending && rt_job_ready)
                job_pending <= 1'b0;

            if (ti_control[2]) begin
                if (!job_pending && !scene_loading) begin
                    job_x         <= wi_job_xyz[0  +: COORD_REG_BITS];
                    job_y         <= wi_job_xyz[8  +: COORD_REG_BITS];
                    job_z         <= wi_job_xyz[16 +: COORD_REG_BITS];
                    job_max_steps <= wi_job_meta[9:0];
                    job_sx        <= wi_job_meta[10];
                    job_sy        <= wi_job_meta[11];
                    job_sz        <= wi_job_meta[12];
                    job_pixel_id  <= wi_job_meta[26:13];
                    job_next_x    <= wi_next_x;
                    job_next_y    <= wi_next_y;
                    job_next_z    <= wi_next_z;
                    job_inc_x     <= wi_inc_x;
                    job_inc_y     <= wi_inc_y;
                    job_inc_z     <= wi_inc_z;
                    job_pending   <= 1'b1;
                end else begin
                    job_overflow <= 1'b1;
                    to_events[2] <= 1'b1;
                end
            end

            if (ti_control[3] && !rt_result_valid) begin
                result_underflow <= 1'b1;
                to_events[2]     <= 1'b1;
            end

            // ---------------------------------------------------------------
            // Bulk job queue: BTPipeIn 0x81 -> job_fifo -> job registers.
            // ---------------------------------------------------------------
            // One word arrives per cycle from the pipe; the sequencer below
            // pulls JOB_WORDS of them into the same job registers the WireIn
            // path writes, then raises job_pending exactly as that path does.
            job_fifo_dout <= job_fifo[job_rd_next];

            if (job_push) begin
                job_fifo[job_wr_ptr] <= jobpipe_data;
                job_wr_ptr           <= job_wr_ptr + 1'b1;
            end
            if (jobpipe_write && job_fifo_full) begin
                job_pipe_error <= 1'b1;
                to_events[2]   <= 1'b1;
            end

            if (jobq_active) begin
                case (jobq_idx)
                    3'd0: begin
                        job_x <= job_fifo_dout[0  +: COORD_REG_BITS];
                        job_y <= job_fifo_dout[8  +: COORD_REG_BITS];
                        job_z <= job_fifo_dout[16 +: COORD_REG_BITS];
                    end
                    3'd1: begin
                        job_max_steps <= job_fifo_dout[9:0];
                        job_sx        <= job_fifo_dout[10];
                        job_sy        <= job_fifo_dout[11];
                        job_sz        <= job_fifo_dout[12];
                        job_pixel_id  <= job_fifo_dout[26:13];
                    end
                    3'd2:    job_next_x <= job_fifo_dout;
                    3'd3:    job_next_y <= job_fifo_dout;
                    3'd4:    job_next_z <= job_fifo_dout;
                    3'd5:    job_inc_x  <= job_fifo_dout;
                    3'd6:    job_inc_y  <= job_fifo_dout;
                    default: job_inc_z  <= job_fifo_dout;
                endcase
                job_rd_ptr <= job_rd_ptr + 1'b1;
                if (jobq_idx == 3'd7) begin
                    jobq_active <= 1'b0;
                    job_pending <= 1'b1;   // overrides the clear above
                end else begin
                    jobq_idx <= jobq_idx + 1'b1;
                end
            end else if (!ti_control[2] && !job_pending && !scene_loading &&
                         (job_count >= JOB_WORDS)) begin
                jobq_idx    <= 3'd0;
                jobq_active <= 1'b1;
            end

            case ({job_push, job_pop})
                2'b10:   job_count <= job_count + 1'b1;
                2'b01:   job_count <= job_count - 1'b1;
                default: ;
            endcase

            // ---------------------------------------------------------------
            // Bulk result queue: tracer result FIFO -> res_fifo -> PipeOut.
            // ---------------------------------------------------------------
            // Emits RES_WORDS words per result, then retires it from the
            // tracer via rt_result_ready (res_pipe_take, state 4).
            if (resq_state != 3'd0) begin
                res_fifo[res_wr_ptr] <= res_word;
                res_wr_ptr           <= res_wr_ptr + 1'b1;
                if (resq_state == 3'd4) begin
                    resq_state <= 3'd0;
                    res_seq    <= res_seq + 1'b1;
                end else begin
                    resq_state <= resq_state + 3'd1;
                end
            end else if (result_pipe_en && rt_result_valid && res_space_ok) begin
                resq_state <= 3'd1;
            end

            // Registered read: the word fetched here is what okPipeOut
            // samples on the following cycle.
            if (res_pop) begin
                res_dout   <= res_fifo[res_rd_ptr];
                res_rd_ptr <= res_rd_ptr + 1'b1;
            end

            if (respipe_read && (res_count == 0)) begin
                res_pipe_error <= 1'b1;
                to_events[2]   <= 1'b1;
            end

            case ({res_push, res_pop})
                2'b10:   res_count <= res_count + 1'b1;
                2'b01:   res_count <= res_count - 1'b1;
                default: ;
            endcase

            // Scene loading may only start from a completely idle engine.
            // Otherwise load_mode would stop active contexts from draining.
            if (ti_control[1]) begin
                if (rt_tracer_idle && !job_pending) begin
                    fifo_wr_ptr       <= '0;
                    fifo_rd_ptr       <= '0;
                    fifo_count        <= '0;
                    accepted_words    <= '0;
                    scene_loading     <= 1'b1;
                    scene_loaded      <= 1'b0;
                    voxels_written    <= '0;
                    word_addr         <= '0;
                    scene_word        <= '0;
                    word_ready        <= 1'b0;
                end else begin
                    pipe_error  <= 1'b1;
                    to_events[2] <= 1'b1;
                end
            end else if (ti_control[5]) begin
                fifo_wr_ptr       <= '0;
                fifo_rd_ptr       <= '0;
                fifo_count        <= '0;
                accepted_words    <= '0;
                scene_loading     <= 1'b0;
                scene_loaded      <= 1'b0;
                voxels_written    <= '0;
                word_addr         <= '0;
                word_ready        <= 1'b0;
            end else if (scene_loading) begin
                if (pipe_write && !fifo_push) begin
                    pipe_error  <= 1'b1;
                    to_events[2] <= 1'b1;
                end

                if (fifo_push) begin
                    pipe_fifo[fifo_wr_ptr] <= pipe_data;
                    fifo_wr_ptr            <= fifo_wr_ptr + 1'b1;
                    accepted_words         <= accepted_words + 1'b1;
                end

                // Synchronous FIFO read: scene_word holds the word from the
                // cycle after the pop, which is when word_ready also rises.
                if (fifo_pop) begin
                    scene_word  <= pipe_fifo[fifo_rd_ptr];
                    fifo_rd_ptr <= fifo_rd_ptr + 1'b1;
                    word_ready  <= 1'b1;
                end

                case ({fifo_push, fifo_pop})
                    2'b10: fifo_count <= fifo_count + 1'b1;
                    2'b01: fifo_count <= fifo_count - 1'b1;
                    default: fifo_count <= fifo_count;
                endcase

                // One accepted beat writes a whole RAM word, i.e.
                // RAM_WORD_BITS voxels. fifo_pop and this are mutually
                // exclusive (one needs word_ready low, the other high), so
                // the two word_ready assignments never collide.
                if (rt_load_accept) begin
                    voxels_written <= voxels_written + RAM_WORD_BITS;
                    word_ready     <= 1'b0;

                    if (word_addr == {WORD_ADDR_BITS{1'b1}}) begin
                        scene_loading <= 1'b0;
                        scene_loaded  <= 1'b1;
                        to_events[1]  <= 1'b1;
                    end else begin
                        word_addr <= word_addr + 1'b1;
                    end
                end
            end
        end
    end

    raytracer_top #(
        .COORD_WIDTH(16),
        .COORD_W(COORD_REG_BITS),
        .TIMER_WIDTH(32),
        .W(32),
        .MAX_VAL(MAX_COORD_VAL),
        .ADDR_BITS(ADDR_BITS),
        .WORD_BITS(RAM_WORD_BITS),
        .X_BITS(COORD_REG_BITS),
        .Y_BITS(COORD_REG_BITS),
        .Z_BITS(COORD_REG_BITS),
        .MAX_STEPS_BITS(10),
        .STEP_COUNT_WIDTH(16),
        .PIXEL_ID_WIDTH(14),
        .RAY_ID_WIDTH(3)
    ) u_raytracer (
        .clk(okClk),
        .rst_n(rst_n),

        .job_valid(job_pending),
        .job_ready(rt_job_ready),
        .job_ix0(job_x),
        .job_iy0(job_y),
        .job_iz0(job_z),
        .job_sx(job_sx),
        .job_sy(job_sy),
        .job_sz(job_sz),
        .job_next_x(job_next_x),
        .job_next_y(job_next_y),
        .job_next_z(job_next_z),
        .job_inc_x(job_inc_x),
        .job_inc_y(job_inc_y),
        .job_inc_z(job_inc_z),
        .job_max_steps(job_max_steps),
        .job_pixel_id(job_pixel_id),

        .load_mode(scene_loading),
        .load_valid(rt_load_valid),
        .load_ready(rt_load_ready),
        .load_addr(word_addr),
        .load_data(rt_load_data),
        .write_count(rt_write_count),
        .load_complete(rt_load_complete),

        .ray_done(unused_ray_done),
        .ray_hit(unused_ray_hit),
        .ray_timeout(unused_ray_timeout),
        .hit_voxel_x(unused_hit_x),
        .hit_voxel_y(unused_hit_y),
        .hit_voxel_z(unused_hit_z),
        .hit_face_id(unused_hit_face),
        .steps_taken(unused_steps),

        .result_valid(rt_result_valid),
        .result_ready(rt_result_ready),
        .result_pixel_id(rt_result_pixel_id),
        .result_hit(rt_result_hit),
        .result_timeout(rt_result_timeout),
        .result_hit_voxel_x(rt_result_x),
        .result_hit_voxel_y(rt_result_y),
        .result_hit_voxel_z(rt_result_z),
        .result_face_id(rt_result_face),
        .result_steps(rt_result_steps),
        .tracer_idle(rt_tracer_idle)
    );

    // -------------------------------------------------------------------------
    // Host-visible status and results
    // -------------------------------------------------------------------------
    always_comb begin
        wo_status = 32'd0;
        wo_status[0]     = rst_n;
        wo_status[1]     = scene_loading;
        wo_status[2]     = scene_loaded;
        wo_status[3]     = (fifo_count != 0) || word_ready;
        wo_status[4]     = job_pending;
        wo_status[5]     = rt_job_ready;
        wo_status[6]     = rt_result_valid;
        wo_status[7]     = rt_tracer_idle;
        wo_status[8]     = rt_load_ready;
        wo_status[9]     = pipe_ready;
        wo_status[10]    = fifo_full;
        wo_status[11]    = pipe_error;
        wo_status[12]    = job_overflow;
        wo_status[13]    = result_underflow;
        wo_status[14]    = job_pipe_error;
        wo_status[15]    = res_pipe_error;
        wo_status[26:16] = fifo_count;

        wo_scene_progress = voxels_written;

        wo_pipe_status = 32'd0;
        wo_pipe_status[15:0]  = res_count;   // words ready to read from 0xA0
        wo_pipe_status[31:16] = job_count;   // words still queued from 0x81

        wo_result_info = 32'd0;
        wo_result_info[13:0]  = rt_result_pixel_id;
        wo_result_info[14]    = rt_result_hit;
        wo_result_info[15]    = rt_result_timeout;
        wo_result_info[18:16] = rt_result_face;

        wo_result_xy      = {rt_result_y, rt_result_x};
        wo_result_z_steps = {rt_result_steps, rt_result_z};
    end

    // XEM7310 LEDs are active-low/open-drain in the Opal Kelly reference.
    function automatic [7:0] xem7310_led(input [7:0] value);
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                xem7310_led[i] = value[i] ? 1'b0 : 1'bz;
        end
    endfunction

    assign led = xem7310_led({pipe_error | job_overflow | result_underflow,
                              rt_result_valid, job_pending, pipe_ready,
                              scene_loaded, scene_loading, rt_tracer_idle, rst_n});

    // pipe_blockstrobe is consumed by the endpoint internally. The streaming
    // FIFO backpressures via pipe_ready whenever it fills, so the BTPipeIn
    // block size only affects burst granularity, never correctness.
    wire unused_pipe_blockstrobe = pipe_blockstrobe;
    wire unused_jobpipe_blockstrobe = jobpipe_blockstrobe;
    wire unused_rt_load_complete = rt_load_complete;
    wire [15:0] unused_rt_write_count = rt_write_count;

endmodule

`default_nettype wire
