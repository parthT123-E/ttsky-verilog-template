// =============================================================================
// Streaming QSPI voxel ray stepper with five in-flight ray contexts.
//
// Scene storage remains off chip. The host streams 4x4x4 aligned voxel tiles
// into context-owned cache banks. Every context can read all cache banks, but
// context N can only refill bank N.
// =============================================================================
`default_nettype none

module qspi_stream_raytracer (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    localparam [7:0] CMD_RESET         = 8'h00;
    localparam [7:0] CMD_WRITE_CONTEXT = 8'h10;
    localparam [7:0] CMD_FILL_CACHE    = 8'h20;
    localparam [7:0] CMD_STEP          = 8'h30;
    localparam [7:0] CMD_RUN_N         = 8'h31;
    localparam [7:0] CMD_READ_STATUS   = 8'h40;
    localparam [7:0] CMD_READ_REQUEST  = 8'h41;
    localparam [7:0] CMD_READ_RESULT   = 8'h42;
    localparam [7:0] CMD_POP_RESULT    = 8'h43;

    localparam [2:0] NUM_CONTEXTS = 3'd5;
    localparam [2:0] RESULT_DEPTH = 3'd4;
    localparam [5:0] SCENE_MAX = 6'd31;
    localparam [4:0] CONTEXT_BYTES = 5'd18;

    wire qspi_sck_in  = ui_in[0];
    wire qspi_cs_n_in = ui_in[1];

    reg sck_meta;
    reg sck_sync;
    reg sck_prev;
    reg cs_meta;
    reg cs_sync;
    reg cs_prev;
    reg [3:0] dq_meta;
    reg [3:0] dq_sync;

    wire cs_active = ~cs_sync;
    wire sck_rise  = cs_active &  sck_sync & ~sck_prev;
    wire sck_fall  = cs_active & ~sck_sync &  sck_prev;
    wire cs_start  = ~cs_sync & cs_prev;
    wire cs_end    =  cs_sync & ~cs_prev;

    reg [7:0] cmd_shift;
    reg       cmd_half;
    reg [7:0] active_cmd;
    reg       payload_half;
    reg [3:0] payload_hi;
    reg [4:0] payload_index;
    reg       receiving_payload;
    reg [2:0] load_slot;
    reg [2:0] fill_slot;

    reg       tx_active;
    reg       tx_phase;
    reg [3:0] tx_nibble;
    reg [3:0] tx_index;
    reg [1:0] tx_packet;
    reg [3:0] tx_len;

    reg [5:0]  ctx_voxel_x [0:4];
    reg [5:0]  ctx_voxel_y [0:4];
    reg [5:0]  ctx_voxel_z [0:4];
    reg        ctx_step_x_neg [0:4];
    reg        ctx_step_y_neg [0:4];
    reg        ctx_step_z_neg [0:4];
    reg [15:0] ctx_timer_x [0:4];
    reg [15:0] ctx_timer_y [0:4];
    reg [15:0] ctx_timer_z [0:4];
    reg [15:0] ctx_inc_x [0:4];
    reg [15:0] ctx_inc_y [0:4];
    reg [15:0] ctx_inc_z [0:4];
    reg [7:0]  ctx_max_steps [0:4];
    reg [7:0]  ctx_pixel_id [0:4];
    reg [7:0]  ctx_step_count [0:4];
    reg [2:0]  ctx_face_id [0:4];
    reg        ctx_valid [0:4];
    reg        ctx_needs_tile [0:4];

    reg        cache_valid [0:4];
    reg [3:0]  cache_tag_x [0:4];
    reg [3:0]  cache_tag_y [0:4];
    reg [3:0]  cache_tag_z [0:4];
    reg [63:0] cache_bits [0:4];

    reg [2:0] sched_ptr;
    reg       error_flag;
    reg       run_active;
    reg [8:0] run_budget;
    reg [2:0] stop_reason;

    reg [2:0] result_ctx [0:3];
    reg [7:0] result_status [0:3];
    reg [7:0] result_pixel_id [0:3];
    reg [5:0] result_x [0:3];
    reg [5:0] result_y [0:3];
    reg [5:0] result_z [0:3];
    reg [7:0] result_steps [0:3];
    reg [2:0] result_face [0:3];
    reg [1:0] result_wr_ptr;
    reg [1:0] result_rd_ptr;
    reg [2:0] result_count;

    reg free_found;
    reg [2:0] free_id;
    reg request_found;
    reg [2:0] request_id;
    reg ready_found;
    reg [2:0] ready_id;
    reg any_active;

    wire result_full = (result_count == 3'd4);

    wire [7:0] status_byte = {
        error_flag,
        tx_active,
        ready_found,
        !free_found,
        (result_count != 3'd0),
        request_found,
        free_found,
        any_active
    };

    wire qspi_rx_event = sck_rise && !tx_active;

    always @* begin
        integer i;
        free_found = 1'b0;
        free_id = 3'd0;
        request_found = 1'b0;
        request_id = 3'd0;
        any_active = 1'b0;

        for (i = 0; i < 5; i = i + 1) begin
            if (ctx_valid[i]) begin
                any_active = 1'b1;
            end
            if (!free_found && !ctx_valid[i]) begin
                free_found = 1'b1;
                free_id = i;
            end
            if (!request_found && ctx_valid[i] && ctx_needs_tile[i]) begin
                request_found = 1'b1;
                request_id = i;
            end
        end
    end

    always @* begin
        integer j;
        reg [2:0] probe;
        ready_found = 1'b0;
        ready_id = sched_ptr;
        for (j = 0; j < 5; j = j + 1) begin
            probe = sched_ptr + j;
            if (probe >= NUM_CONTEXTS) begin
                probe = probe - NUM_CONTEXTS;
            end
            if (!ready_found && ctx_valid[probe] && !ctx_needs_tile[probe] && !result_full) begin
                ready_found = 1'b1;
                ready_id = probe;
            end
        end
    end

    task clear_context;
        input [2:0] idx;
        begin
            ctx_voxel_x[idx]        <= 6'd0;
            ctx_voxel_y[idx]        <= 6'd0;
            ctx_voxel_z[idx]        <= 6'd0;
            ctx_step_x_neg[idx]     <= 1'b0;
            ctx_step_y_neg[idx]     <= 1'b0;
            ctx_step_z_neg[idx]     <= 1'b0;
            ctx_timer_x[idx]        <= 16'd0;
            ctx_timer_y[idx]        <= 16'd0;
            ctx_timer_z[idx]        <= 16'd0;
            ctx_inc_x[idx]          <= 16'd0;
            ctx_inc_y[idx]          <= 16'd0;
            ctx_inc_z[idx]          <= 16'd0;
            ctx_max_steps[idx]      <= 8'd0;
            ctx_pixel_id[idx]       <= 8'd0;
            ctx_step_count[idx]     <= 8'd0;
            ctx_face_id[idx]        <= 3'd0;
            ctx_valid[idx]          <= 1'b0;
            ctx_needs_tile[idx]     <= 1'b0;
        end
    endtask

    task clear_cache;
        input [2:0] idx;
        begin
            cache_valid[idx] <= 1'b0;
            cache_tag_x[idx] <= 4'd0;
            cache_tag_y[idx] <= 4'd0;
            cache_tag_z[idx] <= 4'd0;
            cache_bits[idx]  <= 64'd0;
        end
    endtask

    task clear_engine;
        integer k;
        begin
            for (k = 0; k < 5; k = k + 1) begin
                clear_context(k);
                clear_cache(k);
            end
            for (k = 0; k < 4; k = k + 1) begin
                result_ctx[k]      <= 3'd0;
                result_status[k]   <= 8'd0;
                result_pixel_id[k] <= 8'd0;
                result_x[k]        <= 6'd0;
                result_y[k]        <= 6'd0;
                result_z[k]        <= 6'd0;
                result_steps[k]    <= 8'd0;
                result_face[k]     <= 3'd0;
            end
            result_wr_ptr <= 2'd0;
            result_rd_ptr <= 2'd0;
            result_count  <= 3'd0;
            sched_ptr     <= 3'd0;
            error_flag    <= 1'b0;
            run_active    <= 1'b0;
            run_budget    <= 9'd0;
            stop_reason   <= 3'd0;
        end
    endtask

    task push_result;
        input [2:0] idx;
        input       hit;
        input       timeout;
        input [5:0] x_value;
        input [5:0] y_value;
        input [5:0] z_value;
        input [2:0] face_value;
        begin
            if (!result_full) begin
                result_ctx[result_wr_ptr]      <= idx;
                result_status[result_wr_ptr]   <= {6'b000000, timeout, hit};
                result_pixel_id[result_wr_ptr] <= ctx_pixel_id[idx];
                result_x[result_wr_ptr]        <= x_value;
                result_y[result_wr_ptr]        <= y_value;
                result_z[result_wr_ptr]        <= z_value;
                result_steps[result_wr_ptr]    <= ctx_step_count[idx] + 8'd1;
                result_face[result_wr_ptr]     <= face_value;
                result_wr_ptr                  <= result_wr_ptr + 2'd1;
                result_count                   <= result_count + 3'd1;
                clear_context(idx);
            end else begin
                error_flag <= 1'b1;
            end
        end
    endtask

    task pop_result;
        begin
            if (result_count != 3'd0) begin
                result_rd_ptr <= result_rd_ptr + 2'd1;
                result_count  <= result_count - 3'd1;
            end else begin
                error_flag <= 1'b1;
            end
        end
    endtask

    task set_tx_nibble;
        input [1:0] packet;
        input [3:0] index;
        input       high_half;
        reg [7:0] value;
        begin
            value = 8'h00;
            case (packet)
                2'd0: begin
                    case (index)
                        4'd0: value = status_byte;
                        4'd1: value = free_found ? {5'b00000, free_id} : 8'hff;
                        4'd2: value = request_found ? {5'b00000, request_id} : 8'hff;
                        4'd3: value = {5'b00000, result_count};
                        4'd4: value = {1'b0, stop_reason, 3'b000, run_active};
                        default: value = 8'h00;
                    endcase
                end
                2'd1: begin
                    case (index)
                        4'd0: value = request_found ? {5'b00000, request_id} : 8'hff;
                        4'd1: value = request_found ? {4'b0000, ctx_voxel_x[request_id][5:2]} : 8'h00;
                        4'd2: value = request_found ? {4'b0000, ctx_voxel_y[request_id][5:2]} : 8'h00;
                        4'd3: value = request_found ? {4'b0000, ctx_voxel_z[request_id][5:2]} : 8'h00;
                        default: value = 8'h00;
                    endcase
                end
                default: begin
                    case (index)
                        4'd0: value = (result_count != 3'd0) ? {5'b00000, result_ctx[result_rd_ptr]} : 8'hff;
                        4'd1: value = (result_count != 3'd0) ? result_status[result_rd_ptr] : 8'h00;
                        4'd2: value = (result_count != 3'd0) ? result_pixel_id[result_rd_ptr] : 8'h00;
                        4'd3: value = (result_count != 3'd0) ? {2'b00, result_x[result_rd_ptr]} : 8'h00;
                        4'd4: value = (result_count != 3'd0) ? {2'b00, result_y[result_rd_ptr]} : 8'h00;
                        4'd5: value = (result_count != 3'd0) ? {2'b00, result_z[result_rd_ptr]} : 8'h00;
                        4'd6: value = (result_count != 3'd0) ? result_steps[result_rd_ptr] : 8'h00;
                        4'd7: value = (result_count != 3'd0) ? {5'b00000, result_face[result_rd_ptr]} : 8'h00;
                        default: value = 8'h00;
                    endcase
                end
            endcase
            tx_nibble <= high_half ? value[7:4] : value[3:0];
        end
    endtask

    task start_tx;
        input [1:0] packet;
        input [3:0] length;
        begin
            tx_active <= 1'b1;
            tx_phase  <= 1'b0;
            tx_index  <= 4'd0;
            tx_packet <= packet;
            tx_len    <= length;
            set_tx_nibble(packet, 4'd0, 1'b1);
        end
    endtask

    task accept_context_byte;
        input [2:0] idx;
        input [4:0] index;
        input [7:0] value;
        begin
            case (index)
                5'd0:  ctx_voxel_x[idx]    <= value[5:0];
                5'd1:  ctx_voxel_y[idx]    <= value[5:0];
                5'd2:  ctx_voxel_z[idx]    <= value[5:0];
                5'd3:  begin
                    ctx_step_x_neg[idx] <= value[0];
                    ctx_step_y_neg[idx] <= value[1];
                    ctx_step_z_neg[idx] <= value[2];
                end
                5'd4:  ctx_timer_x[idx][15:8] <= value;
                5'd5:  ctx_timer_x[idx][7:0]  <= value;
                5'd6:  ctx_timer_y[idx][15:8] <= value;
                5'd7:  ctx_timer_y[idx][7:0]  <= value;
                5'd8:  ctx_timer_z[idx][15:8] <= value;
                5'd9:  ctx_timer_z[idx][7:0]  <= value;
                5'd10: ctx_inc_x[idx][15:8]   <= value;
                5'd11: ctx_inc_x[idx][7:0]    <= value;
                5'd12: ctx_inc_y[idx][15:8]   <= value;
                5'd13: ctx_inc_y[idx][7:0]    <= value;
                5'd14: ctx_inc_z[idx][15:8]   <= value;
                5'd15: ctx_inc_z[idx][7:0]    <= value;
                5'd16: ctx_max_steps[idx]     <= value;
                5'd17: begin
                    ctx_pixel_id[idx]       <= value;
                    ctx_step_count[idx]     <= 8'd0;
                    ctx_face_id[idx]        <= 3'd0;
                    ctx_valid[idx]          <= 1'b1;
                    ctx_needs_tile[idx]     <= 1'b1;
                end
                default: begin end
            endcase
        end
    endtask

    task accept_cache_byte;
        input [4:0] index;
        input [7:0] value;
        begin
            if (index == 5'd0) begin
                fill_slot <= value[2:0];
            end else if ((fill_slot < NUM_CONTEXTS) && ctx_valid[fill_slot]) begin
                case (index)
                    5'd1: cache_tag_x[fill_slot] <= value[3:0];
                    5'd2: cache_tag_y[fill_slot] <= value[3:0];
                    5'd3: cache_tag_z[fill_slot] <= value[3:0];
                    5'd4: cache_bits[fill_slot][63:56] <= value;
                    5'd5: cache_bits[fill_slot][55:48] <= value;
                    5'd6: cache_bits[fill_slot][47:40] <= value;
                    5'd7: cache_bits[fill_slot][39:32] <= value;
                    5'd8: cache_bits[fill_slot][31:24] <= value;
                    5'd9: cache_bits[fill_slot][23:16] <= value;
                    5'd10: cache_bits[fill_slot][15:8] <= value;
                    5'd11: begin
                        cache_bits[fill_slot][7:0] <= value;
                        cache_valid[fill_slot] <= 1'b1;
                        ctx_needs_tile[fill_slot] <= 1'b0;
                    end
                    default: begin end
                endcase
            end else begin
                error_flag <= 1'b1;
            end
        end
    endtask

    task run_step_for_ctx;
        input [2:0] idx;
        reg choose_x;
        reg choose_y;
        reg [5:0] next_x;
        reg [5:0] next_y;
        reg [5:0] next_z;
        reg [7:0] next_step;
        reg [2:0] next_face;
        reg [5:0] bit_index;
        reg       cache_hit;
        reg       cached_occupied;
        integer bank;
        begin
            if (ctx_valid[idx] && !ctx_needs_tile[idx] && !result_full) begin
                bit_index = {
                    ctx_voxel_z[idx][1:0],
                    ctx_voxel_y[idx][1:0],
                    ctx_voxel_x[idx][1:0]
                };
                cache_hit = 1'b0;
                cached_occupied = 1'b0;
                for (bank = 0; bank < 5; bank = bank + 1) begin
                    if (!cache_hit &&
                        cache_valid[bank] &&
                        (cache_tag_x[bank] == ctx_voxel_x[idx][5:2]) &&
                        (cache_tag_y[bank] == ctx_voxel_y[idx][5:2]) &&
                        (cache_tag_z[bank] == ctx_voxel_z[idx][5:2])) begin
                        cache_hit = 1'b1;
                        cached_occupied = cache_bits[bank][bit_index];
                    end
                end

                if (!cache_hit) begin
                    ctx_needs_tile[idx] <= 1'b1;
                    sched_ptr <= (idx == (NUM_CONTEXTS - 3'd1)) ? 3'd0 : (idx + 3'd1);
                end else begin
                next_step = ctx_step_count[idx] + 8'd1;
                if (cached_occupied) begin
                    push_result(idx, 1'b1, 1'b0, ctx_voxel_x[idx], ctx_voxel_y[idx], ctx_voxel_z[idx], ctx_face_id[idx]);
                end else if ((ctx_voxel_x[idx] > SCENE_MAX) ||
                             (ctx_voxel_y[idx] > SCENE_MAX) ||
                             (ctx_voxel_z[idx] > SCENE_MAX) ||
                             (next_step >= ctx_max_steps[idx])) begin
                    push_result(idx, 1'b0, 1'b1, ctx_voxel_x[idx], ctx_voxel_y[idx], ctx_voxel_z[idx], ctx_face_id[idx]);
                end else begin
                    choose_x = (ctx_timer_x[idx] <= ctx_timer_y[idx]) && (ctx_timer_x[idx] <= ctx_timer_z[idx]);
                    choose_y = (ctx_timer_y[idx] <  ctx_timer_x[idx]) && (ctx_timer_y[idx] <= ctx_timer_z[idx]);
                    next_x = ctx_voxel_x[idx];
                    next_y = ctx_voxel_y[idx];
                    next_z = ctx_voxel_z[idx];
                    next_face = ctx_face_id[idx];

                    if (choose_x) begin
                        next_x = ctx_step_x_neg[idx] ? (ctx_voxel_x[idx] - 6'd1) : (ctx_voxel_x[idx] + 6'd1);
                        ctx_timer_x[idx] <= ctx_timer_x[idx] + ctx_inc_x[idx];
                        next_face = ctx_step_x_neg[idx] ? 3'd1 : 3'd2;
                    end else if (choose_y) begin
                        next_y = ctx_step_y_neg[idx] ? (ctx_voxel_y[idx] - 6'd1) : (ctx_voxel_y[idx] + 6'd1);
                        ctx_timer_y[idx] <= ctx_timer_y[idx] + ctx_inc_y[idx];
                        next_face = ctx_step_y_neg[idx] ? 3'd3 : 3'd4;
                    end else begin
                        next_z = ctx_step_z_neg[idx] ? (ctx_voxel_z[idx] - 6'd1) : (ctx_voxel_z[idx] + 6'd1);
                        ctx_timer_z[idx] <= ctx_timer_z[idx] + ctx_inc_z[idx];
                        next_face = ctx_step_z_neg[idx] ? 3'd5 : 3'd6;
                    end

                    ctx_voxel_x[idx]     <= next_x;
                    ctx_voxel_y[idx]     <= next_y;
                    ctx_voxel_z[idx]     <= next_z;
                    ctx_step_count[idx]  <= next_step;
                    ctx_face_id[idx]     <= next_face;
                    ctx_needs_tile[idx]  <= 1'b0;

                    if ((next_x > SCENE_MAX) || (next_y > SCENE_MAX) || (next_z > SCENE_MAX)) begin
                        ctx_face_id[idx] <= next_face;
                        push_result(idx, 1'b0, 1'b1, next_x, next_y, next_z, next_face);
                    end
                end
                sched_ptr <= (idx == (NUM_CONTEXTS - 3'd1)) ? 3'd0 : (idx + 3'd1);
                end
            end else begin
                error_flag <= 1'b1;
            end
        end
    endtask

    always @(posedge clk) begin
        if (!rst_n) begin
            sck_meta          <= 1'b0;
            sck_sync          <= 1'b0;
            sck_prev          <= 1'b0;
            cs_meta           <= 1'b1;
            cs_sync           <= 1'b1;
            cs_prev           <= 1'b1;
            dq_meta           <= 4'd0;
            dq_sync           <= 4'd0;
            cmd_shift         <= 8'd0;
            cmd_half          <= 1'b0;
            active_cmd        <= 8'd0;
            payload_half      <= 1'b0;
            payload_hi        <= 4'd0;
            payload_index     <= 5'd0;
            receiving_payload <= 1'b0;
            load_slot         <= 3'd0;
            fill_slot         <= 3'd0;
            run_active        <= 1'b0;
            run_budget        <= 9'd0;
            stop_reason       <= 3'd0;
            tx_active         <= 1'b0;
            tx_phase          <= 1'b0;
            tx_nibble         <= 4'd0;
            tx_index          <= 4'd0;
            tx_packet         <= 2'd0;
            tx_len            <= 4'd0;
            clear_engine();
        end else begin
            sck_meta <= qspi_sck_in;
            sck_sync <= sck_meta;
            sck_prev <= sck_sync;
            cs_meta  <= qspi_cs_n_in;
            cs_sync  <= cs_meta;
            cs_prev  <= cs_sync;
            dq_meta  <= uio_in[3:0];
            dq_sync  <= dq_meta;

            if (cs_start || cs_end) begin
                cmd_half          <= 1'b0;
                payload_half      <= 1'b0;
                receiving_payload <= 1'b0;
                tx_active         <= 1'b0;
                tx_nibble         <= 4'd0;
            end

            if (qspi_rx_event) begin
                if (receiving_payload) begin
                    if (!payload_half) begin
                        payload_hi   <= dq_sync;
                        payload_half <= 1'b1;
                    end else begin
                        payload_half <= 1'b0;
                        if (active_cmd == CMD_WRITE_CONTEXT) begin
                            accept_context_byte(load_slot, payload_index, {payload_hi, dq_sync});
                            payload_index <= payload_index + 5'd1;
                            if (payload_index == (CONTEXT_BYTES - 5'd1)) begin
                                receiving_payload <= 1'b0;
                            end
                        end else if (active_cmd == CMD_FILL_CACHE) begin
                            accept_cache_byte(payload_index, {payload_hi, dq_sync});
                            payload_index <= payload_index + 5'd1;
                            if (payload_index == 5'd11) begin
                                receiving_payload <= 1'b0;
                            end
                        end else if (active_cmd == CMD_RUN_N) begin
                            run_budget <= ({payload_hi, dq_sync} == 8'd0) ? 9'd256 : {1'b0, payload_hi, dq_sync};
                            run_active <= 1'b1;
                            stop_reason <= 3'd0;
                            receiving_payload <= 1'b0;
                        end
                    end
                end else if (!cmd_half) begin
                    cmd_shift <= {dq_sync, 4'h0};
                    cmd_half  <= 1'b1;
                end else begin
                    active_cmd <= {cmd_shift[7:4], dq_sync};
                    cmd_half   <= 1'b0;
                    case ({cmd_shift[7:4], dq_sync})
                        CMD_RESET: begin
                            clear_engine();
                        end
                        CMD_WRITE_CONTEXT: begin
                            if (free_found) begin
                                load_slot <= free_id;
                                receiving_payload <= 1'b1;
                                payload_index <= 5'd0;
                                payload_half <= 1'b0;
                            end else begin
                                error_flag <= 1'b1;
                            end
                        end
                        CMD_FILL_CACHE: begin
                            receiving_payload <= 1'b1;
                            payload_index <= 5'd0;
                            payload_half <= 1'b0;
                        end
                        CMD_STEP: begin
                            if (ready_found) begin
                                run_step_for_ctx(ready_id);
                            end else begin
                                error_flag <= 1'b1;
                            end
                        end
                        CMD_RUN_N: begin
                            receiving_payload <= 1'b1;
                            payload_index <= 5'd0;
                            payload_half <= 1'b0;
                        end
                        CMD_POP_RESULT: begin
                            pop_result();
                        end
                        CMD_READ_STATUS: begin
                            start_tx(2'd0, 4'd5);
                        end
                        CMD_READ_REQUEST: begin
                            start_tx(2'd1, 4'd4);
                        end
                        CMD_READ_RESULT: begin
                            start_tx(2'd2, 4'd8);
                        end
                        default: begin
                            error_flag <= 1'b1;
                        end
                    endcase
                end
            end

            if (sck_fall && tx_active) begin
                if (!tx_phase) begin
                    set_tx_nibble(tx_packet, tx_index, 1'b0);
                    tx_phase <= 1'b1;
                end else begin
                    tx_phase <= 1'b0;
                    if ((tx_index + 4'd1) >= tx_len) begin
                        tx_active <= 1'b0;
                        tx_nibble <= 4'd0;
                    end else begin
                        tx_index <= tx_index + 4'd1;
                        set_tx_nibble(tx_packet, tx_index + 4'd1, 1'b1);
                    end
                end
            end

            if (run_active && cs_sync && !tx_active) begin
                if (result_full) begin
                    run_active <= 1'b0;
                    stop_reason <= 3'd1;
                end else if (!ready_found) begin
                    run_active <= 1'b0;
                    stop_reason <= request_found ? 3'd2 : 3'd3;
                end else if (run_budget == 9'd0) begin
                    run_active <= 1'b0;
                    stop_reason <= 3'd4;
                end else begin
                    run_step_for_ctx(ready_id);
                    run_budget <= run_budget - 9'd1;
                    if (run_budget == 9'd1) begin
                        run_active <= 1'b0;
                        stop_reason <= 3'd4;
                    end
                end
            end
        end
    end

    assign uio_out = {4'h0, tx_nibble};
    assign uio_oe  = {4'h0, {4{tx_active}}};
    assign uo_out  = ena ? status_byte : 8'h00;

endmodule

`default_nettype wire
