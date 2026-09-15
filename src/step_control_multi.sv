`default_nettype none

// =============================================================================
// Module: step_control_multi
// Five-context round-robin controller for the 5-stage DDA pipeline.
// =============================================================================
module step_control_multi #(
    parameter int X_BITS           = 5,
    parameter int Y_BITS           = 5,
    parameter int Z_BITS           = 5,
    parameter int TIMER_WIDTH      = 32,
    parameter int STEP_COUNT_WIDTH = 16,
    parameter int PIXEL_ID_WIDTH   = 14,
    parameter int RAY_ID_WIDTH     = 3,
    parameter int NUM_CONTEXTS     = 5,
    parameter int RESULT_DEPTH     = 8
)(
    input  wire logic                     clock,
    input  wire logic                     reset,

    input  wire logic                     load_mode,

    input  wire logic                     job_valid,
    output logic                          job_ready,
    input  wire logic [PIXEL_ID_WIDTH-1:0] job_pixel_id,
    input  wire logic [X_BITS-1:0]        job_init_x,
    input  wire logic [Y_BITS-1:0]        job_init_y,
    input  wire logic [Z_BITS-1:0]        job_init_z,
    input  wire logic                     job_sx,
    input  wire logic                     job_sy,
    input  wire logic                     job_sz,
    input  wire logic [TIMER_WIDTH-1:0]   job_timer_x,
    input  wire logic [TIMER_WIDTH-1:0]   job_timer_y,
    input  wire logic [TIMER_WIDTH-1:0]   job_timer_z,
    input  wire logic [TIMER_WIDTH-1:0]   job_inc_x,
    input  wire logic [TIMER_WIDTH-1:0]   job_inc_y,
    input  wire logic [TIMER_WIDTH-1:0]   job_inc_z,
    input  wire logic [STEP_COUNT_WIDTH-1:0] max_steps,

    input  wire logic                     pipeline_valid,
    input  wire logic [RAY_ID_WIDTH-1:0]  pipeline_ray_id,
    input  wire logic                     solid_bit,
    input  wire logic                     out_of_bounds,
    input  wire logic [X_BITS-1:0]        pipeline_curr_x,
    input  wire logic [Y_BITS-1:0]        pipeline_curr_y,
    input  wire logic [Z_BITS-1:0]        pipeline_curr_z,
    input  wire logic [X_BITS-1:0]        pipeline_next_x,
    input  wire logic [Y_BITS-1:0]        pipeline_next_y,
    input  wire logic [Z_BITS-1:0]        pipeline_next_z,
    input  wire logic [TIMER_WIDTH-1:0]   pipeline_next_timer_x,
    input  wire logic [TIMER_WIDTH-1:0]   pipeline_next_timer_y,
    input  wire logic [TIMER_WIDTH-1:0]   pipeline_next_timer_z,
    input  wire logic [2:0]               pipeline_face_id,

    output logic                          issue_valid,
    output logic [RAY_ID_WIDTH-1:0]       issue_ray_id,
    output logic [X_BITS-1:0]             issue_voxel_x,
    output logic [Y_BITS-1:0]             issue_voxel_y,
    output logic [Z_BITS-1:0]             issue_voxel_z,
    output logic                          issue_sx,
    output logic                          issue_sy,
    output logic                          issue_sz,
    output logic [TIMER_WIDTH-1:0]        issue_timer_x,
    output logic [TIMER_WIDTH-1:0]        issue_timer_y,
    output logic [TIMER_WIDTH-1:0]        issue_timer_z,
    output logic [TIMER_WIDTH-1:0]        issue_inc_x,
    output logic [TIMER_WIDTH-1:0]        issue_inc_y,
    output logic [TIMER_WIDTH-1:0]        issue_inc_z,

    output logic                          result_valid,
    input  wire logic                     result_ready,
    output logic [PIXEL_ID_WIDTH-1:0]     result_pixel_id,
    output logic                          result_hit,
    output logic                          result_timeout,
    output logic [X_BITS-1:0]             result_hit_x,
    output logic [Y_BITS-1:0]             result_hit_y,
    output logic [Z_BITS-1:0]             result_hit_z,
    output logic [2:0]                    result_face_id,
    output logic [STEP_COUNT_WIDTH-1:0]   result_steps,
    output logic                          all_idle
);

    localparam int CTX_W = (NUM_CONTEXTS <= 2) ? 1 :
                           (NUM_CONTEXTS <= 4) ? 2 : 3;
    localparam int FIFO_PTR_W = (RESULT_DEPTH <= 2) ? 1 :
                                (RESULT_DEPTH <= 4) ? 2 :
                                (RESULT_DEPTH <= 8) ? 3 : 4;

    logic [X_BITS-1:0]           voxel_x [0:NUM_CONTEXTS-1];
    logic [Y_BITS-1:0]           voxel_y [0:NUM_CONTEXTS-1];
    logic [Z_BITS-1:0]           voxel_z [0:NUM_CONTEXTS-1];
    logic [TIMER_WIDTH-1:0]      timer_x [0:NUM_CONTEXTS-1];
    logic [TIMER_WIDTH-1:0]      timer_y [0:NUM_CONTEXTS-1];
    logic [TIMER_WIDTH-1:0]      timer_z [0:NUM_CONTEXTS-1];
    logic [TIMER_WIDTH-1:0]      inc_x   [0:NUM_CONTEXTS-1];
    logic [TIMER_WIDTH-1:0]      inc_y   [0:NUM_CONTEXTS-1];
    logic [TIMER_WIDTH-1:0]      inc_z   [0:NUM_CONTEXTS-1];
    logic [STEP_COUNT_WIDTH-1:0] step_ctr [0:NUM_CONTEXTS-1];
    logic [STEP_COUNT_WIDTH-1:0] max_steps_reg [0:NUM_CONTEXTS-1];
    logic [PIXEL_ID_WIDTH-1:0]   pixel_id [0:NUM_CONTEXTS-1];
    logic [2:0]                  face_reg [0:NUM_CONTEXTS-1];
    logic                        sx_reg   [0:NUM_CONTEXTS-1];
    logic                        sy_reg   [0:NUM_CONTEXTS-1];
    logic                        sz_reg   [0:NUM_CONTEXTS-1];
    logic                        ctx_active [0:NUM_CONTEXTS-1];
    logic                        ctx_in_pipe [0:NUM_CONTEXTS-1];
    logic                        ctx_pending [0:NUM_CONTEXTS-1];
    logic                        pend_hit [0:NUM_CONTEXTS-1];
    logic                        pend_timeout [0:NUM_CONTEXTS-1];
    logic [X_BITS-1:0]           pend_hit_x [0:NUM_CONTEXTS-1];
    logic [Y_BITS-1:0]           pend_hit_y [0:NUM_CONTEXTS-1];
    logic [Z_BITS-1:0]           pend_hit_z [0:NUM_CONTEXTS-1];
    logic [2:0]                  pend_face [0:NUM_CONTEXTS-1];

    logic [CTX_W-1:0] issue_ptr;
    logic [CTX_W-1:0] free_idx;
    logic             free_found;

    logic [PIXEL_ID_WIDTH-1:0]   fifo_pixel [0:RESULT_DEPTH-1];
    logic                        fifo_hit [0:RESULT_DEPTH-1];
    logic                        fifo_timeout [0:RESULT_DEPTH-1];
    logic [X_BITS-1:0]           fifo_hit_x [0:RESULT_DEPTH-1];
    logic [Y_BITS-1:0]           fifo_hit_y [0:RESULT_DEPTH-1];
    logic [Z_BITS-1:0]           fifo_hit_z [0:RESULT_DEPTH-1];
    logic [2:0]                  fifo_face [0:RESULT_DEPTH-1];
    logic [STEP_COUNT_WIDTH-1:0] fifo_steps [0:RESULT_DEPTH-1];
    logic [FIFO_PTR_W-1:0]       fifo_wr_ptr, fifo_rd_ptr;
    logic [FIFO_PTR_W:0]         fifo_count;
    logic                        fifo_full;
    logic                        fifo_pop;

    assign fifo_full    = (fifo_count == RESULT_DEPTH[FIFO_PTR_W:0]);
    assign result_valid = (fifo_count != '0);
    assign fifo_pop     = result_valid && result_ready;

    assign result_pixel_id = fifo_pixel[fifo_rd_ptr];
    assign result_hit      = fifo_hit[fifo_rd_ptr];
    assign result_timeout  = fifo_timeout[fifo_rd_ptr];
    assign result_hit_x    = fifo_hit_x[fifo_rd_ptr];
    assign result_hit_y    = fifo_hit_y[fifo_rd_ptr];
    assign result_hit_z    = fifo_hit_z[fifo_rd_ptr];
    // result_face_id is 0..5 (a real face) or 3'd6 -- sentinel meaning the
    // ray hit its starting voxel with no traversal, so no entry face exists.
    assign result_face_id  = fifo_face[fifo_rd_ptr];
    assign result_steps    = fifo_steps[fifo_rd_ptr];

    always_comb begin
        free_found = 1'b0;
        free_idx   = '0;
        for (int i = 0; i < NUM_CONTEXTS; i = i + 1) begin
            if (!free_found && !ctx_active[i] && !ctx_in_pipe[i] && !ctx_pending[i]) begin
                free_found = 1'b1;
                free_idx   = i;
            end
        end
    end

    assign job_ready = (!load_mode) && free_found;

    always_comb begin
        all_idle = (fifo_count == '0);
        for (int i = 0; i < NUM_CONTEXTS; i = i + 1) begin
            if (ctx_active[i] || ctx_in_pipe[i] || ctx_pending[i])
                all_idle = 1'b0;
        end
    end

    always_comb begin
        issue_valid   = 1'b0;
        issue_ray_id  = '0;
        issue_ray_id[CTX_W-1:0] = issue_ptr;
        issue_voxel_x = voxel_x[issue_ptr];
        issue_voxel_y = voxel_y[issue_ptr];
        issue_voxel_z = voxel_z[issue_ptr];
        issue_sx      = sx_reg[issue_ptr];
        issue_sy      = sy_reg[issue_ptr];
        issue_sz      = sz_reg[issue_ptr];
        issue_timer_x = timer_x[issue_ptr];
        issue_timer_y = timer_y[issue_ptr];
        issue_timer_z = timer_z[issue_ptr];
        issue_inc_x   = inc_x[issue_ptr];
        issue_inc_y   = inc_y[issue_ptr];
        issue_inc_z   = inc_z[issue_ptr];
        if (!load_mode && ctx_active[issue_ptr] && !ctx_in_pipe[issue_ptr] && !ctx_pending[issue_ptr])
            issue_valid = 1'b1;
    end

    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            issue_ptr   <= '0;
            fifo_wr_ptr <= '0;
            fifo_rd_ptr <= '0;
            fifo_count  <= '0;
            for (int i = 0; i < NUM_CONTEXTS; i = i + 1) begin
                voxel_x[i] <= '0; voxel_y[i] <= '0; voxel_z[i] <= '0;
                timer_x[i] <= '0; timer_y[i] <= '0; timer_z[i] <= '0;
                inc_x[i] <= '0; inc_y[i] <= '0; inc_z[i] <= '0;
                step_ctr[i] <= '0; max_steps_reg[i] <= '0; pixel_id[i] <= '0;
                face_reg[i] <= 3'd6;
                sx_reg[i] <= 1'b0; sy_reg[i] <= 1'b0; sz_reg[i] <= 1'b0;
                ctx_active[i] <= 1'b0;
                ctx_in_pipe[i] <= 1'b0;
                ctx_pending[i] <= 1'b0;
                pend_hit[i] <= 1'b0; pend_timeout[i] <= 1'b0;
                pend_hit_x[i] <= '0; pend_hit_y[i] <= '0; pend_hit_z[i] <= '0;
                pend_face[i] <= 3'd6;
            end
        end else begin
            logic fifo_push;
            logic [CTX_W-1:0] push_ctx;
            logic push_hit;
            logic push_timeout;
            logic [X_BITS-1:0] push_hit_x;
            logic [Y_BITS-1:0] push_hit_y;
            logic [Z_BITS-1:0] push_hit_z;
            logic [2:0] push_face;
            logic [STEP_COUNT_WIDTH-1:0] next_step;
            logic finish_now;

            fifo_push    = 1'b0;
            push_ctx     = '0;
            push_hit     = 1'b0;
            push_timeout = 1'b0;
            push_hit_x   = '0;
            push_hit_y   = '0;
            push_hit_z   = '0;
            push_face    = 3'd6;

            if (issue_ptr == (NUM_CONTEXTS-1))
                issue_ptr <= '0;
            else
                issue_ptr <= issue_ptr + 1'b1;

            if (issue_valid)
                ctx_in_pipe[issue_ptr] <= 1'b1;

            if (job_valid && job_ready) begin
                ctx_active[free_idx] <= 1'b1;
                voxel_x[free_idx] <= job_init_x;
                voxel_y[free_idx] <= job_init_y;
                voxel_z[free_idx] <= job_init_z;
                timer_x[free_idx] <= job_timer_x;
                timer_y[free_idx] <= job_timer_y;
                timer_z[free_idx] <= job_timer_z;
                inc_x[free_idx] <= job_inc_x;
                inc_y[free_idx] <= job_inc_y;
                inc_z[free_idx] <= job_inc_z;
                sx_reg[free_idx] <= job_sx;
                sy_reg[free_idx] <= job_sy;
                sz_reg[free_idx] <= job_sz;
                max_steps_reg[free_idx] <= max_steps;
                step_ctr[free_idx] <= '0;
                pixel_id[free_idx] <= job_pixel_id;
                // Sentinel: no voxel has been entered yet, so if this ray
                // hits its own starting voxel there is no entry face.
                face_reg[free_idx] <= 3'd6;
            end

            if (pipeline_valid) begin
                ctx_in_pipe[pipeline_ray_id[CTX_W-1:0]] <= 1'b0;
                next_step = step_ctr[pipeline_ray_id[CTX_W-1:0]] + 1'b1;
                finish_now = solid_bit || out_of_bounds ||
                             (next_step >= max_steps_reg[pipeline_ray_id[CTX_W-1:0]]);

                if (finish_now) begin
                    // pipeline_face_id here describes the step OUT of the
                    // voxel under test (into whatever voxel comes next) --
                    // that step is never taken because the ray terminates
                    // this pass. The face the ray actually entered THIS
                    // (hit) voxel through is whatever the previous,
                    // non-terminating pass latched into face_reg below, so
                    // that old value -- not pipeline_face_id -- is the
                    // correct entry face to report.
                    ctx_active[pipeline_ray_id[CTX_W-1:0]] <= 1'b0;
                    ctx_pending[pipeline_ray_id[CTX_W-1:0]] <= 1'b1;
                    pend_hit[pipeline_ray_id[CTX_W-1:0]] <= solid_bit;
                    pend_timeout[pipeline_ray_id[CTX_W-1:0]] <= (!solid_bit && !out_of_bounds);
                    pend_hit_x[pipeline_ray_id[CTX_W-1:0]] <= pipeline_curr_x;
                    pend_hit_y[pipeline_ray_id[CTX_W-1:0]] <= pipeline_curr_y;
                    pend_hit_z[pipeline_ray_id[CTX_W-1:0]] <= pipeline_curr_z;
                    pend_face[pipeline_ray_id[CTX_W-1:0]] <= face_reg[pipeline_ray_id[CTX_W-1:0]];
                    step_ctr[pipeline_ray_id[CTX_W-1:0]] <= next_step;
                end else begin
                    voxel_x[pipeline_ray_id[CTX_W-1:0]] <= pipeline_next_x;
                    voxel_y[pipeline_ray_id[CTX_W-1:0]] <= pipeline_next_y;
                    voxel_z[pipeline_ray_id[CTX_W-1:0]] <= pipeline_next_z;
                    timer_x[pipeline_ray_id[CTX_W-1:0]] <= pipeline_next_timer_x;
                    timer_y[pipeline_ray_id[CTX_W-1:0]] <= pipeline_next_timer_y;
                    timer_z[pipeline_ray_id[CTX_W-1:0]] <= pipeline_next_timer_z;
                    step_ctr[pipeline_ray_id[CTX_W-1:0]] <= next_step;
                    // Latch the face this step crosses INTO. If the next
                    // voxel tested turns out solid, this value becomes its
                    // (correct) entry face above.
                    face_reg[pipeline_ray_id[CTX_W-1:0]] <= pipeline_face_id;
                end
            end

            for (int i = 0; i < NUM_CONTEXTS; i = i + 1) begin
                if (!fifo_push && ctx_pending[i] && !fifo_full) begin
                    fifo_push = 1'b1;
                    push_ctx = i;
                    push_hit = pend_hit[i];
                    push_timeout = pend_timeout[i];
                    push_hit_x = pend_hit_x[i];
                    push_hit_y = pend_hit_y[i];
                    push_hit_z = pend_hit_z[i];
                    push_face = pend_face[i];
                end
            end

            if (fifo_push) begin
                fifo_pixel[fifo_wr_ptr] <= pixel_id[push_ctx];
                fifo_hit[fifo_wr_ptr] <= push_hit;
                fifo_timeout[fifo_wr_ptr] <= push_timeout;
                fifo_hit_x[fifo_wr_ptr] <= push_hit_x;
                fifo_hit_y[fifo_wr_ptr] <= push_hit_y;
                fifo_hit_z[fifo_wr_ptr] <= push_hit_z;
                fifo_face[fifo_wr_ptr] <= push_face;
                fifo_steps[fifo_wr_ptr] <= step_ctr[push_ctx];
                fifo_wr_ptr <= fifo_wr_ptr + 1'b1;
                ctx_pending[push_ctx] <= 1'b0;
            end

            if (fifo_pop)
                fifo_rd_ptr <= fifo_rd_ptr + 1'b1;

            case ({fifo_push, fifo_pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
