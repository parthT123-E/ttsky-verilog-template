`default_nettype none

// =============================================================================
// Module: raytracer_top  (synthesis-safe copy — timescale removed)
// Full DDA raytracer, integrating ray_job_if + step_control_fsm + voxel_raytracer_core.
// =============================================================================
module raytracer_top #(
    parameter int COORD_WIDTH     = 16,
    parameter int COORD_W         = 6,
    parameter int TIMER_WIDTH     = 32,
    parameter int W               = 32,
    parameter int MAX_VAL         = 31,
    parameter int ADDR_BITS       = 15,
    parameter int WORD_BITS       = 32,   // voxel RAM data width
    parameter int X_BITS          = 6,
    parameter int Y_BITS          = 6,
    parameter int Z_BITS          = 6,
    parameter int MAX_STEPS_BITS  = 10,
    parameter int STEP_COUNT_WIDTH = 16,
    parameter int PIXEL_ID_WIDTH   = 14,
    parameter int RAY_ID_WIDTH     = 3
)(
    input  wire logic                     clk,
    input  wire logic                     rst_n,

    // Ray-job input
    input  wire logic                     job_valid,
    output logic                          job_ready,
    input  wire logic [X_BITS-1:0]        job_ix0,
    input  wire logic [Y_BITS-1:0]        job_iy0,
    input  wire logic [Z_BITS-1:0]        job_iz0,
    input  wire logic                     job_sx,
    input  wire logic                     job_sy,
    input  wire logic                     job_sz,
    input  wire logic [W-1:0]             job_next_x,
    input  wire logic [W-1:0]             job_next_y,
    input  wire logic [W-1:0]             job_next_z,
    input  wire logic [W-1:0]             job_inc_x,
    input  wire logic [W-1:0]             job_inc_y,
    input  wire logic [W-1:0]             job_inc_z,
    input  wire logic [MAX_STEPS_BITS-1:0] job_max_steps,
    input  wire logic [PIXEL_ID_WIDTH-1:0] job_pixel_id,

    // Scene loading
    input  wire logic                     load_mode,
    input  wire logic                     load_valid,
    output logic                          load_ready,
    input  wire logic [ADDR_BITS-$clog2(WORD_BITS)-1:0] load_addr,
    input  wire logic [WORD_BITS-1:0]     load_data,
    output logic [ADDR_BITS:0]            write_count,
    output logic                          load_complete,

    // Results
    output logic                          ray_done,
    output logic                          ray_hit,
    output logic                          ray_timeout,
    output logic [COORD_WIDTH-1:0]        hit_voxel_x,
    output logic [COORD_WIDTH-1:0]        hit_voxel_y,
    output logic [COORD_WIDTH-1:0]        hit_voxel_z,
    output logic [2:0]                    hit_face_id,
    output logic [STEP_COUNT_WIDTH-1:0]   steps_taken,

    // Streaming results for the interleaved ray engine
    output logic                          result_valid,
    input  wire logic                     result_ready,
    output logic [PIXEL_ID_WIDTH-1:0]     result_pixel_id,
    output logic                          result_hit,
    output logic                          result_timeout,
    output logic [COORD_WIDTH-1:0]        result_hit_voxel_x,
    output logic [COORD_WIDTH-1:0]        result_hit_voxel_y,
    output logic [COORD_WIDTH-1:0]        result_hit_voxel_z,
    output logic [2:0]                    result_face_id,
    output logic [STEP_COUNT_WIDTH-1:0]   result_steps,
    output logic                          tracer_idle
);

    // Internal signals
    logic                   issue_valid;
    logic [RAY_ID_WIDTH-1:0] issue_ray_id;
    logic [X_BITS-1:0]      current_ix, current_iy, current_iz;
    logic [W-1:0]           current_next_x, current_next_y, current_next_z;
    logic                   current_sx, current_sy, current_sz;
    logic [W-1:0]           current_inc_x, current_inc_y, current_inc_z;

    logic [X_BITS-1:0]      next_ix, next_iy, next_iz;
    logic [X_BITS-1:0]      curr_ix_out, curr_iy_out, curr_iz_out;
    logic [W-1:0]           next_next_x, next_next_y, next_next_z;
    logic [2:0]             face_mask, primary_face_id;
    logic                   out_of_bounds, voxel_occupied, step_valid_out;
    logic [RAY_ID_WIDTH-1:0] result_ray_id;
    logic                   core_load_ready;
    logic [X_BITS-1:0]      result_hit_x_raw;
    logic [Y_BITS-1:0]      result_hit_y_raw;
    logic [Z_BITS-1:0]      result_hit_z_raw;

    // -------------------------------------------------------------------
    step_control_multi #(
        .X_BITS(X_BITS), .Y_BITS(Y_BITS), .Z_BITS(Z_BITS),
        .TIMER_WIDTH(TIMER_WIDTH), .STEP_COUNT_WIDTH(STEP_COUNT_WIDTH),
        .PIXEL_ID_WIDTH(PIXEL_ID_WIDTH), .RAY_ID_WIDTH(RAY_ID_WIDTH),
        .NUM_CONTEXTS(5), .RESULT_DEPTH(8)
    ) u_step_control_multi (
        .clock(clk), .reset(~rst_n),
        .load_mode(load_mode),
        .job_valid(job_valid), .job_ready(job_ready),
        .job_pixel_id(job_pixel_id),
        .job_init_x(job_ix0), .job_init_y(job_iy0), .job_init_z(job_iz0),
        .job_sx(job_sx), .job_sy(job_sy), .job_sz(job_sz),
        .job_timer_x({{(TIMER_WIDTH-W){1'b0}}, job_next_x}),
        .job_timer_y({{(TIMER_WIDTH-W){1'b0}}, job_next_y}),
        .job_timer_z({{(TIMER_WIDTH-W){1'b0}}, job_next_z}),
        .job_inc_x({{(TIMER_WIDTH-W){1'b0}}, job_inc_x}),
        .job_inc_y({{(TIMER_WIDTH-W){1'b0}}, job_inc_y}),
        .job_inc_z({{(TIMER_WIDTH-W){1'b0}}, job_inc_z}),
        .max_steps({{(STEP_COUNT_WIDTH-MAX_STEPS_BITS){1'b0}}, job_max_steps}),
        .pipeline_valid(step_valid_out),
        .pipeline_ray_id(result_ray_id),
        .solid_bit(voxel_occupied),
        .out_of_bounds(out_of_bounds),
        .pipeline_curr_x(curr_ix_out),
        .pipeline_curr_y(curr_iy_out),
        .pipeline_curr_z(curr_iz_out),
        .pipeline_next_x(next_ix),
        .pipeline_next_y(next_iy),
        .pipeline_next_z(next_iz),
        .pipeline_next_timer_x({{(TIMER_WIDTH-W){1'b0}}, next_next_x}),
        .pipeline_next_timer_y({{(TIMER_WIDTH-W){1'b0}}, next_next_y}),
        .pipeline_next_timer_z({{(TIMER_WIDTH-W){1'b0}}, next_next_z}),
        .pipeline_face_id(primary_face_id),
        .issue_valid(issue_valid),
        .issue_ray_id(issue_ray_id),
        .issue_voxel_x(current_ix),
        .issue_voxel_y(current_iy),
        .issue_voxel_z(current_iz),
        .issue_sx(current_sx),
        .issue_sy(current_sy),
        .issue_sz(current_sz),
        .issue_timer_x(current_next_x),
        .issue_timer_y(current_next_y),
        .issue_timer_z(current_next_z),
        .issue_inc_x(current_inc_x),
        .issue_inc_y(current_inc_y),
        .issue_inc_z(current_inc_z),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_pixel_id(result_pixel_id),
        .result_hit(result_hit),
        .result_timeout(result_timeout),
        .result_hit_x(result_hit_x_raw),
        .result_hit_y(result_hit_y_raw),
        .result_hit_z(result_hit_z_raw),
        .result_face_id(result_face_id),
        .result_steps(result_steps),
        .all_idle(tracer_idle)
    );

    assign ray_done    = result_valid;
    assign ray_hit     = result_hit;
    assign ray_timeout = result_timeout;
    assign hit_face_id = result_face_id;
    assign steps_taken = result_steps;
    assign hit_voxel_x[X_BITS-1:0] = result_hit_x_raw;
    assign hit_voxel_y[Y_BITS-1:0] = result_hit_y_raw;
    assign hit_voxel_z[Z_BITS-1:0] = result_hit_z_raw;
    assign result_hit_voxel_x[X_BITS-1:0] = result_hit_x_raw;
    assign result_hit_voxel_y[Y_BITS-1:0] = result_hit_y_raw;
    assign result_hit_voxel_z[Z_BITS-1:0] = result_hit_z_raw;

    // Zero-extend upper coordinate bits if COORD_WIDTH > X_BITS
    generate
        if (COORD_WIDTH > X_BITS) begin : g_ext
            assign hit_voxel_x[COORD_WIDTH-1:X_BITS] = '0;
            assign hit_voxel_y[COORD_WIDTH-1:Y_BITS] = '0;
            assign hit_voxel_z[COORD_WIDTH-1:Z_BITS] = '0;
            assign result_hit_voxel_x[COORD_WIDTH-1:X_BITS] = '0;
            assign result_hit_voxel_y[COORD_WIDTH-1:Y_BITS] = '0;
            assign result_hit_voxel_z[COORD_WIDTH-1:Z_BITS] = '0;
        end
    endgenerate

    // -------------------------------------------------------------------
    assign load_ready = core_load_ready && tracer_idle;

    voxel_raytracer_core #(
        .W(W), .COORD_W(COORD_W), .MAX_VAL(MAX_VAL), .ADDR_BITS(ADDR_BITS),
        .WORD_BITS(WORD_BITS), .RAY_ID_WIDTH(RAY_ID_WIDTH)
    ) u_core (
        .clk(clk), .rst_n(rst_n),
        .ix_in(current_ix), .iy_in(current_iy), .iz_in(current_iz),
        .sx_in(current_sx), .sy_in(current_sy), .sz_in(current_sz),
        .next_x_in(current_next_x[W-1:0]),
        .next_y_in(current_next_y[W-1:0]),
        .next_z_in(current_next_z[W-1:0]),
        .inc_x_in(current_inc_x[W-1:0]), .inc_y_in(current_inc_y[W-1:0]), .inc_z_in(current_inc_z[W-1:0]),
        .ray_id_in(issue_ray_id),
        .step_valid_in(issue_valid),
        .load_mode(load_mode && tracer_idle), .load_valid(load_valid && tracer_idle),
        .load_ready(core_load_ready), .load_addr(load_addr), .load_data(load_data),
        .write_count(write_count), .load_complete(load_complete),
        .ix_out(next_ix), .iy_out(next_iy), .iz_out(next_iz),
        .ix_curr_out(curr_ix_out), .iy_curr_out(curr_iy_out), .iz_curr_out(curr_iz_out),
        .next_x_out(next_next_x), .next_y_out(next_next_y), .next_z_out(next_next_z),
        .face_mask_out(face_mask), .primary_face_id_out(primary_face_id),
        .ray_id_out(result_ray_id),
        .out_of_bounds_out(out_of_bounds), .voxel_occupied_out(voxel_occupied),
        .step_valid_out(step_valid_out)
    );

endmodule

`default_nettype wire
