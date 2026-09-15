`default_nettype none

// =============================================================================
// Module: step_control_fsm  (synthesis-safe copy)
// Uses active-high reset (keep same as original; raytracer_top inverts rst_n).
// =============================================================================
module step_control_fsm #(
    parameter int X_BITS           = 5,
    parameter int Y_BITS           = 5,
    parameter int Z_BITS           = 5,
    parameter int TIMER_WIDTH      = 32,
    parameter int STEP_COUNT_WIDTH = 16
)(
    input  wire logic                     clock,
    input  wire logic                     reset,

    input  wire logic                     job_loaded,
    output logic                          ready,
    output logic                          active,

    input  wire logic [X_BITS-1:0]        job_init_x,
    input  wire logic [Y_BITS-1:0]        job_init_y,
    input  wire logic [Z_BITS-1:0]        job_init_z,
    input  wire logic [TIMER_WIDTH-1:0]   job_timer_x,
    input  wire logic [TIMER_WIDTH-1:0]   job_timer_y,
    input  wire logic [TIMER_WIDTH-1:0]   job_timer_z,
    input  wire logic [STEP_COUNT_WIDTH-1:0] max_steps,

    input  wire logic                     solid_bit,
    input  wire logic                     solid_valid,
    input  wire logic                     out_of_bounds,

    input  wire logic [X_BITS-1:0]        pipeline_next_x,
    input  wire logic [Y_BITS-1:0]        pipeline_next_y,
    input  wire logic [Z_BITS-1:0]        pipeline_next_z,
    input  wire logic [TIMER_WIDTH-1:0]   pipeline_next_timer_x,
    input  wire logic [TIMER_WIDTH-1:0]   pipeline_next_timer_y,
    input  wire logic [TIMER_WIDTH-1:0]   pipeline_next_timer_z,
    input  wire logic [2:0]               pipeline_face_id,

    output logic [X_BITS-1:0]             current_voxel_x,
    output logic [Y_BITS-1:0]             current_voxel_y,
    output logic [Z_BITS-1:0]             current_voxel_z,
    output logic [TIMER_WIDTH-1:0]        current_timer_x,
    output logic [TIMER_WIDTH-1:0]        current_timer_y,
    output logic [TIMER_WIDTH-1:0]        current_timer_z,
    output logic [STEP_COUNT_WIDTH-1:0]   steps_taken,

    output logic                          done,
    output logic                          hit,
    output logic                          timeout,

    output logic [X_BITS-1:0]             hit_voxel_x,
    output logic [Y_BITS-1:0]             hit_voxel_y,
    output logic [Z_BITS-1:0]             hit_voxel_z,
    output logic [2:0]                    face_id
);

    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        INIT    = 2'b01,
        RUNNING = 2'b10,
        FINISH  = 2'b11
    } state_t;

    state_t current_state, next_state;

    logic [X_BITS-1:0]           voxel_x_reg, voxel_y_reg, voxel_z_reg;
    logic [TIMER_WIDTH-1:0]      timer_x_reg, timer_y_reg, timer_z_reg;
    logic [STEP_COUNT_WIDTH-1:0] step_counter, max_steps_reg;
    logic [X_BITS-1:0]           hit_x_reg, hit_y_reg, hit_z_reg;
    logic [2:0]                  face_reg;
    logic                        hit_flag, timeout_flag, bounds_flag;
    logic [X_BITS-1:0]           voxel_x_prev, voxel_y_prev, voxel_z_prev;

    // State register
    always_ff @(posedge clock or posedge reset) begin
        if (reset) current_state <= IDLE;
        else        current_state <= next_state;
    end

    // Next state
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE:    if (job_loaded) next_state = INIT;
            INIT:    next_state = RUNNING;
            RUNNING: begin
                if (solid_valid) begin
                    if (hit_flag || timeout_flag || bounds_flag ||
                        solid_bit || out_of_bounds || (step_counter >= max_steps_reg))
                        next_state = FINISH;
                end
            end
            FINISH:  next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // Datapath
    always_ff @(posedge clock or posedge reset) begin
        if (reset) begin
            voxel_x_reg  <= '0; voxel_y_reg  <= '0; voxel_z_reg  <= '0;
            timer_x_reg  <= '0; timer_y_reg  <= '0; timer_z_reg  <= '0;
            step_counter <= '0; max_steps_reg <= '0;
            hit_x_reg    <= '0; hit_y_reg    <= '0; hit_z_reg    <= '0;
            face_reg     <= '0;
            hit_flag     <= 1'b0; timeout_flag <= 1'b0; bounds_flag <= 1'b0;
            voxel_x_prev <= '0; voxel_y_prev <= '0; voxel_z_prev <= '0;
        end else begin
            case (current_state)
                IDLE: begin
                    hit_flag     <= 1'b0;
                    timeout_flag <= 1'b0;
                    bounds_flag  <= 1'b0;
                    step_counter <= '0;
                    voxel_x_prev <= '0; voxel_y_prev <= '0; voxel_z_prev <= '0;
                end
                INIT: begin
                    voxel_x_reg   <= job_init_x;
                    voxel_y_reg   <= job_init_y;
                    voxel_z_reg   <= job_init_z;
                    timer_x_reg   <= job_timer_x;
                    timer_y_reg   <= job_timer_y;
                    timer_z_reg   <= job_timer_z;
                    max_steps_reg <= max_steps;
                    step_counter  <= '0;
                end
                RUNNING: begin
                    voxel_x_prev <= voxel_x_reg;
                    voxel_y_prev <= voxel_y_reg;
                    voxel_z_prev <= voxel_z_reg;
                    if (solid_valid) begin
                        if (solid_bit) begin
                            hit_flag  <= 1'b1;
                            hit_x_reg <= voxel_x_reg;
                            hit_y_reg <= voxel_y_reg;
                            hit_z_reg <= voxel_z_reg;
                        end
                        if (step_counter >= max_steps_reg) timeout_flag <= 1'b1;
                        if (out_of_bounds)                  bounds_flag  <= 1'b1;
                        if (!(hit_flag || timeout_flag || bounds_flag ||
                              solid_bit || out_of_bounds || (step_counter >= max_steps_reg))) begin
                            voxel_x_reg <= pipeline_next_x;
                            voxel_y_reg <= pipeline_next_y;
                            voxel_z_reg <= pipeline_next_z;
                            timer_x_reg <= pipeline_next_timer_x;
                            timer_y_reg <= pipeline_next_timer_y;
                            timer_z_reg <= pipeline_next_timer_z;
                            face_reg    <= pipeline_face_id;
                            if ((voxel_x_reg != voxel_x_prev) ||
                                (voxel_y_reg != voxel_y_prev) ||
                                (voxel_z_reg != voxel_z_prev))
                                step_counter <= step_counter + 1'b1;
                        end
                    end
                end
                FINISH: ; // Hold
                default: ;
            endcase
        end
    end

    // Outputs
    always_comb begin
        ready           = (current_state == IDLE);
        active          = (current_state == RUNNING);
        done            = (current_state == FINISH);
        hit             = hit_flag;
        timeout         = timeout_flag;
        current_voxel_x = voxel_x_reg;
        current_voxel_y = voxel_y_reg;
        current_voxel_z = voxel_z_reg;
        current_timer_x = timer_x_reg;
        current_timer_y = timer_y_reg;
        current_timer_z = timer_z_reg;
        steps_taken     = step_counter;
        hit_voxel_x     = hit_x_reg;
        hit_voxel_y     = hit_y_reg;
        hit_voxel_z     = hit_z_reg;
        face_id         = face_reg;
    end

endmodule

`default_nettype wire
