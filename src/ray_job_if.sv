`default_nettype none

// ============================================================
// ray_job_if — synthesis-safe copy (SIM $display guarded)
// ============================================================
module ray_job_if #(
    parameter int X_BITS         = 5,
    parameter int Y_BITS         = 5,
    parameter int Z_BITS         = 5,
    parameter int W              = 24,
    parameter int MAX_STEPS_BITS = 10
)(
    input  wire logic                 clk,
    input  wire logic                 rst_n,

    input  wire logic                 load_mode,

    input  wire logic                 job_valid,
    output logic                      job_ready,

    input  wire logic [X_BITS-1:0]    ix0,
    input  wire logic [Y_BITS-1:0]    iy0,
    input  wire logic [Z_BITS-1:0]    iz0,

    input  wire logic                 sx,
    input  wire logic                 sy,
    input  wire logic                 sz,

    input  wire logic [W-1:0]         next_x,
    input  wire logic [W-1:0]         next_y,
    input  wire logic [W-1:0]         next_z,

    input  wire logic [W-1:0]         inc_x,
    input  wire logic [W-1:0]         inc_y,
    input  wire logic [W-1:0]         inc_z,

    input  wire logic [MAX_STEPS_BITS-1:0] max_steps,

    input  wire logic                 job_done,

    output logic                      job_loaded,
    output logic                      job_active,

    output logic [X_BITS-1:0]         ix0_q,
    output logic [Y_BITS-1:0]         iy0_q,
    output logic [Z_BITS-1:0]         iz0_q,

    output logic                      sx_q,
    output logic                      sy_q,
    output logic                      sz_q,

    output logic [W-1:0]              next_x_q,
    output logic [W-1:0]              next_y_q,
    output logic [W-1:0]              next_z_q,

    output logic [W-1:0]              inc_x_q,
    output logic [W-1:0]              inc_y_q,
    output logic [W-1:0]              inc_z_q,

    output logic [MAX_STEPS_BITS-1:0] max_steps_q
);

    always_comb begin
        job_ready = (!load_mode) && (!job_active || job_done);
    end

    wire accept = job_valid && job_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            job_loaded  <= 1'b0;
            job_active  <= 1'b0;
            ix0_q       <= '0;
            iy0_q       <= '0;
            iz0_q       <= '0;
            sx_q        <= 1'b0;
            sy_q        <= 1'b0;
            sz_q        <= 1'b0;
            next_x_q    <= '0;
            next_y_q    <= '0;
            next_z_q    <= '0;
            inc_x_q     <= '0;
            inc_y_q     <= '0;
            inc_z_q     <= '0;
            max_steps_q <= '0;
        end else begin
            job_loaded <= accept;

            if (accept)      job_active <= 1'b1;
            else if (job_done) job_active <= 1'b0;

            if (accept) begin
                ix0_q       <= ix0;
                iy0_q       <= iy0;
                iz0_q       <= iz0;
                sx_q        <= sx;
                sy_q        <= sy;
                sz_q        <= sz;
                next_x_q    <= next_x;
                next_y_q    <= next_y;
                next_z_q    <= next_z;
                inc_x_q     <= inc_x;
                inc_y_q     <= inc_y;
                inc_z_q     <= inc_z;
                max_steps_q <= max_steps;
            end
        end
    end

`ifdef SIM
    initial begin
        if (X_BITS != 5 || Y_BITS != 5 || Z_BITS != 5)
            $display("INFO(ray_job_if): non-5-bit voxel indices X=%0d Y=%0d Z=%0d",
                     X_BITS, Y_BITS, Z_BITS);
    end
`endif

endmodule

`default_nettype wire
