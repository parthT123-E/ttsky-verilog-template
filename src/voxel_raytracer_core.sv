`default_nettype none

// =============================================================================
// Module: voxel_raytracer_core  (synthesis-safe copy — timescale removed)
// 5-stage DDA pipeline.  No changes to logic vs. original.
// =============================================================================
module voxel_raytracer_core #(
    parameter int W        = 32,
    parameter int COORD_W  = 6,
    parameter int MAX_VAL  = 31,
    parameter int ADDR_BITS = 15,
    parameter int WORD_BITS = 32,      // voxel RAM data width
    parameter int RAY_ID_WIDTH = 3
)(
    input  wire logic             clk,
    input  wire logic             rst_n,

    input  wire logic [COORD_W-1:0] ix_in,
    input  wire logic [COORD_W-1:0] iy_in,
    input  wire logic [COORD_W-1:0] iz_in,
    input  wire logic             sx_in,
    input  wire logic             sy_in,
    input  wire logic             sz_in,
    input  wire logic [W-1:0]     next_x_in,
    input  wire logic [W-1:0]     next_y_in,
    input  wire logic [W-1:0]     next_z_in,
    input  wire logic [W-1:0]     inc_x_in,
    input  wire logic [W-1:0]     inc_y_in,
    input  wire logic [W-1:0]     inc_z_in,
    input  wire logic [RAY_ID_WIDTH-1:0] ray_id_in,
    input  wire logic             step_valid_in,

    input  wire logic             load_mode,
    input  wire logic             load_valid,
    output logic                  load_ready,
    input  wire logic [ADDR_BITS-$clog2(WORD_BITS)-1:0] load_addr,
    input  wire logic [WORD_BITS-1:0] load_data,
    output logic [ADDR_BITS:0]    write_count,
    output logic                  load_complete,

    output logic [COORD_W-1:0]    ix_out,
    output logic [COORD_W-1:0]    iy_out,
    output logic [COORD_W-1:0]    iz_out,
    output logic [COORD_W-1:0]    ix_curr_out,
    output logic [COORD_W-1:0]    iy_curr_out,
    output logic [COORD_W-1:0]    iz_curr_out,
    output logic [W-1:0]          next_x_out,
    output logic [W-1:0]          next_y_out,
    output logic [W-1:0]          next_z_out,
    output logic [2:0]            face_mask_out,
    output logic [2:0]            primary_face_id_out,
    output logic [RAY_ID_WIDTH-1:0] ray_id_out,
    output logic                  out_of_bounds_out,
    output logic                  voxel_occupied_out,
    output logic                  step_valid_out
);

    // =========================================================
    // Stage 1: Input registers
    // =========================================================
    logic [COORD_W-1:0] ix_s1, iy_s1, iz_s1;
    logic         sx_s1, sy_s1, sz_s1;
    logic [W-1:0] next_x_s1, next_y_s1, next_z_s1;
    logic [W-1:0] inc_x_s1, inc_y_s1, inc_z_s1;
    logic [RAY_ID_WIDTH-1:0] ray_id_s1;
    logic         valid_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ix_s1 <= '0; iy_s1 <= '0; iz_s1 <= '0;
            sx_s1 <= '0; sy_s1 <= '0; sz_s1 <= '0;
            next_x_s1 <= '0; next_y_s1 <= '0; next_z_s1 <= '0;
            inc_x_s1  <= '0; inc_y_s1  <= '0; inc_z_s1  <= '0;
            ray_id_s1 <= '0;
            valid_s1  <= '0;
        end else begin
            ix_s1 <= ix_in; iy_s1 <= iy_in; iz_s1 <= iz_in;
            sx_s1 <= sx_in; sy_s1 <= sy_in; sz_s1 <= sz_in;
            next_x_s1 <= next_x_in; next_y_s1 <= next_y_in; next_z_s1 <= next_z_in;
            inc_x_s1  <= inc_x_in;  inc_y_s1  <= inc_y_in;  inc_z_s1  <= inc_z_in;
            ray_id_s1 <= ray_id_in;
            valid_s1  <= step_valid_in;
        end
    end

    // =========================================================
    // Stage 2: axis_choose (comb) + register
    // =========================================================
    logic [2:0] step_mask_s2;
    logic [1:0] primary_sel_s2;

    axis_choose #(.W(W)) u_axis_choose (
        .a(next_x_s1), .b(next_y_s1), .c(next_z_s1),
        .step_mask(step_mask_s2), .primary_sel(primary_sel_s2)
    );

    logic [COORD_W-1:0] ix_s2, iy_s2, iz_s2;
    logic         sx_s2, sy_s2, sz_s2;
    logic [W-1:0] next_x_s2, next_y_s2, next_z_s2;
    logic [W-1:0] inc_x_s2, inc_y_s2, inc_z_s2;
    logic [2:0]   step_mask_s2_q;
    logic [1:0]   primary_sel_s2_q;
    logic [RAY_ID_WIDTH-1:0] ray_id_s2;
    logic         valid_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ix_s2 <= '0; iy_s2 <= '0; iz_s2 <= '0;
            sx_s2 <= '0; sy_s2 <= '0; sz_s2 <= '0;
            next_x_s2 <= '0; next_y_s2 <= '0; next_z_s2 <= '0;
            inc_x_s2  <= '0; inc_y_s2  <= '0; inc_z_s2  <= '0;
            step_mask_s2_q <= '0; primary_sel_s2_q <= '0;
            ray_id_s2 <= '0;
            valid_s2 <= '0;
        end else begin
            ix_s2 <= ix_s1; iy_s2 <= iy_s1; iz_s2 <= iz_s1;
            sx_s2 <= sx_s1; sy_s2 <= sy_s1; sz_s2 <= sz_s1;
            next_x_s2 <= next_x_s1; next_y_s2 <= next_y_s1; next_z_s2 <= next_z_s1;
            inc_x_s2  <= inc_x_s1;  inc_y_s2  <= inc_y_s1;  inc_z_s2  <= inc_z_s1;
            step_mask_s2_q   <= step_mask_s2;
            primary_sel_s2_q <= primary_sel_s2;
            ray_id_s2 <= ray_id_s1;
            valid_s2 <= valid_s1;
        end
    end

    // =========================================================
    // Stage 3: step_update (comb) + register
    // =========================================================
    logic [COORD_W-1:0] ix_next_s3, iy_next_s3, iz_next_s3;
    logic [W-1:0] next_x_next_s3, next_y_next_s3, next_z_next_s3;
    logic [2:0]   face_mask_s3, primary_face_id_s3;

    step_update #(.W(W), .COORD_W(COORD_W)) u_step_update (
        .ix(ix_s2), .iy(iy_s2), .iz(iz_s2),
        .sx(sx_s2), .sy(sy_s2), .sz(sz_s2),
        .next_x(next_x_s2), .next_y(next_y_s2), .next_z(next_z_s2),
        .inc_x(inc_x_s2),   .inc_y(inc_y_s2),   .inc_z(inc_z_s2),
        .step_mask(step_mask_s2_q), .primary_sel(primary_sel_s2_q),
        .ix_next(ix_next_s3), .iy_next(iy_next_s3), .iz_next(iz_next_s3),
        .next_x_next(next_x_next_s3), .next_y_next(next_y_next_s3), .next_z_next(next_z_next_s3),
        .face_mask(face_mask_s3), .primary_face_id(primary_face_id_s3)
    );

    logic [COORD_W-1:0] ix_s3_curr, iy_s3_curr, iz_s3_curr;
    logic [COORD_W-1:0] ix_s3, iy_s3, iz_s3;
    logic [W-1:0] next_x_s3, next_y_s3, next_z_s3;
    logic [2:0]   face_mask_s3_q, primary_face_id_s3_q;
    logic [RAY_ID_WIDTH-1:0] ray_id_s3;
    logic         valid_s3;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ix_s3_curr <= '0; iy_s3_curr <= '0; iz_s3_curr <= '0;
            ix_s3 <= '0; iy_s3 <= '0; iz_s3 <= '0;
            next_x_s3 <= '0; next_y_s3 <= '0; next_z_s3 <= '0;
            face_mask_s3_q <= '0; primary_face_id_s3_q <= '0;
            ray_id_s3 <= '0;
            valid_s3 <= '0;
        end else begin
            ix_s3_curr <= ix_s2; iy_s3_curr <= iy_s2; iz_s3_curr <= iz_s2;
            ix_s3 <= ix_next_s3; iy_s3 <= iy_next_s3; iz_s3 <= iz_next_s3;
            next_x_s3 <= next_x_next_s3; next_y_s3 <= next_y_next_s3; next_z_s3 <= next_z_next_s3;
            face_mask_s3_q       <= face_mask_s3;
            primary_face_id_s3_q <= primary_face_id_s3;
            ray_id_s3 <= ray_id_s2;
            valid_s3 <= valid_s2;
        end
    end

    // =========================================================
    // Stage 4: bounds_check + voxel_addr_map (comb) + register
    // =========================================================
    localparam int ADDR_COORD_W = ADDR_BITS / 3;
    localparam int MAP_ADDR_BITS = ADDR_COORD_W * 3;
    logic                 out_of_bounds_s4;
    logic [MAP_ADDR_BITS-1:0] voxel_addr_full_s4;
    logic [ADDR_BITS-1:0] voxel_addr_s4;

    bounds_check #(.COORD_W(COORD_W), .MAX_VAL(MAX_VAL)) u_bounds_check (
        .ix(ix_s3), .iy(iy_s3), .iz(iz_s3),
        .out_of_bounds(out_of_bounds_s4)
    );

    voxel_addr_map #(.X_BITS(ADDR_COORD_W), .Y_BITS(ADDR_COORD_W), .Z_BITS(ADDR_COORD_W), .MAP_ZYX(1'b1)) u_voxel_addr_map (
        .x(ix_s3_curr[ADDR_COORD_W-1:0]), .y(iy_s3_curr[ADDR_COORD_W-1:0]), .z(iz_s3_curr[ADDR_COORD_W-1:0]),
        .addr(voxel_addr_full_s4)
    );

    assign voxel_addr_s4 = voxel_addr_full_s4[ADDR_BITS-1:0];

    logic [COORD_W-1:0]   ix_s4, iy_s4, iz_s4;
    logic [W-1:0]         next_x_s4, next_y_s4, next_z_s4;
    logic [COORD_W-1:0]   ix_s4_curr, iy_s4_curr, iz_s4_curr;
    logic [2:0]           face_mask_s4, primary_face_id_s4;
    logic                 out_of_bounds_s4_q;
    logic [ADDR_BITS-1:0] voxel_addr_s4_q;
    logic [RAY_ID_WIDTH-1:0] ray_id_s4;
    logic                 valid_s4;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ix_s4 <= '0; iy_s4 <= '0; iz_s4 <= '0;
            ix_s4_curr <= '0; iy_s4_curr <= '0; iz_s4_curr <= '0;
            next_x_s4 <= '0; next_y_s4 <= '0; next_z_s4 <= '0;
            face_mask_s4 <= '0; primary_face_id_s4 <= '0;
            out_of_bounds_s4_q <= '0; voxel_addr_s4_q <= '0;
            ray_id_s4 <= '0;
            valid_s4 <= '0;
        end else begin
            ix_s4 <= ix_s3; iy_s4 <= iy_s3; iz_s4 <= iz_s3;
            ix_s4_curr <= ix_s3_curr; iy_s4_curr <= iy_s3_curr; iz_s4_curr <= iz_s3_curr;
            next_x_s4 <= next_x_s3; next_y_s4 <= next_y_s3; next_z_s4 <= next_z_s3;
            face_mask_s4     <= face_mask_s3_q;
            primary_face_id_s4 <= primary_face_id_s3_q;
            out_of_bounds_s4_q <= out_of_bounds_s4;
            voxel_addr_s4_q    <= voxel_addr_s4;
            ray_id_s4 <= ray_id_s3;
            valid_s4 <= valid_s3;
        end
    end

    // =========================================================
    // Stage 5: voxel_ram + scene_loader_if
    // =========================================================
    localparam int RAM_WADDR_W = ADDR_BITS - $clog2(WORD_BITS);

    logic                   voxel_occupied_s5;
    logic                   we_ram;
    logic [RAM_WADDR_W-1:0] waddr_ram;
    logic [WORD_BITS-1:0]   wdata_ram;

    scene_loader_if #(
        .ADDR_BITS(ADDR_BITS), .WORD_BITS(WORD_BITS), .ENABLE_COUNTER(1'b1)
    ) u_scene_loader (
        .clk(clk), .rst_n(rst_n),
        .load_mode(load_mode), .load_valid(load_valid),
        .load_ready(load_ready), .load_addr(load_addr), .load_data(load_data),
        .we(we_ram), .waddr(waddr_ram), .wdata(wdata_ram),
        .write_count(write_count), .load_complete(load_complete)
    );

    voxel_ram #(
        .ADDR_BITS(ADDR_BITS), .WORD_BITS(WORD_BITS),
        .SYNC_READ(1'b1), .WRITE_FIRST(1'b1)
    ) u_voxel_ram (
        .clk(clk), .rst_n(rst_n),
        .raddr(voxel_addr_s4),  .rdata(voxel_occupied_s5),
        .we(we_ram), .waddr(waddr_ram), .wdata(wdata_ram)
    );

    logic [COORD_W-1:0] ix_s5, iy_s5, iz_s5;
    logic [COORD_W-1:0] ix_s5_curr, iy_s5_curr, iz_s5_curr;
    logic [W-1:0] next_x_s5, next_y_s5, next_z_s5;
    logic [2:0]   face_mask_s5, primary_face_id_s5;
    logic         out_of_bounds_s5;
    logic [RAY_ID_WIDTH-1:0] ray_id_s5;
    logic         valid_s5;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ix_s5 <= '0; iy_s5 <= '0; iz_s5 <= '0;
            ix_s5_curr <= '0; iy_s5_curr <= '0; iz_s5_curr <= '0;
            next_x_s5 <= '0; next_y_s5 <= '0; next_z_s5 <= '0;
            face_mask_s5 <= '0; primary_face_id_s5 <= '0;
            out_of_bounds_s5 <= '0; ray_id_s5 <= '0; valid_s5 <= '0;
        end else begin
            ix_s5 <= ix_s4; iy_s5 <= iy_s4; iz_s5 <= iz_s4;
            ix_s5_curr <= ix_s4_curr; iy_s5_curr <= iy_s4_curr; iz_s5_curr <= iz_s4_curr;
            next_x_s5 <= next_x_s4; next_y_s5 <= next_y_s4; next_z_s5 <= next_z_s4;
            face_mask_s5       <= face_mask_s4;
            primary_face_id_s5 <= primary_face_id_s4;
            out_of_bounds_s5   <= out_of_bounds_s4_q;
            ray_id_s5          <= ray_id_s4;
            valid_s5           <= valid_s4;
        end
    end

    // =========================================================
    // Outputs
    // =========================================================
    assign ix_out              = ix_s5;
    assign iy_out              = iy_s5;
    assign iz_out              = iz_s5;
    assign ix_curr_out         = ix_s5_curr;
    assign iy_curr_out         = iy_s5_curr;
    assign iz_curr_out         = iz_s5_curr;
    assign next_x_out          = next_x_s5;
    assign next_y_out          = next_y_s5;
    assign next_z_out          = next_z_s5;
    assign face_mask_out       = face_mask_s5;
    assign primary_face_id_out = primary_face_id_s5;
    assign ray_id_out          = ray_id_s5;
    assign out_of_bounds_out   = out_of_bounds_s5;
    assign voxel_occupied_out  = voxel_occupied_s5;
    assign step_valid_out      = valid_s5;

endmodule

`default_nettype wire
