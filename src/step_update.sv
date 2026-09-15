`default_nettype none

// =============================================================================
// Module: step_update  (synthesis-safe copy)
// =============================================================================
module step_update #(
    parameter int W      = 32,
    parameter int COORD_W = 6   // coordinate bit width (use GRID_COORD_BITS)
)(
    input  wire logic [COORD_W-1:0] ix,
    input  wire logic [COORD_W-1:0] iy,
    input  wire logic [COORD_W-1:0] iz,
    input  wire logic         sx,
    input  wire logic         sy,
    input  wire logic         sz,
    input  wire logic [W-1:0] next_x,
    input  wire logic [W-1:0] next_y,
    input  wire logic [W-1:0] next_z,
    input  wire logic [W-1:0] inc_x,
    input  wire logic [W-1:0] inc_y,
    input  wire logic [W-1:0] inc_z,
    input  wire logic [2:0]   step_mask,
    input  wire logic [1:0]   primary_sel,
    output logic [COORD_W-1:0] ix_next,
    output logic [COORD_W-1:0] iy_next,
    output logic [COORD_W-1:0] iz_next,
    output logic [W-1:0] next_x_next,
    output logic [W-1:0] next_y_next,
    output logic [W-1:0] next_z_next,
    output logic [2:0]   face_mask,
    output logic [2:0]   primary_face_id
);

    logic signed [COORD_W:0] step_xv, step_yv, step_zv;  // (COORD_W+1) bits for ±1

    always_comb begin
        step_xv = sx ? {{COORD_W{1'b0}}, 1'b1} : {1'b1, {COORD_W{1'b1}}};  // +1 / -1
        step_yv = sy ? {{COORD_W{1'b0}}, 1'b1} : {1'b1, {COORD_W{1'b1}}};
        step_zv = sz ? {{COORD_W{1'b0}}, 1'b1} : {1'b1, {COORD_W{1'b1}}};
    end

    always_comb begin
        ix_next = step_mask[0] ? ix + step_xv[COORD_W-1:0] : ix;
        iy_next = step_mask[1] ? iy + step_yv[COORD_W-1:0] : iy;
        iz_next = step_mask[2] ? iz + step_zv[COORD_W-1:0] : iz;

        next_x_next = step_mask[0] ? next_x + inc_x : next_x;
        next_y_next = step_mask[1] ? next_y + inc_y : next_y;
        next_z_next = step_mask[2] ? next_z + inc_z : next_z;
    end

    assign face_mask = step_mask;

    always_comb begin
        case (primary_sel)
            2'd0: primary_face_id = sx ? 3'd0 : 3'd1;
            2'd1: primary_face_id = sy ? 3'd2 : 3'd3;
            2'd2: primary_face_id = sz ? 3'd4 : 3'd5;
            default: primary_face_id = 3'd0;
        endcase
    end

endmodule

`default_nettype wire
