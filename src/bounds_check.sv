`default_nettype none

// =============================================================================
// Module: bounds_check  (synthesis-safe copy)
// =============================================================================
module bounds_check #(
    parameter int COORD_W = 6,
    parameter int MAX_VAL = 31
)(
    input  wire logic [COORD_W-1:0] ix,
    input  wire logic [COORD_W-1:0] iy,
    input  wire logic [COORD_W-1:0] iz,
    output logic               out_of_bounds
);

    assign out_of_bounds = (ix > MAX_VAL[COORD_W-1:0]) |
                           (iy > MAX_VAL[COORD_W-1:0]) |
                           (iz > MAX_VAL[COORD_W-1:0]);

endmodule

`default_nettype wire
