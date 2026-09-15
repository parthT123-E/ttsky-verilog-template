`default_nettype none

// =============================================================================
// Module: voxel_addr_map  (synthesis-safe copy)
// =============================================================================
module voxel_addr_map #(
    parameter int X_BITS = 5,
    parameter int Y_BITS = 5,
    parameter int Z_BITS = 5,
    parameter bit MAP_ZYX = 1'b1
)(
    input  wire logic [X_BITS-1:0] x,
    input  wire logic [Y_BITS-1:0] y,
    input  wire logic [Z_BITS-1:0] z,
    output logic [X_BITS+Y_BITS+Z_BITS-1:0] addr
);

`ifdef SIM
    initial begin
        if (X_BITS < 1 || X_BITS > 16) $error("voxel_addr_map: X_BITS=%0d out of range", X_BITS);
        if (Y_BITS < 1 || Y_BITS > 16) $error("voxel_addr_map: Y_BITS=%0d out of range", Y_BITS);
        if (Z_BITS < 1 || Z_BITS > 16) $error("voxel_addr_map: Z_BITS=%0d out of range", Z_BITS);
    end
`endif

    always_comb begin
        if (MAP_ZYX) addr = {z, y, x};
        else         addr = {x, y, z};
    end

endmodule

`default_nettype wire
