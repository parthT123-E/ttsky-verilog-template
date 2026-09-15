// =============================================================================
// TinyTapeout submission wrapper for the ASIC voxel ray tracer.
//
// This is the template-facing top module. It keeps the TinyTapeout pin contract
// at the boundary and delegates the streaming QSPI protocol to the project
// implementation wrapper.
// =============================================================================
`default_nettype none

module tt_um_ttsky_asic_raytracer (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // Bidirectional IO input path
    output wire [7:0] uio_out,  // Bidirectional IO output path
    output wire [7:0] uio_oe,   // Bidirectional IO enables: 0=input, 1=output
    input  wire       ena,      // Design enable, normally held high by TinyTapeout
    input  wire       clk,      // Clock
    input  wire       rst_n     // Active-low reset
);

    wire [7:0] core_uo_out;
    wire [7:0] core_uio_out;
    wire [7:0] core_uio_oe;

    qspi_stream_raytracer u_project (
        .ui_in(ui_in),
        .uo_out(core_uo_out),
        .uio_in(uio_in),
        .uio_out(core_uio_out),
        .uio_oe(core_uio_oe),
        .ena(ena),
        .clk(clk),
        .rst_n(rst_n)
    );

    // Hold outputs quiet when the wrapper is disabled.
    assign uo_out  = ena ? core_uo_out  : 8'h00;
    assign uio_out = ena ? core_uio_out : 8'h00;
    assign uio_oe  = ena ? core_uio_oe  : 8'h00;

endmodule

`default_nettype wire
