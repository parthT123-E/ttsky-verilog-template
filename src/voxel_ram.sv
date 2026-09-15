`default_nettype none

// ============================================================
// Module: voxel_ram
//   Single-bit occupancy store holding 2^ADDR_BITS voxels, backed by a RAM
//   that is WORD_BITS wide and 2^(ADDR_BITS-WORD_AW) deep.
//
//   The RAM is WIDE rather than 1 bit wide because the Block Memory Generator
//   caps a port at 2^20 addresses: a 128^3 grid needs 2^21 voxels, which will
//   not fit as 1x2M but fits comfortably as 32x64K. Width also matches the
//   32-bit scene-pipe word, so scene loading writes whole words instead of
//   serialising each one into 32 single-bit writes.
//
//   READ  side keeps a one-voxel interface: raddr is a BIT address. The low
//         WORD_AW bits select a bit out of the fetched word, and are delayed
//         to line up with the RAM's two-cycle read latency.
//   WRITE side is word-wide: waddr is a WORD address and wdata a whole word.
//         Single-bit writes are deliberately not supported -- they would need
//         a read-modify-write, and nothing needs them.
//
//   The `initial` zero-fill is SIM-only (guarded with `ifdef SIM). On real
//   hardware the scene load writes every word once (see scene_loader_if).
//
//   SYNTHESIS: backed by the blk_mem_gen_0 Vivado IP, which must be
//   regenerated (IP catalog, not just this file) whenever the geometry
//   changes -- its port widths are fixed by that IP customization, not
//   parameterized at the Verilog level.
//
//   Required IP configuration -- the instantiation below drives exactly these
//   pins, so any option that ADDS a pin will leave it floating and fail Opt
//   Design with an "unconnected pin" warning followed by a LUT with a missing
//   input inside the IP's address decoder:
//     - native Simple Dual Port RAM, common clock
//     - Port A: write, 32 bits wide x 65536 deep
//     - Port B: read,  32 bits wide x 65536 deep
//     - Enable Port Type: "Always Enabled" on BOTH ports
//         (anything else exposes ena/enb, which are not connected here)
//     - Byte write enable: OFF (otherwise wea widens past one bit)
//     - Port B read latency: 2 clocks -- the bit-select delay below is
//         hardwired to that figure
// ============================================================
module voxel_ram #(
    parameter int  ADDR_BITS   = 21,   // bit-address width (voxels = 2^this)
    parameter int  WORD_BITS   = 32,   // RAM data width
    parameter bit  SYNC_READ   = 1'b1,
    parameter bit  WRITE_FIRST = 1'b1
)(
    input  wire logic             clk,
    input  wire logic             rst_n,

    input  wire logic [ADDR_BITS-1:0] raddr,   // BIT address
    output logic                  rdata,       // the selected voxel

    input  wire logic             we,
    input  wire logic [ADDR_BITS-$clog2(WORD_BITS)-1:0] waddr,  // WORD address
    input  wire logic [WORD_BITS-1:0] wdata
);

    localparam int WORD_AW   = $clog2(WORD_BITS);        // 5 for 32-bit words
    localparam int WADDR_W   = ADDR_BITS - WORD_AW;      // word-address width
    localparam int DEPTH     = 1 << WADDR_W;             // words in the RAM

    wire [WADDR_W-1:0] rword_addr = raddr[ADDR_BITS-1:WORD_AW];
    wire [WORD_AW-1:0] rbit_sel   = raddr[WORD_AW-1:0];

`ifdef SYNTHESIS
    // Scene loading and tracing are mutually exclusive, so a Port A write and
    // Port B read never intentionally target the RAM at the same time.
    logic [WORD_BITS-1:0] bram_doutb;

    blk_mem_gen_0 u_voxel_bram (
        .clka  (clk),
        .wea   ({we}),
        .addra (waddr),
        .dina  (wdata),
        .clkb  (clk),
        .addrb (rword_addr),
        .doutb (bram_doutb)
    );

    // The fetched word arrives two clocks after addrb, so the bit select has
    // to be delayed by the same two clocks to pick the right voxel out of it.
    logic [WORD_AW-1:0] sel_d1, sel_d2;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sel_d1 <= '0;
            sel_d2 <= '0;
        end else begin
            sel_d1 <= rbit_sel;
            sel_d2 <= sel_d1;
        end
    end

    always_comb rdata = bram_doutb[sel_d2];

`else
    // Portable behavioral model used by Icarus and Vivado behavioral
    // simulation. Its two-cycle read timing matches blk_mem_gen_0.
    logic [WORD_BITS-1:0] mem [0:DEPTH-1];

`ifdef SIM
    // SIM-only: initialise all words to 0 so unloaded voxels are transparent.
    integer _ii;
    initial begin
        for (_ii = 0; _ii < DEPTH; _ii = _ii + 1)
            mem[_ii] = '0;
    end
`endif

    // Write (sync), a whole word at a time
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    generate
        if (SYNC_READ) begin : g_sync_read
            logic [WADDR_W-1:0]   rword_addr_q;
            logic [WORD_AW-1:0]   rbit_sel_q, rbit_sel_q2;
            logic [WORD_BITS-1:0] word_q;

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    rword_addr_q <= '0;
                    rbit_sel_q   <= '0;
                    rbit_sel_q2  <= '0;
                end else begin
                    rword_addr_q <= rword_addr;
                    rbit_sel_q   <= rbit_sel;
                    rbit_sel_q2  <= rbit_sel_q;
                end
            end

            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    word_q <= '0;
                end else begin
                    // Write-first bypass: a word written this cycle to the
                    // address being read must be seen by the read.
                    if (WRITE_FIRST && we && (waddr == rword_addr_q))
                        word_q <= wdata;
                    else
                        word_q <= mem[rword_addr_q];
                end
            end

            always_comb rdata = word_q[rbit_sel_q2];
        end else begin : g_comb_read
            always_comb rdata = mem[rword_addr][rbit_sel];
        end
    endgenerate
`endif

endmodule

`default_nettype wire
