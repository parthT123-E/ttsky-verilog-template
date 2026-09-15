`default_nettype none

// ============================================================
// Module: scene_loader_if  (synthesis-safe copy)
//   Passes the scene-load stream through to the voxel RAM's write port.
//   The stream is WORD-wide (WORD_BITS voxels per beat) because voxel_ram is
//   backed by a wide RAM; write_count therefore counts WORDS, not voxels.
// ============================================================
module scene_loader_if #(
    parameter int  ADDR_BITS      = 21,   // bit-address width of the store
    parameter int  WORD_BITS      = 32,   // voxels carried per beat
    parameter bit  ENABLE_COUNTER = 1'b1
)(
    input  wire logic             clk,
    input  wire logic             rst_n,

    input  wire logic             load_mode,
    input  wire logic             load_valid,
    output logic                  load_ready,
    input  wire logic [ADDR_BITS-$clog2(WORD_BITS)-1:0] load_addr,
    input  wire logic [WORD_BITS-1:0] load_data,

    output logic                  we,
    output logic [ADDR_BITS-$clog2(WORD_BITS)-1:0] waddr,
    output logic [WORD_BITS-1:0]  wdata,

    output logic [ADDR_BITS:0]    write_count,
    output logic                  load_complete
);

    localparam int WADDR_W     = ADDR_BITS - $clog2(WORD_BITS);
    localparam int TOTAL_WORDS = (1 << WADDR_W);

    always_comb begin
        load_ready = 1'b1;
        we    = load_mode && load_valid && load_ready;
        waddr = load_addr;
        wdata = load_data;
    end

    generate
        if (ENABLE_COUNTER) begin : g_counter
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    write_count   <= '0;
                    load_complete <= 1'b0;
                end else begin
                    if (!load_mode) begin
                        write_count   <= '0;
                        load_complete <= 1'b0;
                    end else if (we) begin
                        write_count <= write_count + 1'b1;
                        if (write_count == (TOTAL_WORDS[ADDR_BITS:0] - 1'b1))
                            load_complete <= 1'b1;
                    end
                end
            end
        end else begin : g_no_counter
            assign write_count   = '0;
            assign load_complete = 1'b0;
        end
    endgenerate

endmodule

`default_nettype wire
