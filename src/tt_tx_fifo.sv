`default_nettype none
// =============================================================================
// Module: tt_tx_fifo
// Description: 64-byte synchronous FIFO for transmit path.
//   push_en / push_data : write side (from command handler)
//   pop_en              : read side (HOST_RD_STB)
//   rdata               : current head byte (uo_out)
//   empty               : TX_VALID = !empty
//   full                : guard against overflow
// =============================================================================
module tt_tx_fifo (
    input  wire logic       clk,
    input  wire logic       rst_n,
    input  wire logic       push_en,
    input  wire logic [7:0] push_data,
    input  wire logic       pop_en,
    output logic [7:0] rdata,
    output logic       empty,
    output logic       full
);

    localparam int DEPTH = 64;
    localparam int PTR   = 6;   // log2(DEPTH)

    logic [7:0]     mem [0:DEPTH-1];
    logic [PTR-1:0] wr_ptr;
    logic [PTR-1:0] rd_ptr;
    logic [PTR:0]   count;      // PTR+1 bits to distinguish full vs empty

    assign empty = (count == 0);
    assign full  = (count == DEPTH[PTR:0]);
    assign rdata = mem[rd_ptr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            // Push: only if not full
            if (push_en && !full) begin
                mem[wr_ptr] <= push_data;
                wr_ptr      <= wr_ptr + 1'b1;
            end
            // Pop: only if not empty
            if (pop_en && !empty) begin
                rd_ptr <= rd_ptr + 1'b1;
            end
            // Update count
            case ({push_en && !full, pop_en && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: ;
            endcase
        end
    end

endmodule

`default_nettype wire
