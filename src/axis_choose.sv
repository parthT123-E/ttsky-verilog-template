`default_nettype none

// =============================================================================
// Module: axis_choose  (synthesis-safe copy — no timescale)
// Deterministically chooses the minimum of three unsigned values.
// Output is always 1-hot (exactly one axis selected).
// =============================================================================
module axis_choose #(
    parameter int W = 32
)(
    input  wire logic [W-1:0] a,
    input  wire logic [W-1:0] b,
    input  wire logic [W-1:0] c,
    output logic [2:0]   step_mask,
    output logic [1:0]   primary_sel
);

    always_comb begin
        if (a <= b && a <= c) begin
            primary_sel = 2'd0;
        end else if (b <= c) begin
            primary_sel = 2'd1;
        end else begin
            primary_sel = 2'd2;
        end
        step_mask = (3'b001 << primary_sel);
    end

endmodule

`default_nettype wire
