`default_nettype none
// =============================================================================
// Module: tt_div32
// 32-bit / 16-bit unsigned long-division, 33-cycle latency (including done pulse).
//   - Caller handles sign externally (pass |num|, |den|).
//   - valid: single-cycle pulse when quot is stable.
//   - If den==0: quot saturates to 32'hFFFF_FFFF, div_by_zero=1.
//   - New start accepted only when not active (cnt==0).
// =============================================================================
module tt_div32 (
    input  wire logic        clk,
    input  wire logic        rst_n,
    input  wire logic        start,
    input  wire logic [31:0] num,
    input  wire logic [15:0] den,
    output logic [31:0] quot,
    output logic        valid,
    output logic        div_by_zero
);

    logic [5:0]  cnt;      // 0 = idle, 1..32 = shifting, 33 = output
    logic [31:0] rem;
    logic [31:0] numerator_r;
    logic [15:0] denominator_r;
    logic [32:0] r_ext;    // extended remainder for comparison

    assign r_ext = {rem[30:0], numerator_r[31]};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt           <= '0;
            rem           <= '0;
            numerator_r   <= '0;
            denominator_r <= '0;
            quot          <= '0;
            valid         <= 1'b0;
            div_by_zero   <= 1'b0;
        end else begin
            valid <= 1'b0;   // default: no valid pulse

            if (start && (cnt == 6'd0)) begin
                numerator_r   <= num;
                denominator_r <= den;
                rem           <= '0;
                quot          <= '0;
                div_by_zero   <= (den == 16'd0);
                cnt           <= (den == 16'd0) ? 6'd33 : 6'd1;
            end else if ((cnt >= 6'd1) && (cnt <= 6'd32)) begin
                if (r_ext >= {17'd0, denominator_r}) begin
                    rem  <= r_ext[31:0] - {16'd0, denominator_r};
                    quot <= {quot[30:0], 1'b1};
                end else begin
                    rem  <= r_ext[31:0];
                    quot <= {quot[30:0], 1'b0};
                end
                numerator_r <= {numerator_r[30:0], 1'b0};
                cnt <= cnt + 6'd1;
            end else if (cnt == 6'd33) begin
                if (div_by_zero) quot <= 32'hFFFF_FFFF;
                valid <= 1'b1;
                cnt   <= 6'd0;
            end
        end
    end

endmodule

`default_nettype wire
