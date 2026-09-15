# TT ASIC Voxel Ray Tracer — timing constraints
# Clock: 40 MHz → 25 ns period (matches CLOCK_PERIOD in config.json)

create_clock -name clk -period 25 [get_ports clk]

# Input/output delays are an absolute off-chip budget, so they stay at 4 ns
# rather than scaling with the period: slowing the clock is meant to hand the
# extra 5 ns to internal logic, not to the pads. (At 25 ns this is 16 %, not
# the 20 % it was at 50 MHz.)
set_input_delay -clock clk -max 4 [get_ports {ui_in[*]}]
set_input_delay -clock clk -max 4 [get_ports {uio_in[*]}]
set_input_delay -clock clk -max 4 [get_ports ena]
set_input_delay -clock clk -max 4 [get_ports rst_n]

# Output delays
set_output_delay -clock clk -max 4 [get_ports {uo_out[*]}]
set_output_delay -clock clk -max 4 [get_ports {uio_out[*]}]
set_output_delay -clock clk -max 4 [get_ports {uio_oe[*]}]

# rst_n is asynchronous — cut timing paths from it
set_false_path -from [get_ports rst_n]
