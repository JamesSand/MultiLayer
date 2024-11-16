
log_path = "cpu_log.txt"

with open(log_path, "r") as f:
    for line in f:
        line = line.rstrip("\n")
        line_list = line.split("\t")
        dim, relative_err, actual_time, our_time = line_list
        dim = int(dim.split(":")[-1])
        relative_err = float(relative_err.split(":")[-1])
        actual_time = float(actual_time.split(":")[-1].rstrip("s"))
        our_time = float(our_time.split(":")[-1].rstrip("s"))

        length_n = 10 * 2**dim

        accelerate_ratio = actual_time / our_time

        # in percetage
        relative_err *= 100

        # in ms
        actual_time *= 1000
        our_time *= 1000

        # output_str = f"|{dim}|10*2**d={length_n}|{relative_err:.3f}|{actual_time:.3f}|{our_time:.3f}|{accelerate_ratio:.3f}|"

        # print(output_str)

        print(f"dim:{dim}\tseq_len:{length_n}\trelative_error:{relative_err:.3f}\tactual:{actual_time:.3f}\tour:{our_time:.3f}\tspeedup:{accelerate_ratio:.3f}")



