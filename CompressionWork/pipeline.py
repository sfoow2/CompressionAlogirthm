import subprocess
import os
import zlib
import numpy as np
import constriction
from concurrent.futures import ThreadPoolExecutor

TRANSFORM_EXE = r"C:\Users\lukac\Documents\compressor\remake.exe"
WORK_DIR      = r"C:\Users\lukac\Documents\compressor"
ORIGINAL_PATH = r"C:\Users\lukac\Documents\compressor\archive9"

TMP_IN  = os.path.join(WORK_DIR, "pipe_in.bin")
TMP_OUT = os.path.join(WORK_DIR, "pipe_out.bin")


N_WORKERS = os.cpu_count() or 4


def ans2nd_compress(data):
    symbols = np.frombuffer(data, dtype=np.uint8).astype(np.int32)

    chunk   = max(1, (256 + N_WORKERS - 1) // N_WORKERS)
    batches = [range(i, min(i + chunk, 256)) for i in range(0, 256, chunk)]

    def process_batch(ctx_range):
        rows = []
        stream_total = 0
        for ctx in ctx_range:
            ctx_syms = symbols[1:][symbols[:-1] == ctx]
            ctx_raw  = (np.bincount(ctx_syms, minlength=256).astype(np.int32)
                        if len(ctx_syms) else np.zeros(256, dtype=np.int32))
            rows.append(ctx_raw)
            if len(ctx_syms):
                ctx_counts = ctx_raw.astype(np.float64) + 1
                ctx_model  = constriction.stream.model.Categorical(
                    ctx_counts / ctx_counts.sum(), perfect=False)
                coder = constriction.stream.stack.AnsCoder()
                coder.encode_reverse(ctx_syms.astype(np.int32), ctx_model)
                stream_total += len(coder.get_compressed().tobytes())
        return rows, stream_total

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_batch, b) for b in batches]
        all_rows     = []
        total_stream = 0
        for f in futures:
            rows, s = f.result()
            all_rows.extend(rows)
            total_stream += s

    tables_arr    = np.array(all_rows, dtype=np.int32)
    tables_deltas = np.diff(tables_arr, axis=1, prepend=tables_arr[:, :1]).astype(np.int32)
    table_overhead = len(zlib.compress(tables_deltas.tobytes(), level=9))

    return total_stream + table_overhead


def run_transform(input_path, output_path):
    result = subprocess.run(
        [TRANSFORM_EXE, input_path, output_path],
        capture_output=False, text=True
    )
    # parse HEADER_BYTES=N from the separate stderr-like line
    # (remake.c prints it to stdout)
    return result  # caller re-reads stdout via captured output


def run_transform_get_header(input_path, output_path):
    result = subprocess.run(
        [TRANSFORM_EXE, input_path, output_path],
        capture_output=True, text=True
    )
    print(result.stdout, end="")
    for line in result.stdout.splitlines():
        if line.startswith("HEADER_BYTES="):
            return int(line.split("=")[1])
    return 0


# ── main loop ────────────────────────────────────────────────────────────────

original      = open(ORIGINAL_PATH, "rb").read()
current       = original
cumulative_hdr = 0
round_num     = 0

print(f"Original: {len(original)} bytes\n{'='*60}")

while True:
    round_num += 1
    print(f"\n{'='*60}  ROUND {round_num}  {'='*60}")

    # write current data to temp file and run C transform
    with open(TMP_IN, "wb") as f:
        f.write(current)

    header_cost = run_transform_get_header(TMP_IN, TMP_OUT)
    transformed = open(TMP_OUT, "rb").read()

    # ANS-2nd-order compress the transformed data
    print("Running ANS-2nd-order compression...")
    compressed_size = ans2nd_compress(transformed)
    total_cost      = compressed_size + cumulative_hdr + header_cost

    print(f"\nRound {round_num} summary:")
    print(f"  transformed : {len(transformed):>10} bytes")
    print(f"  ANS-2nd     : {compressed_size:>10} bytes  ({100*compressed_size/len(original):.2f}% of original)")
    print(f"  header this : {header_cost:>10} bytes")
    print(f"  cumul. hdr  : {cumulative_hdr:>10} bytes")
    print(f"  TOTAL so far: {total_cost:>10} bytes  ({100*total_cost/len(original):.2f}% of original)")

    prev_total = len(current) + cumulative_hdr
    if total_cost >= prev_total:
        print(f"\n  No improvement ({total_cost} >= {prev_total}) — stopping.")
        break

    # for next round, the "data" is the ANS output bytes.
    # since we only measure size (not actual bytes), we approximate with
    # random bytes of the right length — transform doesn't care about
    # content beyond the < 128 / >= 128 split, which is ~50/50 for random.
    # Use os.urandom so entropy is correct for the next transform pass.
    cumulative_hdr += header_cost
    current = os.urandom(compressed_size)   # surrogate for actual ANS output
    print(f"\n  -> next round input: {len(current)} bytes (simulated ANS output)")

print(f"\n{'='*60}")
print(f"Done after {round_num} round(s).")
print(f"Final total: {total_cost} bytes  ({100*total_cost/len(original):.2f}% of original)")
print(f"  breakdown: {compressed_size} (ANS) + {cumulative_hdr} (all headers)")
