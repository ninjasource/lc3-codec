# Profiling

To setup follow instructions here:
https://github.com/flamegraph-rs/flamegraph
Note that the debug symbols should be included in release mode and you should only profile release builds

```
# Cargo.toml
[profile.release]
debug = true
```

To run:

Use flamegraph directly:
```
# enable unprivileged profiling
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# record performance stats for the decode binary
flamegraph -o flamegraph.svg ./target/release/decode
flamegraph -o flamegraph.svg ./target/release/examples/spectral_noise_shaping_decode

# disable unprivileged profiling (for security reasons)
echo 3 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

or use cargo-flamegraph for more advanced features (andd no need to change `perf_event_paranoid`):
```
cargo flamegraph -c "record -e instructions -c 100 --call-graph lbr -g" --example decode --root

cargo flamegraph --bench modified_dct_decode --root -- --bench

cargo flamegraph -c "record -e instructions -c 100 --call-graph lbr -g" --example mdct_decode --root

cargo flamegraph -c "record -e instructions -c 100 --call-graph lbr -g" --example mdct_decode --root


```