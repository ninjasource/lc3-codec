# lc3-codec

Low Complexity Communication Codec. This is an audio codec targeting `no_std` environments.

To start take a look at the `lc3_decoder.rs` and `lc3_encoder.rs` files.

## Lines of code

To check lines of code run:
```
loc --exclude ./tables/*
```

## Unit tests

The unit tests in place right now are primarily here to facilitate a major refactoring effort by helping me identify refactors with material changes to the input and output data. Basically I don't want to break things and this helps with that. More granular and useful tests will come as soon as the codebase stabilizes a little.

## Profiling

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

# License

``` 
Copyright 2022 David Haig

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```